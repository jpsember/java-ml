package ml.yolo;

import java.util.Arrays;
import java.util.List;

import static js.base.Tools.*;
import static ml.NetworkUtil.*;
import static ml.yolo.YoloUtil.*;

import js.data.DataUtil;
import js.geometry.FPoint;
import js.geometry.FRect;
import js.geometry.IPoint;
import js.geometry.IRect;
import js.geometry.MyMath;
import js.graphics.PolygonElement;
import js.graphics.RectElement;
import js.graphics.ScriptElement;
import js.graphics.ScriptUtil;
import js.graphics.gen.ElementProperties;
import js.graphics.gen.Script;
import js.json.JSMap;
import ml.ModelWrapper;
import ml.NetworkAnalyzer;
import gen.ImageSetInfo;
import gen.Layer;
import gen.LayerType;
import gen.PlotInferenceResultsConfig;
import gen.Vol;
import gen.Yolo;
import ml.NetworkUtil;
import ml.VolumeUtil;

public final class YoloModelWrapper extends ModelWrapper<Yolo> {

  @Override
  public void extractShapes(Script script, List<ScriptElement> target) {
    assertNoMixing(script);
    target.addAll(ScriptUtil.polygonElements(script));
    for (RectElement b : ScriptUtil.rectElements(script)) {
      target.add(b);
    }
  }

  @Override
  public boolean processLayer(NetworkAnalyzer analyzer, int layerIndex) {
    Layer.Builder builder = analyzer.layerBuilder(layerIndex);
    if (builder.type() != LayerType.YOLO)
      return false;
    auxProcessLayer(analyzer, builder);
    return true;
  }

  private void auxProcessLayer(NetworkAnalyzer analyzer, Layer.Builder layer) {
    Yolo yol = modelConfig();
    yol = YoloUtil.validate(yol);

    IPoint inputImageSize = VolumeUtil.spatialDimension(layer.inputVolume());
    IPoint gridSize = YoloUtil.gridSize(yol);

    if (!inputImageSize.equals(gridSize)) {
      todo("Perhaps this shouldn't be a problem if fullyConnected is in effect?");
      analyzer.addProblem("Spatial dimension of input volume:", inputImageSize, "!= Yolo grid size",
          gridSize);
      return;
    }

    int valuesPerBlock = YoloUtil.valuesPerBlock(yol);
    IPoint grid = YoloUtil.gridSize(yol);

    // It is unclear to me how the various versions of YOLO work with the output layer.
    //

    int inputDepth = layer.inputVolume().depth();

    if (inputDepth != valuesPerBlock) {
      // Add a fully-connected layer to generate outputs from the inputs.

      Vol inBox = layer.inputVolume();
      Vol outputBox = VolumeUtil.build(grid.x, grid.y, valuesPerBlock);

      int inputVolume = VolumeUtil.product(inBox);
      int outputVolume = VolumeUtil.product(outputBox);
      layer.outputVolume(outputBox);

      NetworkUtil.calcWeightsForFC(layer, inputVolume, outputVolume);

    } else {
      // The input volume has the same dimensions as the YOLO output layer, so assume the intention
      // is to treat the input volume as the YOLO output directly.
      layer.numWeights(0);
    }
  }

  @Override
  public void describeLayer(NetworkAnalyzer an, Layer layer, StringBuilder sb) {
    Yolo yol = modelConfig();
    sb.append("anchors:" + yol.anchorBoxesPixels().size());
    sb.append(" categories:" + yol.categoryCount());
  }

  @Override
  public void init() {
    Yolo yolo = modelConfig();
    mAnchorSizeRelImage = YoloUtil.anchorBoxesRelativeToImageSize(yolo);
    mAnchorSize = YoloUtil.anchorBoxSizes(yolo);
    mBlockSize = yolo.blockSize();
    mGridSize = YoloUtil.gridSize(yolo);
    mGridToImageScale = yolo.blockSize().toFPoint();
    mImageToGridScale = new FPoint(1f / mGridToImageScale.x, 1f / mGridToImageScale.y);
    constructOutputLayer();
  }

  @Override
  public void accept(Object imagePixelsArray, List<ScriptElement> scriptElementList) {

    writeImage(imagePixelsArray);

    clearOutputLayer();
    ScriptUtil.assertNoMixing(scriptElementList);

    // Compile annotations into ones that have a single bounding box
    List<RectElement> boxes = arrayList();

    for (ScriptElement elem : scriptElementList) {
      switch (elem.tag()) {
      default:
        throw badArg("unsupported element type", elem);
      case RectElement.TAG:
      case PolygonElement.TAG: {
        // We will assume that the polygon bounding box is a good enough approximation of the
        // object's bounding rectangle.  We hopefully have used the 'truncate box, then rotate its points'
        // heuristic earlier (if the original object bounds was a box; if it was a polygon, that heuristic
        // doesn't apply){
        IRect bounds = elem.bounds();
        boxes.add(
            labelledBox(bounds, ScriptUtil.categoryOrZero(elem), 1f, ScriptUtil.rotationDegreesOrZero(elem)));
      }
        break;
      }
    }

    List<RectElement> neighbors = generateNeighborVersions(boxes);
    boxes.addAll(neighbors);
    boxes.sort((a, b) -> -Integer.compare(ScriptUtil.confidence(a), ScriptUtil.confidence(b)));

    log("sorted boxes, including neighbors:", INDENT, boxes);

    for (RectElement box : boxes) {
      if (verbose()) {
        log(" box:", box.toJson().toString());
        log("  cp:", box.bounds().midPoint());
      }
      chooseAnchorBox(box.bounds().size());
      if (convertBoxToCell(box.bounds()))
        writeBoxToFieldsBuffer(box);
    }

    writeLabels(mOutputLayer);
  }

  @Override
  public void storeImageSetInfo(ImageSetInfo.Builder imageSetInfo) {
    imageSetInfo //
        .labelLengthBytes(mFieldsPerImage * bytesPerValue(network().labelDataType())) //
        .imageLengthBytes(inputImageVolumeProduct() * bytesPerValue(network().imageDataType())) //
    ;
  }

  @Override
  public void parseInferenceResult(byte[] modelOutput, int confidencePct, Script.Builder script) {
    Yolo yolo = modelConfig();
    float[] f = DataUtil.bytesToFloatsLittleEndian(modelOutput);

    log("Constructing YOLO result for image; confidence threshold %", confidencePct);

    List<ScriptElement> boxList = arrayList();
    final int anchorBoxCount = numAnchorBoxes();
    final int fieldsPerBox = YoloUtil.valuesPerAnchorBox(yolo);
    final float logitMinForResult = NetworkUtil.logit(confidencePct / 100f);
    final int categoryCount = yolo.categoryCount();

    float highestObjectnessLogitSeen = 0f;
    int fieldSetIndex = 0;
    for (int cellY = 0; cellY < mGridSize.y; cellY++) {
      for (int cellX = 0; cellX < mGridSize.x; cellX++) {
        for (int anchorBox = 0; anchorBox < anchorBoxCount; anchorBox++, fieldSetIndex += fieldsPerBox) {

          float objectnessLogit = f[fieldSetIndex + F_CONFIDENCE];
          // Note, this check will cause us to skip a lot of computation, which
          // suggests we probably don't want to embed the sigmoid/exp postprocessing steps into the model;
          // but then again, the model probably has very optimized, parallel versions of those functions
          if (objectnessLogit < logitMinForResult)
            continue;

          highestObjectnessLogitSeen = Math.max(highestObjectnessLogitSeen, objectnessLogit);
          float objectnessConfidence = NetworkUtil.sigmoid(objectnessLogit);

          IPoint anchorBoxPixels = yolo.anchorBoxesPixels().get(anchorBox);
          float anchorBoxWidth = anchorBoxPixels.x;
          float anchorBoxHeight = anchorBoxPixels.y;

          int bestCategory = 0;
          if (categoryCount > 1) {
            float bestConf = -1;
            int k = fieldSetIndex + F_CLASS_PROBABILITIES;
            for (int i = 0; i < categoryCount; i++) {
              float conf = f[k + i];
              if (i == 0 || bestConf < conf) {
                bestCategory = k;
                bestConf = conf;
              }
            }
          }

          int k = fieldSetIndex + F_BOX_XYWH;

          // Note that the x,y coordinates (which represent the center of the box)
          // are relative to the cell,
          // while the width and height are relative to the anchor box size

          // I considered making the x,y coordinates relative to the center of the cell, but there
          // is no point as we are predicting the *sigmoid* of the relative to the cell, so 0 naturally
          // predicts its center.
          //
          float bx = NetworkUtil.sigmoid(f[k + 0]);
          float by = NetworkUtil.sigmoid(f[k + 1]);
          float ws = NetworkUtil.exp(f[k + 2]);
          float hs = NetworkUtil.exp(f[k + 3]);

          float bw = anchorBoxWidth * ws;
          float bh = anchorBoxHeight * hs;

          float midpointX = (bx + cellX) * mGridToImageScale.x;
          float midpointY = (by + cellY) * mGridToImageScale.y;

          IRect boxRect = new FRect(midpointX - bw / 2, midpointY - bh / 2, bw, bh).toIRect();

          // I think we want to use the 'objectness' confidence, without incorporating the best category's confidence in any way
          ElementProperties.Builder prop = ElementProperties.newBuilder();
          prop.category(bestCategory);
          prop.confidence(MyMath.parameterToPercentage(objectnessConfidence));
          RectElement boxObj = new RectElement(prop, boxRect);
          boxList.add(boxObj);
        }
      }
    }

    if (verbose()) {
      if (boxList.isEmpty())
        pr("*** No boxes detected");
      else {
        pr("Valid anchor boxes:", boxList.size());
        pr("Highest conf:", NetworkUtil.sigmoid(highestObjectnessLogitSeen) * 100);
      }
    }

    if (mParserConfig.maxIOverU() > 0)
      boxList = YoloUtil.performNonMaximumSuppression(boxList, mParserConfig.maxIOverU());
    script.items(boxList);
  }

  private PlotInferenceResultsConfig mParserConfig = PlotInferenceResultsConfig.DEFAULT_INSTANCE;

  // ------------------------------------------------------------------

  private List<RectElement> generateNeighborVersions(List<RectElement> boxes) {
    List<RectElement> neighborList = arrayList();
    Yolo yolo = modelConfig();

    if (yolo.neighborFactor() == 0)
      return neighborList;

    log("generate neighbors, factor", yolo.neighborFactor(), "for box count", boxes.size());

    for (RectElement original : boxes) {

      // We are computing these twice; once here, and once later on when we process the original box
      if (!convertBoxToCell(original.bounds()))
        continue;

      IRect obox = original.bounds();
      IPoint cp = obox.center();

      for (int nbInd = 0; nbInd < sNeighborCellOffsets.length; nbInd += 2) {
        IPoint ngridCell = mBoxGridCell.sumWith(sNeighborCellOffsets[nbInd], sNeighborCellOffsets[nbInd + 1]);

        if (!cellWithinGrid(ngridCell))
          continue;

        // Determine bounds of neighboring cell, in pixels.  Inset by a pixel or two to avoid extreme logit values
        int inset = 1;
        IRect ncellBounds = new IRect(ngridCell.x * mBlockSize.x + inset, ngridCell.y * mBlockSize.y + inset,
            mBlockSize.x - 2 * inset, mBlockSize.y - 2 * inset);

        IPoint ncp = cp.clampTo(ncellBounds);

        // Determine bounding box of the neighbor version.  We add half the offset to try to contain
        // the part that got shifted to the side, without adding too much to the opposite side

        IPoint neighborOffset = IPoint.difference(ncp, cp);
        int neighborWidth = obox.width + Math.abs(neighborOffset.x) / 2;
        int neighborHeight = obox.height + Math.abs(neighborOffset.y) / 2;
        IRect nbox = new IRect(ncp.x - neighborWidth / 2, ncp.y - neighborHeight / 2, neighborWidth,
            neighborHeight);

        // Determine the confidence estimate from the IOU of the neighbor over the original

        float iOverU = MyMath.intersectionOverUnion(//
            obox.x, obox.y, obox.width, obox.height, //
            nbox.x, nbox.y, nbox.width, nbox.height//
        );
        if (verbose()) {
          log("original:", obox);
          log("neighbor:", nbox);
          log("iOverU  :", iOverU);
        }

        if (iOverU < yolo.neighborFactor())
          continue;

        // Scale the stored confidence down further, to penalize it for having incorrect coordinates
        ElementProperties.Builder prop = original.properties().toBuilder();
        RectElement neighbor = new RectElement(prop.confidence(MyMath.parameterToPercentage(iOverU * iOverU)),
            nbox);
        neighborList.add(neighbor);
      }
    }
    return neighborList;
  }

  private static int sNeighborCellOffsets[] = { -1, 0, 0, -1, 1, 0, 0, 1 };

  private void writeBoxToFieldsBuffer(RectElement box) {

    float[] b = mOutputLayer;

    int cellIndex = mBoxGridCell.x + mGridSize.x * mBoxGridCell.y;

    int f = mFieldsPerGridCell * cellIndex + mAnchorBox * mFieldsPerAnchorBox;

    // If this field is already used, do nothing else; we don't want to have to clear
    // it (its conditional probabilities) before overwriting it

    // This will also happen if we are augmenting with approximations of images lying to neighboring cells

    if (b[f + YoloUtil.F_CONFIDENCE] != 0f) {
      if (verbose()) {
        log("field buffer already occupied:", mBoxGridCell, "anchor:", mAnchorBox);
      }
      return;
    }

    // The x and y coordinates can range from 0...1.
    //
    b[f + YoloUtil.F_BOX_XYWH + 0] = NetworkUtil.logit(mBoxLocationRelativeToCell.x);
    b[f + YoloUtil.F_BOX_XYWH + 1] = NetworkUtil.logit(mBoxLocationRelativeToCell.y);

    // The width and height can range from 0...+inf.
    //
    b[f + YoloUtil.F_BOX_XYWH + 2] = NetworkUtil.ln(mBoxSizeRelativeToAnchorBox.x);
    b[f + YoloUtil.F_BOX_XYWH + 3] = NetworkUtil.ln(mBoxSizeRelativeToAnchorBox.y);

    // The ground truth values for confidence are stored as *indicator variables*, hence 0f or 1f.
    // Hence, we store logit(1), since a box exists here.  Actually, the confidence could
    // be < 100% (really???); so read it from the box...
    //
    b[f + YoloUtil.F_CONFIDENCE] = NetworkUtil.logit(MyMath.percentToParameter(ScriptUtil.confidence(box)));

    // The class probabilities are the same; we store a one-hot indicator variable for this box's class.
    // We could just store the category number as a scalar instead of a one-hot vector, to save a bit of memory
    // and a bit of Python code, but this keeps the structure of the input and output box information the same

    b[f + YoloUtil.F_CLASS_PROBABILITIES + ScriptUtil.categoryOrZero(box)] = NetworkUtil.LOGIT_1;
  }

  private void constructOutputLayer() {
    mFieldsPerAnchorBox = YoloUtil.valuesPerAnchorBox(modelConfig());
    mFieldsPerGridCell = mFieldsPerAnchorBox * numAnchorBoxes();
    mFieldsPerImage = mFieldsPerGridCell * mGridSize.product();
    mOutputLayer = new float[mFieldsPerImage];

    if (verbose()) {
      JSMap m = map();
      m.putNumbered("categories", modelConfig().categoryCount());
      m.putNumbered("f per anchor", mFieldsPerAnchorBox);
      m.putNumbered("anchor boxes", numAnchorBoxes());
      m.putNumbered("f per cell", mFieldsPerGridCell);
      m.putNumbered("cells", mGridSize.product());
      m.putNumbered("f per image", mOutputLayer.length);
      log("output grid:", INDENT, m);
    }
  }

  private void clearOutputLayer() {
    Arrays.fill(mOutputLayer, 0);
  }

  private IPoint determineBoxGridCell(IPoint pointWithinImage) {
    IPoint blockSize = modelConfig().blockSize();
    return new IPoint(Math.floorDiv(pointWithinImage.x, blockSize.x),
        Math.floorDiv(pointWithinImage.y, blockSize.y));
  }

  private boolean cellWithinGrid(IPoint gridCell) {
    return !(gridCell.x < 0 || gridCell.y < 0 || gridCell.x >= mGridSize.x || gridCell.y >= mGridSize.y);
  }

  /**
   * Convert box to grid and YOLO image space
   * 
   * If center of box lies outside the image, or the box's size is too small,
   * return false
   * 
   * Otherwise, initializes mBoxLocationRelativeToCell and
   * mBoxSizeRelativeToAnchorBox
   * 
   */
  private boolean convertBoxToCell(IRect box) {
    Yolo yolo = modelConfig();
    mBoxLocationRelativeToCell = null;
    mBoxSizeRelativeToAnchorBox = null;
    mBoxGridCell = null;

    final int minBoxDimension = 8;
    if (box.minDim() < minBoxDimension) {
      log("  box dimensions are too small:", box);
      return false;
    }

    IPoint midPoint = box.midPoint();
    IPoint gridCell = determineBoxGridCell(midPoint);
    if (!cellWithinGrid(gridCell)) {
      log("  box centerpoint not within image");
      return false;
    }

    mBoxSizeRelativeToAnchorBox = new FPoint(//
        (box.width / (float) yolo.imageSize().x) / mAnchorSizeRelImage[mAnchorBox * 2 + 0], //
        (box.height / (float) yolo.imageSize().y) / mAnchorSizeRelImage[mAnchorBox * 2 + 1]);

    mBoxGridCell = gridCell;

    mBoxLocationRelativeToCell = new FPoint(//
        midPoint.x * mImageToGridScale.x - gridCell.x, //
        midPoint.y * mImageToGridScale.y - gridCell.y);

    log("  grid cell:", mBoxGridCell);
    log("  loc(cell):", mBoxLocationRelativeToCell);
    log("  size(img):", mBoxSizeRelativeToAnchorBox);
    return true;
  }

  private int numAnchorBoxes() {
    // TODO: we can optimize things by precomputing this and other constants, and storing in instance fields
    return mAnchorSizeRelImage.length / 2;
  }

  /**
   * Choose best anchor box for current box
   */
  private void chooseAnchorBox(IPoint boxSizeI) {
    float boxSizeX = boxSizeI.x;
    float boxSizeY = boxSizeI.y;
    float boxArea = boxSizeX * boxSizeY;
    float bestIOverU = 0;
    int bestAnchorBoxIndex = 0;

    for (int i = 0; i < numAnchorBoxes(); i++) {
      float anchorSizeX = mAnchorSize[i * 2 + 0];
      float anchorSizeY = mAnchorSize[i * 2 + 1];

      // We want the intersection / union.
      // I think this is the same whether we align the two boxes at their centerpoints 
      // OR at their bottom left corners.

      float minWidth = Math.min(boxSizeX, anchorSizeX);
      float minHeight = Math.min(boxSizeY, anchorSizeY);
      float intersection = minWidth * minHeight;
      float union = boxArea + anchorSizeX * anchorSizeY - intersection;
      float currentIOverU = intersection / union;

      if (currentIOverU > bestIOverU) {
        bestIOverU = currentIOverU;
        bestAnchorBoxIndex = i;
      }
    }
    mAnchorBox = bestAnchorBoxIndex;
    mIOverU = bestIOverU;
    log("  anchor box:", mAnchorBox);
    log("    I over U:", mIOverU);
  }

  private static RectElement labelledBox(IRect box, int category, float confidence, int rotationDegrees) {
    return new RectElement(ElementProperties.newBuilder().category(category)
        .confidence(MyMath.parameterToPercentage(confidence)).rotation(rotationDegrees), box);
  }

  private float[] mAnchorSizeRelImage;
  private float[] mAnchorSize;

  private IPoint mGridSize;
  private FPoint mGridToImageScale;
  private IPoint mBlockSize;
  private FPoint mImageToGridScale;

  private IPoint mBoxGridCell;
  private FPoint mBoxLocationRelativeToCell;
  private FPoint mBoxSizeRelativeToAnchorBox;
  private int mAnchorBox;
  private float mIOverU;

  private int mFieldsPerAnchorBox;
  private int mFieldsPerGridCell;
  private int mFieldsPerImage;
  private float[] mOutputLayer;

}
