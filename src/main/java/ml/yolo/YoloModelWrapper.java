package ml.yolo;

import java.awt.image.BufferedImage;
import java.io.DataInputStream;
import java.io.File;
import java.util.Arrays;
import java.util.List;

import static js.base.Tools.*;

import js.data.DataUtil;
import js.file.Files;
import js.geometry.FPoint;
import js.geometry.FRect;
import js.geometry.IPoint;
import js.geometry.IRect;
import js.geometry.MyMath;
import js.graphics.ImgUtil;
import js.graphics.PolygonElement;
import js.graphics.RectElement;
import js.graphics.ScriptElement;
import js.graphics.ScriptUtil;
import js.graphics.gen.ElementProperties;
import js.graphics.gen.Script;
import js.graphics.gen.ScriptElementList;
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
import static ml.yolo.YoloUtil.I20;

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

    // I think what happens here is we apply 1x1 spatial filters to the input volume
    // to produce an output volume that has the same spatial dimension as the input,
    // but with the number of filters chosen to equal valuesPerBlock
    int numFilters = valuesPerBlock;

    if (layer.filters() != 0 && layer.filters() != numFilters) {
      analyzer.addProblem("Unexpected Yolo filters:", layer.filters(), "!=", numFilters);
      return;
    }
    layer.filters(numFilters);

    Vol inBox = layer.inputVolume();
    Vol outputBox = VolumeUtil.build(grid.x, grid.y, valuesPerBlock);

    NetworkUtil.calcWeightsForConv(layer, VolumeUtil.fibre(inBox.depth()), valuesPerBlock, outputBox);
    layer.outputVolume(outputBox);
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
    mAnchorSize  = YoloUtil.anchorBoxSizes(yolo);
    mBlockSize = yolo.blockSize();
    mPixelToGridCellScale = new FPoint(1f / mBlockSize.x, 1f / mBlockSize.y);
    mGridSize = YoloUtil.gridSize(yolo);
    constructOutputLayer();
  }

  @Override
  public void accept(float[] image, ScriptElementList scriptElementList) {
    writeImage(image);
    writeScriptElements(scriptElementList);
  }

  @Override
  public void storeImageSetInfo(ImageSetInfo.Builder imageSetInfo) {
    imageSetInfo //
        .labelLengthBytes(Float.BYTES * mFieldsPerImage) //
        .imageLengthBytes(inputImageVolumeProduct() * Float.BYTES) //
    ;
  }

  @Override
  public void parseInferenceResult(byte[] modelOutput, Script.Builder script) {
    if (I20)
      pr("******** parseInferenceResult");
    float[] imageLabelData = DataUtil.bytesToFloatsLittleEndian(modelOutput);
    List<ScriptElement> boxList = resultParser().readImageResult(imageLabelData);
    if (mParserConfig.maxIOverU() > 0) {
      boxList = YoloUtil.performNonMaximumSuppression(boxList, mParserConfig.maxIOverU());
    }
    script.items(boxList);
    halt();
  }

  private YoloResultParser resultParser() {
    if (mYoloResultParser == null) {
      YoloResultParser parser = new YoloResultParser(modelConfig());
      parser.withConfidenceFilter(mParserConfig.confidencePct() / 100f);
      mYoloResultParser = parser;
    }
    return mYoloResultParser;
  }

  private YoloResultParser mYoloResultParser;
  private PlotInferenceResultsConfig mParserConfig = PlotInferenceResultsConfig.DEFAULT_INSTANCE;

  // ------------------------------------------------------------------

  private void writeScriptElements(ScriptElementList scriptElements) {
    log("writeScriptElements", INDENT, scriptElements);

    clearOutputLayer();
    ScriptUtil.assertNoMixing(scriptElements.elements());

    // Compile annotations into ones that have a single bounding box
    List<RectElement> boxes = arrayList();

    for (ScriptElement elem : scriptElements.elements()) {
      switch (elem.tag()) {
      default:
        throw badArg("unsupported element type", elem);
      case RectElement.TAG:
      case PolygonElement.TAG:
        // We will assume that the polygon bounding box is a good enough approximation of the
        // object's bounding rectangle.  We hopefully have used the 'truncate box, then rotate its points'
        // heuristic earlier (if the original object bounds was a box; if it was a polygon, that heuristic
        // doesn't apply)
        boxes.add(labelledBox(elem.bounds(), ScriptUtil.categoryOrZero(elem), 1f,
            ScriptUtil.rotationDegreesOrZero(elem)));
        break;
      }
    }

    List<RectElement> neighbors = generateNeighborVersions(boxes);
    boxes.addAll(neighbors);
    boxes.sort((a, b) -> -Integer.compare(ScriptUtil.confidence(a), ScriptUtil.confidence(b)));
    log("sorted boxes, including neighbors:", INDENT, boxes);

    for (RectElement box : boxes) {
      if (I20)
        pr("elem bounds:", box.bounds());
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

  private List<RectElement> generateNeighborVersions(List<RectElement> boxes) {
    List<RectElement> neighborList = arrayList();
    Yolo yolo = modelConfig();

    if (yolo.neighborFactor() == 0)
      return neighborList;
    if (I20)
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

    // I wasn't aware there was a choice within TensorFlow for the label ordering: FORMAT_NHWC vs FORMAT_NCHW

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
    // These are ground truth values, and hence don't need to be represented as logits
    //
    b[f + YoloUtil.F_BOX_XYWH + 0] = mBoxLocationRelativeToCell.x;
    b[f + YoloUtil.F_BOX_XYWH + 1] = mBoxLocationRelativeToCell.y;

    // The width and height can range from 0...+inf.
    // These are ground truth values, and hence don't need to be represented as logarithms
    //
    b[f + YoloUtil.F_BOX_XYWH + 2] = mBoxSizeRelativeToAnchorBox.x;
    b[f + YoloUtil.F_BOX_XYWH + 3] = mBoxSizeRelativeToAnchorBox.y;

    // The ground truth values for confidence are stored as *indicator variables*, hence 0f or 1f.
    // Hence, we store 1f here, since a box exists here.
    //
    b[f + YoloUtil.F_CONFIDENCE] = MyMath.percentToParameter(ScriptUtil.confidence(box));

    // The class probabilities are the same; we store a one-hot indicator variable for this box's class.
    // We could just store the category number as a scalar instead of a one-hot vector, to save a bit of memory
    // and a bit of Python code, but this keeps the structure of the input and output box information the same

    b[f + YoloUtil.F_CLASS_PROBABILITIES + ScriptUtil.categoryOrZero(box)] = 1f;
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

  private IPoint determineBoxGridCell(IPoint midpoint) {
    IPoint blockSize = modelConfig().blockSize();
    return new IPoint(Math.floorDiv(midpoint.x, blockSize.x), Math.floorDiv(midpoint.y, blockSize.y));
  }

  private boolean cellWithinGrid(IPoint gridCell) {
    return !(gridCell.x < 0 || gridCell.y < 0 || gridCell.x >= mGridSize.x || gridCell.y >= mGridSize.y);
  }

  /**
   * Convert box to grid and YOLO image space
   * 
   * If center of box lies outside the image, or the box's size is too small,
   * mBoxGridCell will be null.
   * 
   * Otherwise, initializes mBoxLocationRelativeToCell and
   * mBoxSizeRelativeToAnchorBox
   * 
   * Returns true if result was successful
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
    if (I20) {
      pr("box size:", box.size(), "relative to anchor box:", mBoxSizeRelativeToAnchorBox);
    }

    mBoxGridCell = gridCell;
    mBoxLocationRelativeToCell = new FPoint(//
        midPoint.x * mPixelToGridCellScale.x - gridCell.x, //
        midPoint.y * mPixelToGridCellScale.y - gridCell.y);

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
    Yolo yolo = modelConfig();
    FPoint boxSize = boxSizeI.toFPoint();

    float[] anchorBoxes = mAnchorSizeRelImage;

    float bestIOverU = 0;
    int bestAnchorBoxIndex = 0;

    for (int i = 0; i < numAnchorBoxes(); i++) {

      todo("this could be precomputed");
      FPoint anchorSize = new FPoint(anchorBoxes[i * 2 + 0], anchorBoxes[i * 2 + 1])
          .scaledBy(yolo.blockSize());  
      halt("No, these are relative to image, not block size");
      //!!! No, not scaled by block size; scaled by image

      // We want the intersection / union.
      // I think this is the same whether we align the two boxes at their centerpoints 
      // OR at their bottom left corners.

      float minWidth = Math.min(boxSize.x, anchorSize.x);
      float minHeight = Math.min(boxSize.y, anchorSize.y);
      float intersection = minWidth * minHeight;
      float union = boxSize.product() + anchorSize.product() - intersection;
      float currentIOverU = intersection / union;

      if (I20)
        pr("i:", i, "anchorSize:", anchorSize, "boxSize:", boxSize);
      if (currentIOverU > bestIOverU) {
        bestIOverU = currentIOverU;
        bestAnchorBoxIndex = i;
      }
    }
    mAnchorBox = bestAnchorBoxIndex;
    mIOverU = bestIOverU;
    log("  anchor box:", mAnchorBox);
    log("    I over U:", mIOverU);
    if (I20)
      pr("anchorBox:", mAnchorBox, "I over U:", mIOverU);
  }

  private static RectElement labelledBox(IRect box, int category, float confidence, int rotationDegrees) {
    return new RectElement(ElementProperties.newBuilder().category(category)
        .confidence(MyMath.parameterToPercentage(confidence)).rotation(rotationDegrees), box);
  }

  private float[] mAnchorSizeRelImage;
  private float[] mAnchorSize;
  
  private IPoint mGridSize;
  private IPoint mBlockSize;
  private FPoint mPixelToGridCellScale;

  private IPoint mBoxGridCell;
  private FPoint mBoxLocationRelativeToCell;
  private FPoint mBoxSizeRelativeToAnchorBox;
  private int mAnchorBox;
  private float mIOverU;

  private int mFieldsPerAnchorBox;
  private int mFieldsPerGridCell;
  private int mFieldsPerImage;
  private float[] mOutputLayer;

  // ------------------------------------------------------------------
  // To be deleted later
  // ------------------------------------------------------------------

  @Deprecated // The parseInferenceResult method should serve this purpose
  public void plotInferenceResults(PlotInferenceResultsConfig config) {

    final String EVAL_IMAGES_FILENAME = "images.bin";
    final String EVAL_RESULTS_FILENAME = "inference_results.bin";

    Yolo yolo = modelConfig();

    File imagesFile = Files.assertExists(new File(config.inferenceInputDir(), EVAL_IMAGES_FILENAME));
    File resultsFile = Files.assertExists(new File(config.inferenceResultsDir(), EVAL_RESULTS_FILENAME));

    int imageLengthInBytes = inputImageVolumeProduct() * Float.BYTES;
    int imageCount = (int) (imagesFile.length() / imageLengthInBytes);

    Files.assertFileLength(imagesFile, imageCount * (long) imageLengthInBytes, "images");
    Files.assertFileLength(resultsFile, imageCount * (long) YoloUtil.imageLabelFloatCount(yolo) * Float.BYTES,
        "results");

    DataInputStream imagesInput = Files.dataInputStream(imagesFile);
    DataInputStream resultsInput = Files.dataInputStream(resultsFile);

    Files.S.backupAndRemake(config.outputDir());

    File annotationDir = ScriptUtil.scriptDirForProject(config.outputDir());
    Files.S.mkdirs(annotationDir);
    YoloResultParser yr = new YoloResultParser(yolo);
    yr.withConfidenceFilter(config.confidencePct() / 100f);

    for (int imageNumber = 0; imageNumber < imageCount; imageNumber++) {
      yr.setVerbose(config.inspectionFrameNumber() == imageNumber);
      yr.log("Plot Inference Results, image", imageNumber);
      float[] pixels = Files.readFloatsLittleEndian(imagesInput, inputImageVolumeProduct());
      BufferedImage bi = ImgUtil.floatsToBufferedImage(pixels, inputImagePlanarSize(),
          inputImageVolume().depth());

      String imageFilenameRoot = String.format("%03d", imageNumber);
      File imgDest = new File(config.outputDir(), imageFilenameRoot + ".jpg");
      ImgUtil.writeImage(Files.S, bi, imgDest);

      float[] imageLabelData = Files.readFloatsLittleEndian(resultsInput,
          YoloUtil.imageLabelFloatCount(yolo));
      List<ScriptElement> boxList = yr.readImageResult(imageLabelData);
      if (config.maxIOverU() > 0) {
        boxList = YoloUtil.performNonMaximumSuppression(boxList, config.maxIOverU());
      }

      Script.Builder script = Script.newBuilder();

      if (config.plotAnchorBoxes() && imageNumber == 0) {
        generateAnchorBoxSummary(boxList);
      }
      script.items(boxList);

      File annotPath = new File(annotationDir, imageFilenameRoot + ".json");
      ScriptUtil.write(Files.S, script, annotPath);
    }

    Files.close(imagesInput, resultsInput);
  }

  private void generateAnchorBoxSummary(List<ScriptElement> elements) {
    Yolo yolo = modelConfig();
    FPoint blockSize = yolo.blockSize().toFPoint();
    FPoint halfBlock = blockSize.scaledBy(0.5f);
    int numBox = YoloUtil.anchorBoxCount(yolo);

    for (int i = 0; i < numBox; i++) {
      IPoint cpt;
      {
        int row = i / 3;
        int col = (i % 3);
        cpt = new IPoint(col * blockSize.x * 4 + halfBlock.x, row * blockSize.y * 4 + halfBlock.y);
      }

      FPoint box = yolo.anchorBoxesPixels().get(i).toFPoint();
      float x = cpt.x - box.x / 2;
      float y = cpt.y - box.y / 2;
      elements.add(new RectElement(ElementProperties.newBuilder().category(0),
          new FRect(x, y, box.x, box.y).toIRect()));
      elements.add(new RectElement(ElementProperties.newBuilder().category(1),
          new FRect(cpt.x - halfBlock.x, cpt.y - halfBlock.x, blockSize.x, blockSize.y).toIRect()));
    }
  }

}
