package ml;

import static js.base.Tools.*;

import java.io.DataOutputStream;
import java.util.Arrays;
import java.util.List;

import js.file.Files;
import js.base.BaseObject;
import js.geometry.FPoint;
import js.geometry.IPoint;
import js.geometry.IRect;
import js.geometry.MyMath;
import js.graphics.Inspector;
import js.graphics.PolygonElement;
import js.graphics.RectElement;
import js.graphics.ScriptElement;
import js.graphics.ScriptUtil;
import js.graphics.gen.ElementProperties;
import js.json.JSMap;
import ml.ModelInputReceiver;
import ml.YoloUtil;
import js.graphics.gen.ScriptElementList;
import gen.Yolo;

/**
 * Converts images and annotations to format for model training
 */
public final class YoloImageReceiver extends BaseObject implements ModelInputReceiver {

  public final static boolean CONSTANT_BOX = false
      && alert("Replacing Yolo train labels with single fixed box");

  public YoloImageReceiver(Yolo yolo) {
    mYolo = yolo;
    log("Yolo:", INDENT, yolo);
    mAnchorBoxes = YoloUtil.anchorBoxesRelativeToImageSize(yolo);
    mBlockSize = mYolo.blockSize();
    mPixelToGridCellScale = new FPoint(1f / mBlockSize.x, 1f / mBlockSize.y);
    mGridSize = YoloUtil.gridSize(yolo);
    constructOutputLayer();
  }

  public void setImageOutput(DataOutputStream stream) {
    mutable();
    mOutputImagesStream = stream;
  }

  public void setLabelOutput(DataOutputStream stream) {
    mutable();
    mOutputLayerStream = stream;
  }

  private Inspector mInspector = Inspector.NULL_INSPECTOR;

  // ------------------------------------------------------------------
  // ModelInputReceiver interface
  // ------------------------------------------------------------------

  @Override
  public void accept(float[] image, ScriptElementList annotation) {
    prepare();
    mAnnotations.add(annotation);
    writeImage(image);
    writeScriptElements(annotation);

    mInspector.create();
    if (mInspector.used()) {
      mInspector//
          .imageSize(mYolo.imageSize())//
          .channels(mYolo.imageChannels())//
          .elements(annotation.elements())//
          .image(image);
    }
  }

  @Override
  public List<ScriptElementList> annotations() {
    return mAnnotations;
  }

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

    if (CONSTANT_BOX && !boxes.isEmpty()) {
      IRect rect = new IRect(52, 58, 88, 46);
      boxes.clear();
      boxes.add(labelledBox(rect, 0, 1f, 0));
      once(() -> {
        pr("generating constant box:", rect);
        pr("cp:", rect.center());
      });
    }

    for (RectElement box : boxes) {

      if (verbose()) {
        log(" box:", box.toJson().toString());
        log("  cp:", box.bounds().midPoint());
      }
      chooseAnchorBox(box.bounds().size());
      if (convertBoxToCell(box.bounds()))
        writeBoxToFieldsBuffer(box);
    }

    writeOutputGrid();
  }

  private static int nb[] = { -1, 0, 0, -1, 1, 0, 0, 1 };

  private List<RectElement> generateNeighborVersions(List<RectElement> boxes) {
    List<RectElement> neighborList = arrayList();
    if (mYolo.neighborFactor() == 0)
      return neighborList;

    log("generate neighbors, factor", mYolo.neighborFactor(), "for box count", boxes.size());

    for (RectElement original : boxes) {

      // We are computing these twice; once here, and once later on when we process the original box
      if (!convertBoxToCell(original.bounds()))
        continue;

      IRect obox = original.bounds();
      IPoint cp = obox.center();

      for (int nbInd = 0; nbInd < nb.length; nbInd += 2) {
        IPoint ngridCell = mBoxGridCell.sumWith(nb[nbInd], nb[nbInd + 1]);

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

        if (iOverU < mYolo.neighborFactor())
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

  private void writeOutputGrid() {
    Files.S.writeFloatsLittleEndian(mOutputLayer, mOutputLayerStream);
  }

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

  private void writeImage(float[] image) {
    Files.S.writeFloatsLittleEndian(image, mOutputImagesStream);
  }

  private void constructOutputLayer() {
    mFieldsPerAnchorBox = YoloUtil.valuesPerAnchorBox(mYolo);
    mFieldsPerGridCell = mFieldsPerAnchorBox * numAnchorBoxes();
    int fieldsPerImage = mFieldsPerGridCell * mGridSize.product();
    mOutputLayer = new float[fieldsPerImage];

    if (verbose()) {
      JSMap m = map();
      m.putNumbered("categories", numCategories());
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
    IPoint blockSize = mYolo.blockSize();
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
        (box.width / (float) mYolo.imageSize().x) / mAnchorBoxes[mAnchorBox * 2 + 0], //
        (box.height / (float) mYolo.imageSize().y) / mAnchorBoxes[mAnchorBox * 2 + 1]);

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
    // TODO: we can optimize things by precomputing this and other constants, and storing in instance fields,
    return mAnchorBoxes.length / 2;
  }

  private int numCategories() {
    return mYolo.categoryCount();
  }

  /**
   * Choose best anchor box for current box
   */
  private void chooseAnchorBox(IPoint boxSizeI) {
    FPoint boxSize = boxSizeI.toFPoint();

    float[] anchorBoxes = mAnchorBoxes;

    float bestIOverU = 0;
    int bestAnchorBoxIndex = 0;

    for (int i = 0; i < numAnchorBoxes(); i++) {

      // The anchor box dimensions are multiples of the block size.
      FPoint anchorSize = new FPoint(anchorBoxes[i * 2 + 0], anchorBoxes[i * 2 + 1])
          .scaledBy(mYolo.blockSize());

      // We want the intersection / union.
      // I think this is the same whether we align the two boxes at their centerpoints 
      // OR at their bottom left corners.

      float minWidth = Math.min(boxSize.x, anchorSize.x);
      float minHeight = Math.min(boxSize.y, anchorSize.y);
      float intersection = minWidth * minHeight;
      float union = boxSize.product() + anchorSize.product() - intersection;
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

  // ------------------------------------------------------------------
  // Object lifecycle state
  // ------------------------------------------------------------------

  private void prepare() {
    if (mPrepared)
      return;
    mPrepared = true;
  }

  private void mutable() {
    if (mPrepared)
      throw die("already prepared");
  }

  private static RectElement labelledBox(IRect box, int category, float confidence, int rotationDegrees) {
    return new RectElement(ElementProperties.newBuilder().category(category)
        .confidence(MyMath.parameterToPercentage(confidence)).rotation(rotationDegrees), box);
  }

  private final Yolo mYolo;
  private final float[] mAnchorBoxes;
  private final IPoint mGridSize;
  private final IPoint mBlockSize;
  private final FPoint mPixelToGridCellScale;

  private DataOutputStream mOutputImagesStream = Files.NULL_DATA_OUTPUT_STREAM;
  private DataOutputStream mOutputLayerStream = Files.NULL_DATA_OUTPUT_STREAM;

  private IPoint mBoxGridCell;
  private FPoint mBoxLocationRelativeToCell;
  private FPoint mBoxSizeRelativeToAnchorBox;
  private int mAnchorBox;
  private float mIOverU;

  private int mFieldsPerAnchorBox;
  private int mFieldsPerGridCell;
  private float[] mOutputLayer;
  private List<ScriptElementList> mAnnotations = arrayList();
  private boolean mPrepared;
}
