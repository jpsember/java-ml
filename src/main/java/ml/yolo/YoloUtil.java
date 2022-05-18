package ml.yolo;

import static js.base.Tools.*;

import java.util.Comparator;
import java.util.List;

import js.geometry.FPoint;
import js.geometry.FRect;
import js.geometry.IPoint;
import js.geometry.IRect;
import js.geometry.MyMath;
import js.graphics.MaskElement;
import js.graphics.RectElement;
import js.graphics.ScriptElement;
import js.graphics.ScriptUtil;
import js.graphics.gen.ElementProperties;
import js.json.JSList;
import ml.NetworkUtil;
import gen.Yolo;

public final class YoloUtil {

  public static final boolean I20 = false && alert("Issue #20 in effect");

  private static final List<IPoint> DEFAULT_ANCHOR_BOXES = IPoint.toArrayList(//
      18, 22, //
      60, 66 //
  );

  public static Yolo applyDefaults(Yolo yolo) {
    Yolo.Builder b = yolo.toBuilder();
    if (b.anchorBoxesPixels().isEmpty())
      b.anchorBoxesPixels(DEFAULT_ANCHOR_BOXES);
    return validate(b);
  }

  public static Yolo validate(Yolo yolo) {
    checkArgument(yolo.imageSize().positive(), "image_size undefined");
    checkArgument(yolo.blockSize().positive(), "block size undefined");
    checkArgument(yolo.categoryCount() > 0, "num_categories undefined");
    checkArgument(!yolo.anchorBoxesPixels().isEmpty(), "no anchor boxes");
    return yolo.build();
  }

  public static IPoint blocksToPixels(IPoint gridSize, Yolo yolo) {
    return new IPoint(gridSize.x * yolo.blockSize().x, gridSize.y * yolo.blockSize().y);
  }

  public static IPoint gridSize(Yolo yolo) {
    IPoint bs = yolo.blockSize();
    IPoint imageSize = yolo.imageSize();
    IPoint grid = new IPoint(imageSize.x / bs.x, imageSize.y / bs.y);
    if (!grid.positive())
      throw die("gridSize failed, yolo:", INDENT, yolo);
    IPoint image = blocksToPixels(grid, yolo);
    if (!image.equals(imageSize))
      throw die("Image size is not multiple of block size:", imageSize, bs);
    return grid;
  }

  // Fields within each output layer record
  //
  public static final int F_BOX_XYWH = 0;
  public static final int F_CONFIDENCE = F_BOX_XYWH + 4;
  public static final int F_CLASS_PROBABILITIES = F_CONFIDENCE + 1;

  public static int anchorBoxCount(Yolo yolo) {
    return yolo.anchorBoxesPixels().size();
  }

  public static int valuesPerAnchorBox(Yolo yolo) {
    return F_CLASS_PROBABILITIES + yolo.categoryCount();
  }

  public static int valuesPerBlock(Yolo yolo) {
    return valuesPerAnchorBox(yolo) * anchorBoxCount(yolo);
  }

  public static int imageFloatCount(Yolo yolo) {
    return yolo.imageSize().product() * yolo.imageChannels();
  }

  public static int imageLabelFloatCount(Yolo yolo) {
    return valuesPerBlock(yolo) * gridSize(yolo).product();
  }

  /**
   * Parse a training label from a particular slot in a training label data
   * array. Used for test purposes only, since we are only interesting in going
   * the other direction: compiling a training label into the data array
   */
  public static RectElement parseTrainLabel(Yolo yolo, int cellX, int cellY, int anchorIndex,
      float[] floatArray, int offset) {
    int categoryFoundCount = 0;
    int bestCategory = -1;
    float bestProb = -1;
    for (int j = 0; j < yolo.categoryCount(); j++) {
      float prob = floatArray[offset + F_CLASS_PROBABILITIES + j];
      if (prob <= 0)
        continue;
      categoryFoundCount++;
      if (prob > bestProb) {
        bestProb = prob;
        bestCategory = j;
      }
    }
    if (categoryFoundCount > 1)
      throw die("multiple categories found at x:", cellX, "y:", cellY);
    if (bestCategory < 0)
      return RectElement.DEFAULT_INSTANCE;

    float xRel = floatArray[offset + F_BOX_XYWH + 0];
    float yRel = floatArray[offset + F_BOX_XYWH + 1];

    float wRel = floatArray[offset + F_BOX_XYWH + 2];
    float hRel = floatArray[offset + F_BOX_XYWH + 3];

    IPoint anchorSize = yolo.anchorBoxesPixels().get(anchorIndex);
    FPoint anchorSizeRelImage = new FPoint(anchorSize.x / (float) yolo.imageSize().x,
        anchorSize.y / (float) yolo.imageSize().y);
    float wAbs = (wRel * anchorSizeRelImage.x) * yolo.imageSize().x;
    float hAbs = (hRel * anchorSizeRelImage.y) * yolo.imageSize().y;

    float xAbs = (xRel + cellX) * yolo.blockSize().x - wAbs / 2;
    float yAbs = (yRel + cellY) * yolo.blockSize().y - hAbs / 2;

    ElementProperties.Builder prop = ElementProperties.newBuilder();
    prop.category(bestCategory);
    prop.confidence(100);
    prop.anchor(anchorIndex);
    return new RectElement(prop, new FRect(xAbs, yAbs, wAbs, hAbs).toIRect());
  }

  /**
   * Convert value output by the Yolo model to a box rotation, in degrees
   */
  public static int convertNetworkValueToRotationDegrees(float f) {
    return Math.round(NetworkUtil.tanh(f) * RectElement.BOX_ROT_MAX);
  }

  public static float[] anchorBoxSizes(Yolo yolo) {
    float[] result = new float[yolo.anchorBoxesPixels().size() * 2];
    int cursor = 0;
    for (IPoint box : yolo.anchorBoxesPixels()) {
      result[cursor + 0] = box.x;
      result[cursor + 1] = box.y;
      cursor += 2;
    }
    return result;
  }

  public static float[] anchorBoxesRelativeToImageSize(Yolo yolo) {
    float[] result = new float[yolo.anchorBoxesPixels().size() * 2];
    int cursor = 0;
    for (IPoint box : yolo.anchorBoxesPixels()) {
      result[cursor + 0] = box.x / (float) yolo.imageSize().x;
      result[cursor + 1] = box.y / (float) yolo.imageSize().y;
      cursor += 2;
    }
    if (I20) {
      pr("anchorBoxPixels:", yolo.anchorBoxesPixels());
      pr("rel to image size:", JSList.with(result));
    }
    return result;
  }

  public static final Comparator<ScriptElement> COMPARATOR_LEFT_TO_RIGHT = (a, b) -> {
    IPoint ac = a.bounds().center();
    IPoint bc = b.bounds().center();
    int rc = Integer.compare(ac.x, bc.x);
    if (rc == 0)
      rc = Integer.compare(ac.y, bc.y);
    if (rc == 0)
      rc = Integer.compare(a.bounds().width, b.bounds().width);
    if (rc == 0)
      rc = Integer.compare(a.bounds().height, b.bounds().height);
    if (rc == 0)
      rc = Integer.compare(ScriptUtil.categoryOrZero(a), ScriptUtil.categoryOrZero(b));
    return rc;
  };

  public static final Comparator<ScriptElement> COMPARATOR_CONFIDENCE = (a, b) -> {
    int rc = 0;
    if (ScriptUtil.hasConfidence(a) && ScriptUtil.hasConfidence(b))
      rc = -Integer.compare(ScriptUtil.confidence(a), ScriptUtil.confidence(b));
    if (rc == 0)
      rc = COMPARATOR_LEFT_TO_RIGHT.compare(a, b);
    return rc;
  };

  public static List<ScriptElement> performNonMaximumSuppression(List<ScriptElement> boxes, float maxIOverU) {
    return performNonMaximumSuppression(boxes, maxIOverU, false);
  }

  public static List<ScriptElement> performNonMaximumSuppression(List<ScriptElement> boxes, float maxIOverU,
      boolean db) {

    List<ScriptElement> sorted = arrayList();
    sorted.addAll(boxes);
    sorted.sort(COMPARATOR_CONFIDENCE);
    if (db)
      pr("performNonMaxSuppression; sorted elements:", INDENT, sorted);

    List<ScriptElement> result = arrayList();
    omit: for (ScriptElement b : sorted) {
      IRect br = b.bounds();
      if (db)
        pr("testing box:", b);

      for (ScriptElement a : result) {
        IRect ar = a.bounds();
        float iOverU = MyMath.intersectionOverUnion(//
            ar.x, ar.y, ar.width, ar.height, //
            br.x, br.y, br.width, br.height);

        // If logging, only show results if the two boxes have the same category
        if (db && ScriptUtil.categoryOrZero(b) == ScriptUtil.categoryOrZero(a))
          pr("existing:", INDENT, a, CR, "i/u:", iOverU);

        if (iOverU >= maxIOverU) {
          if (db)
            pr("...omitting");
          continue omit;
        }
      }
      if (db)
        pr("...adding to result");
      result.add(b);
    }
    result.sort(COMPARATOR_LEFT_TO_RIGHT);
    return result;
  }

  public static List<ScriptElement> applyOutOfBoundsFilter(Yolo yolo, List<RectElement> boxes,
      float minFractionInsideBounds, boolean replaceWithMasks) {
    final boolean db = false && alert("logging");
    if (db)
      pr("out of bounds filter");
    IRect imageBounds = new IRect(yolo.imageSize());
    if (false && alert("indenting effective image bounds")) {
      imageBounds = imageBounds.withInset(55);
    }
    List<ScriptElement> result = arrayList();
    for (RectElement box : boxes) {
      float fraction = 0f;
      IRect intersect = IRect.intersection(box.bounds(), imageBounds);
      if (intersect != null)
        fraction = intersect.area() / (float) box.bounds().area();
      if (db)
        pr("intersect:", intersect, CR, "fraction:", fraction, "min:", minFractionInsideBounds);
      if (fraction >= minFractionInsideBounds) {
        result.add(box);
        continue;
      }
      if (!replaceWithMasks)
        continue;
      if (db)
        pr("replace with mask");
      result.add(new MaskElement(box.bounds()));
    }
    return result;
  }

}
