package ml;

import static js.base.Tools.*;

import java.io.DataOutputStream;
import java.io.File;
import java.util.List;
import java.util.Random;

import js.file.DirWalk;
import js.file.Files;
import js.geometry.IPoint;
import js.geometry.IRect;
import js.geometry.Matrix;
import js.geometry.MyMath;
import js.geometry.Polygon;
import js.graphics.RectElement;
import js.graphics.ScriptElement;
import js.graphics.ScriptUtil;
import js.graphics.gen.ElementProperties;
import js.graphics.gen.Script;
import gen.AugmentationConfig;
import js.graphics.gen.ScriptElementList;

public final class Util {

  public static final String EVAL_IMAGES_FILENAME = "images.bin";
  public static final String EVAL_ANNOTATIONS_FILENAME = "annotations.json";
  public static final String EVAL_RESULTS_FILENAME = "inference_results.bin";

  public static final AugmentationConfig AUGMENTATIONS_NONE = AugmentationConfig.newBuilder()//
      .adjustBrightness(false)//
      .blurFactor(0)//
      .horizontalFlip(false)//
      .rotateDisable(true)//
      .scaleDisable(true)//
      .shearDisable(true)//
      .translateDisable(true)//
      .build();

  public static DataOutputStream outputDataStream(File dir, String name) {
    File f = prepareOutputFile(dir, name);
    return new DataOutputStream(Files.S.outputStream(f));
  }

  public static File prepareOutputFile(File dir, String name) {
    checkArgument(dir.isDirectory(), "not a directory: " + dir);
    File f = new File(dir, name);
    Files.S.deletePeacefully(f);
    return f;
  }

  public static final List<File> getSourceImageFiles(File inputDir) {
    DirWalk dirWalk = new DirWalk(inputDir).withExtensions("png", "jpg", "raw", "rax");
    return dirWalk.files();
  }

  public static void partitionToPrimaryAndTest(int testPct, List<ImageRecord> src, List<ImageRecord> dest0,
      List<ImageRecord> dest1) {
    checkState(src.size() > 0, "no output image files found");

    dest0.clear();
    dest1.clear();

    int trainFiles = ((100 - testPct) * src.size() + 50) / 100;
    dest0.addAll(src.subList(0, trainFiles));
    dest1.addAll(src.subList(trainFiles, src.size()));
  }

  @SuppressWarnings("unchecked")
  public static void clampSize(List list, int... bounds) {
    int limit = list.size();
    for (int x : bounds)
      if (x > 0)
        limit = Math.min(limit, x);
    removeAllButFirstN(list, limit);
  }

  public static void applyRandomBrightness(Random random, float[] pixels, float minShift, float maxShift) {
    float scale = 1 + MyMath.random(random, minShift, maxShift);
    applyRandomBrightness(pixels, scale);
  }

  public static void applyRandomBrightness(float[] pixels, float scale) {
    for (int i = 0; i < pixels.length; i++)
      pixels[i] = pixels[i] * scale;
  }

  public static ScriptElementList transform(ScriptElementList an, Matrix tfm, int rotationDegreesAdjust) {
    if (an.elements().isEmpty())
      return an;
    ScriptElementList.Builder b = an.toBuilder();
    List<ScriptElement> filtered = ScriptUtil.transform(an.elements(), tfm);
    if (rotationDegreesAdjust != 0) {
      List<ScriptElement> source = filtered;
      filtered = arrayList();
      for (ScriptElement elem : source) {
        ElementProperties.Builder prop = elem.properties().toBuilder();
        prop.rotation(
            clampBoxRotationDegrees(ScriptUtil.rotationDegreesOrZero(elem) + rotationDegreesAdjust));
        elem = elem.withProperties(prop);
        filtered.add(elem);
      }
    }
    b.elements(filtered);
    return b.build();
  }

  /**
   * Determine string that can be used to format integer counters from
   * 0...total-1, using just enough digits, with left-padded zeros.
   * 
   * Returns a string x such that String.format(x,i) yields e.g. "0072"
   * 
   * (public for testing)
   */
  public static String formatStringForTotal(int total) {
    checkArgument(total >= 0);
    int numDigits = 1;
    if (total > 0)
      numDigits = 1 + (int) Math.round(Math.floor(Math.log10(total)));
    return "%0" + numDigits + "d";
  }

  /**
   * Merge zero or more annotations into another
   */
  public static ScriptElementList.Builder mergeAnnotation(ScriptElementList targetOrNull,
      ScriptElementList source) {
    if (targetOrNull == null)
      targetOrNull = ScriptElementList.newBuilder();

    ScriptElementList.Builder b = targetOrNull.toBuilder();
    b.elements().addAll(source.elements());
    return b;
  }

  /**
   * Compile UbAnnotation from list of shapes
   */
  public static ScriptElementList compileAnnotation(List<ScriptElement> shapes) {
    ScriptElementList.Builder b = ScriptElementList.newBuilder();
    b.elements().addAll(shapes);
    return b.build();
  }

  public static ScriptElementList validate(ScriptElementList annotation) {
    ScriptUtil.assertNoMixing(annotation.elements());
    return annotation.build();
  }

  /**
   * Construct polygon representing a rectangle with its corners cut off, to
   * support a heuristic that approximates applying a transform to the rect by
   * taking the bounds of the transformed polygon
   */
  public static Polygon truncatedRect(IRect r, float s) {
    List<IPoint> pts = arrayList();

    float x0 = r.x;
    float y0 = r.y;
    float x3 = r.endX();
    float y3 = r.endY();

    float x1 = x0 * (1 - s) + x3 * s;
    float x2 = x3 * (1 - s) + x0 * s;
    float y1 = y0 * (1 - s) + y3 * s;
    float y2 = y3 * (1 - s) + y0 * s;

    addPt(pts, x1, y0);
    addPt(pts, x2, y0);
    addPt(pts, x3, y1);
    addPt(pts, x3, y2);
    addPt(pts, x2, y3);
    addPt(pts, x1, y3);
    addPt(pts, x0, y2);
    addPt(pts, x0, y1);

    return new Polygon(pts);
  }

  public static Polygon truncatedRect(IRect r) {
    return truncatedRect(r, 0.2f);
  }

  private static void addPt(List<IPoint> dest, float x, float y) {
    dest.add(new IPoint(x, y));
  }

  public static Script.Builder generateScriptFrom(ScriptElementList annotation) {
    validate(annotation);
    Script.Builder script = Script.newBuilder();
    script.items(annotation.elements());
    return script;
  }

  public static void generateScriptForImage(Files fileManager, File imageFile, ScriptElementList annotation) {
    Script.Builder annotationSet = generateScriptFrom(annotation);
    File scriptPath = ScriptUtil.scriptPathForImage(imageFile);
    ScriptUtil.write(fileManager, annotationSet, scriptPath);
  }

  public static int clampBoxRotationDegrees(int value) {
    return MyMath.clamp(value, RectElement.BOX_ROT_MIN, RectElement.BOX_ROT_MAX);
  }

  /**
   * Convert a box's rotation_degrees value (ROT_MIN...ROT_MAX) to training
   * value, which is a float from -1..1
   */
  public static float boxRotationToTrainingValue(int rotationDegrees) {
    if (rotationDegrees < RectElement.BOX_ROT_MIN || rotationDegrees > RectElement.BOX_ROT_MAX)
      throw die("illegal rotation_degrees:", rotationDegrees);
    return rotationDegrees * (1f / RectElement.BOX_ROT_MAX);
  }

}
