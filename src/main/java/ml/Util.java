package ml;

import static js.base.Tools.*;

import java.io.DataOutputStream;
import java.io.File;
import java.util.List;
import java.util.Random;

import js.file.DirWalk;
import js.file.Files;
import js.geometry.Matrix;
import js.geometry.MyMath;
import js.graphics.RectElement;
import js.graphics.ScriptElement;
import js.graphics.ScriptUtil;
import js.graphics.gen.ElementProperties;
import js.graphics.gen.Script;
import gen.AugmentationConfig;
import gen.TransformWrapper;
import js.graphics.gen.ScriptElementList;

public final class Util {

  public static final String EVAL_IMAGES_FILENAME = "images.bin";
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
   * Compile ScriptElementList from list of shapes
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

  public static TransformWrapper transformWrapper(Matrix matrix, int rotationDegrees) {
    TransformWrapper.Builder b = TransformWrapper.newBuilder();
    b.matrix(matrix);
    b.inverse(matrix.invert());
    b.rotationDegrees(rotationDegrees);
    return b.build();
  }

}
