package ml;

import java.awt.image.BufferedImage;
import java.io.DataOutputStream;
import java.io.File;
import java.util.List;
import java.util.Random;

import static js.base.Tools.*;

import js.graphics.ImgUtil;
import js.graphics.ScriptElement;
import js.graphics.ScriptUtil;
import js.graphics.gen.Script;
import js.base.BaseObject;
import js.json.JSMap;
import ml.ModelWrapper;
import gen.AnnotationFile;
import gen.AugmentationConfig;
import gen.Layer;
import gen.NeuralNetwork;
import gen.PlotInferenceResultsConfig;
import js.graphics.gen.ScriptElementList;
import gen.Stats;

public abstract class ModelHandler extends BaseObject {

  public abstract void addImageRecordFilters(ImageHandler imageProcessor);

  public final ModelWrapper model() {
    return mModelConfig;
  }

  public final void setModel(ModelWrapper model) {
    loadTools();
    mModelConfig = model;
    auxSetModel();
  }

  /**
   * Allow subclasses to do additional initialization after model set
   */
  public void auxSetModel() {
  }

  //------------------------------------------------------------------
  // Training progress
  // ------------------------------------------------------------------

  public void updateTrainingProgress(ProgressFile progressFile, JSMap m) {
  }

  // ------------------------------------------------------------------
  // Statistics about training images (optional)
  //
  // This is somewhat limited, since the Stats object has fields for a 
  // single mean and std dev.  Perhaps we can add a JSMap field to allow 
  // future expansion.
  // ------------------------------------------------------------------

  /**
   * Optionally update statistics about images during training
   * 
   * Default implementation does nothing (i.e. stats not supported by this
   * handler)
   */
  public void updateStats(ImageRecord rec, Stats.Builder stats) {
  }

  /**
   * Perform any final stats manipulation. This is called once all the training
   * images have been added
   */
  public void summarizeStats(Stats.Builder stats) {
  }

  public ImageTransformer<BufferedImage> buildImageTransformer(AugmentationConfig augmentationConfig,
      Random random, Stats stats, ImageRecord imageRecord) {
    ImageTransformer<BufferedImage> transformer;
    if (mModelConfig.network().monochromeSourceImages()) {
      transformer = new MonochromeImageTransformer(this, augmentationConfig, random);
    } else {
      transformer = new ColorImageTransformer(model(), augmentationConfig, random);
    }
    return transformer;
  }

  /**
   * Construct a receiver for writing training (or inference) images, to be sent
   * to the Python code
   */
  public abstract ModelInputReceiver buildModelInputReceiver(DataOutputStream imagesStream,
      DataOutputStream labelsStream);

  // ------------------------------------------------------------------
  // Training progress
  // ------------------------------------------------------------------

  /**
   * Plot inference results in some user-friendly form (e.g. a scredit project)
   */
  public abstract void plotInferenceResults(PlotInferenceResultsConfig config);

  protected BufferedImage constructBufferedImage(float[] pixels) {
    return ImgUtil.floatsToBufferedImage(pixels, model().inputImagePlanarSize(),
        model().inputImageVolume().depth());
  }

  protected AnnotationFile readAnnotations(File inputDir) {
    File polygonFile = new File(inputDir, Util.EVAL_ANNOTATIONS_FILENAME);

    AnnotationFile.Builder fb = AnnotationFile.newBuilder();
    JSMap m = JSMap.from(polygonFile);

    for (JSMap polygonJsonMap : m.getList("p").asMaps()) {
      ScriptElementList annotation = ScriptElementList.DEFAULT_INSTANCE.parse(polygonJsonMap);
      fb.annotations().add(annotation);
    }
    return fb.build();
  }

  private ModelWrapper mModelConfig;

  protected RuntimeException notSupported() {
    return die("Unsupported project type:", model().projectType());
  }

  /**
   * Examine an AnnotationSet (script) and extract appropriate AnnotationShapes
   * from it by appending to target
   */
  public void extractShapes(Script script, List<ScriptElement> target) {
    throw notSupported();
  }

  protected final void assertNoMixing(Script script) {
    if (!ScriptUtil.rectElements(script).isEmpty() && !ScriptUtil.polygonElements(script).isEmpty())
      throw die("Cannot mix boxes and polygons");
  }

  /**
   * Calculate the scale and offset to apply to 16-bit monochrome pixel values
   * to convert to floats
   * 
   * Default just scales linearly from (0...0xffff) to (0...1)
   * 
   * TODO: this seems to add a lot of baggage, i.e. MonochromeImageTransformer
   * needs to expose lots of things (random, image stats) to make this pluggable
   */
  public void getIntegerToFloatPixelTransform(float[] mbOutput, ImageTransformer<BufferedImage> transformer) {
    mbOutput[0] = 1.0f / 0xffff;
    mbOutput[1] = 0;
  }

  /**
   * Perform NetworkAnalyzer for custom layers involving this model type
   * 
   * Return true if we handled the layer, false otherwise
   */
  public boolean processLayer(NetworkAnalyzer networkAnalyzer, int layerIndex) {
    return false;
  }

  public void describeLayer(NetworkAnalyzer an, Layer layer, StringBuilder sb) {
    throw die("Unsupported operation");
  }

  public static ModelHandler construct(NeuralNetwork network) {
    return construct(new ModelWrapper(network));
  }

  private static ModelHandler construct(ModelWrapper model) {

    ModelHandler handler = null;
    switch (model.projectType()) {

    case YOLO:
      handler = new YoloModelHandler();
      break;

    default:
      throw die("not supported:", model.projectType());
    }
    handler.setModel(model);
    return handler;
  }

}
