package ml;

import java.awt.image.BufferedImage;
import java.util.List;
import java.util.Random;

import static js.base.Tools.*;

import js.graphics.ImgUtil;
import js.graphics.ScriptElement;
import js.graphics.ScriptUtil;
import js.graphics.gen.Script;
import js.base.BaseObject;
import ml.classifier.ClassifierModelHandler;
import ml.yolo.YoloModelHandler;
import gen.AugmentationConfig;
import gen.Layer;
import gen.NeuralNetwork;
import gen.PlotInferenceResultsConfig;
import gen.Stats;

public abstract class ModelHandler extends BaseObject {

  public abstract void addImageRecordFilters(ImageHandler imageProcessor);

  public final ModelWrapper model() {
    return mModelConfig;
  }

  public final void setModel(ModelWrapper model) {
    todo("Should this class and ModelWrapper be merged?");
    mModelConfig = model;
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
   * Construct object to provide various model-specific services
   */
  public abstract ModelServiceProvider buildModelServiceProvider();

  // ------------------------------------------------------------------
  // Training progress
  // ------------------------------------------------------------------

  /**
   * Plot inference results in some user-friendly form (e.g. a scredit project)
   */
  @Deprecated // This is subsumed by the ModelServiceProvider.parseInferenceResult()
  public abstract void plotInferenceResults(PlotInferenceResultsConfig config);

  protected BufferedImage constructBufferedImage(float[] pixels) {
    return ImgUtil.floatsToBufferedImage(pixels, model().inputImagePlanarSize(),
        model().inputImageVolume().depth());
  }

  private ModelWrapper mModelConfig;

  protected RuntimeException notSupported() {
    return die("Unsupported; project type:", model().projectType());
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

    case CLASSIFIER:
      handler = new ClassifierModelHandler();
      break;

    default:
      throw die("not supported:", model.projectType());
    }
    handler.setModel(model);
    return handler;
  }

}
