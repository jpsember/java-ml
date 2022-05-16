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
//import ml.yolo.YoloModelHandler;
import gen.AugmentationConfig;
import gen.Layer;
import gen.NeuralNetwork;
import gen.PlotInferenceResultsConfig;

/**
 * Abstract class representing a model for a particular type of ml project.
 * 
 * To be merged with ModelWrapper at some point
 */
public abstract class ModelHandler extends BaseObject {

  @Deprecated
  public void addImageRecordFilters(ImageHandler imageProcessor) {
  }

  public final ModelWrapper model() {
    return mModelConfig;
  }

  public final void setModel(ModelWrapper model) {
    todo("!Should this class and ModelWrapper be merged?");
    mModelConfig = model;
  }

  public final ImageTransformer<BufferedImage> buildImageTransformer(AugmentationConfig augmentationConfig,
      Random random) {
    ImageTransformer<BufferedImage> transformer;
    if (model().network().monochromeSourceImages())
      transformer = new MonochromeImageTransformer();
    else
      transformer = new ColorImageTransformer();
    transformer.init(this, augmentationConfig, random);
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

  protected RuntimeException notSupported() {
    return die("Unsupported; project type:", model().projectType());
  }

  /**
   * Examine script and extract appropriate elements from it by appending to
   * target
   */
  public void extractShapes(Script script, List<ScriptElement> target) {
    throw notSupported();
  }

  protected final void assertNoMixing(Script script) {
    if (!ScriptUtil.rectElements(script).isEmpty() && !ScriptUtil.polygonElements(script).isEmpty())
      throw die("Cannot mix boxes and polygons");
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

//    case YOLO:
//      handler = new YoloModelHandler();
//      break;

    case CLASSIFIER:
      handler = new ClassifierModelHandler();
      break;

    default:
      throw die("not supported:", model.projectType());
    }
    handler.setModel(model);
    return handler;
  }

  private ModelWrapper mModelConfig;

}
