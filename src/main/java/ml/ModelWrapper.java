package ml;

import static js.base.Tools.*;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.List;

import js.base.BaseObject;
import js.data.AbstractData;
import js.geometry.IPoint;
import js.graphics.ImgUtil;
import js.graphics.ScriptElement;
import js.graphics.ScriptUtil;
import js.graphics.gen.Script;
import js.json.JSMap;
import ml.classifier.ClassifierModelWrapper;
import ml.yolo.YoloModelWrapper;
import ml.yolo.YoloUtil;
import gen.*;

/**
 * An intelligent wrapper around datagen model class (Yolo, etc.) Parses
 * information about a neural network, its input image dimensions, and whatnot
 * to provide a handier way of manipulating these things.
 */
public abstract class ModelWrapper extends BaseObject {

  public static ModelWrapper constructFor(NeuralNetwork network) {

    ModelWrapper handler = null;
    switch (network.projectType()) {

    case YOLO:
      handler = new YoloModelWrapper();
      break;

    case CLASSIFIER:
      handler = new ClassifierModelWrapper();
      break;

    default:
      throw die("not supported:", network.projectType());
    }
    handler.init(network);
    return handler;
  }

  public static ModelWrapper constructFor(File baseDirectoryOrNull, NeuralNetwork networkOrNull,
      File networkPath) {
    return constructFor(NetworkUtil.resolveNetwork(baseDirectoryOrNull, networkOrNull, networkPath));
  }

  private void init(NeuralNetwork network) {
    todo("make various things final");
    mNetwork = NetworkUtil.validateNetwork(network);
    mInputImageVolume = determineInputImageVolume(network);
    mInputImageChannels = network.modelConfig().getInt("image_channels");
    mInputImagePlanarSize = VolumeUtil.spatialDimension(mInputImageVolume);
    mInputImageVolumeProduct = VolumeUtil.product(mInputImageVolume);
    mModelConfig = parseModelConfig(network.projectType(), network.modelConfig());
  }

  public void transformAnnotations(List<ScriptElement> in, List<ScriptElement> out,
      TransformWrapper transform) {
    for (ScriptElement orig : in)
      out.add(orig.applyTransform(transform.matrix()));
  }

  /**
   * Construct object to provide various model-specific services
   */
  public abstract ModelServiceProvider buildModelServiceProvider();

  // ------------------------------------------------------------------
  // Training progress
  // ------------------------------------------------------------------

  protected BufferedImage constructBufferedImage(float[] pixels) {
    return ImgUtil.floatsToBufferedImage(pixels, inputImagePlanarSize(), inputImageVolume().depth());
  }

  protected RuntimeException notSupported() {
    return die("Unsupported; project type:", projectType());
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

  public static <T extends AbstractData> T parseModelConfig(NetworkProjectType projectType, JSMap jsMap) {
    AbstractData prototype;
    switch (projectType) {
    default:
      throw die("unsupported project type:", projectType);
    case YOLO:
      prototype = Yolo.DEFAULT_INSTANCE;
      break;
    case CLASSIFIER:
      prototype = Classifier.DEFAULT_INSTANCE;
      break;
    }
    return (T) prototype.parse(jsMap);
  }

  /**
   * Get the AbstractMessage representing this model (e.g. Yolo, Classifier)
   */
  public <T extends AbstractData> T modelConfig() {
    return (T) mModelConfig;
  }

  public final NeuralNetwork network() {
    return mNetwork;
  }

  public final Vol inputImageVolume() {
    return mInputImageVolume;
  }

  public final IPoint inputImagePlanarSize() {
    return mInputImagePlanarSize;
  }

  public final int inputImageChannels() {
    return mInputImageChannels;
  }

  public final int inputImageVolumeProduct() {
    return mInputImageVolumeProduct;
  }

  public final NetworkProjectType projectType() {
    return network().projectType();
  }

  public final long[] inputImageTensorShape() {
    long[] shape = new long[3];
    Vol v = inputImageVolume();
    shape[0] = v.depth();
    shape[1] = v.height(); // Note order of y,x
    shape[2] = v.width();
    return shape;
  }

  @Deprecated // Refactor to use inheritance
  public int imageLabelFloatCount() {
    switch (network().projectType()) {
    case YOLO:
      return YoloUtil.imageLabelFloatCount(modelConfig());
    default:
      throw die("unsupported for project type:", projectType());
    }
  }

  private static Vol determineInputImageVolume(NeuralNetwork network) {
    JSMap modelConfig = network.modelConfig();
    IPoint imageSize = IPoint.get(modelConfig, "image_size");
    int imageChannels = modelConfig.getInt("image_channels");
    return VolumeUtil.build(imageSize.x, imageSize.y, imageChannels);
  }

  private NeuralNetwork mNetwork;
  private Vol mInputImageVolume;
  private AbstractData mModelConfig;
  private IPoint mInputImagePlanarSize;
  private int mInputImageChannels;
  private int mInputImageVolumeProduct;

}
