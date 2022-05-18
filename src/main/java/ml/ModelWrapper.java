package ml;

import static js.base.Tools.*;

import java.io.DataOutputStream;
import java.io.File;
import java.util.List;

import js.base.BaseObject;
import js.data.AbstractData;
import js.data.DataUtil;
import js.file.Files;
import js.geometry.IPoint;
import js.graphics.ScriptElement;
import js.graphics.ScriptUtil;
import js.graphics.gen.Script;
import js.graphics.gen.ScriptElementList;
import js.json.JSMap;

import gen.Classifier;
import gen.ImageSetInfo;
import gen.Layer;
import gen.NetworkProjectType;
import gen.NeuralNetwork;
import gen.TransformWrapper;
import gen.Vol;
import gen.Yolo;

import ml.classifier.ClassifierModelWrapper;
import ml.yolo.YoloModelWrapper;

/**
 * An intelligent wrapper around datagen model class (Yolo, etc.) Parses
 * information about a neural network, its input image dimensions, and whatnot
 * to provide a handier way of manipulating these things.
 */
public abstract class ModelWrapper extends BaseObject {

  /**
   * Construct an appropriate concrete ModelWrapper for a network
   */
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
      throw notSupported(network.projectType());
    }
    handler.auxInit(network);
    return handler;
  }

  public static ModelWrapper constructFor(File baseDirectoryOrNull, NeuralNetwork networkOrNull,
      File networkPath) {
    return constructFor(NetworkUtil.resolveNetwork(baseDirectoryOrNull, networkOrNull, networkPath));
  }

  private void auxInit(NeuralNetwork network) {
    mNetwork = NetworkUtil.validateNetwork(network);
    mInputImageVolume = determineInputImageVolume(network);
    mInputImageChannels = network.modelConfig().getInt("image_channels");
    mInputImagePlanarSize = VolumeUtil.spatialDimension(mInputImageVolume);
    mInputImageVolumeProduct = VolumeUtil.product(mInputImageVolume);
    mModelConfig = parseModelConfig(network.projectType(), network.modelConfig());
    init();
  }

  /**
   * Optional initialization of subclasses; default does nothing
   */
  public void init() {
  }

  public void transformAnnotations(List<ScriptElement> in, List<ScriptElement> out,
      TransformWrapper transform) {
    for (ScriptElement orig : in)
      out.add(orig.applyTransform(transform.matrix()));
  }

  public RuntimeException modelNotSupported() {
    return die("Unsupported; project type:", projectType());
  }

  /**
   * Examine script and extract appropriate elements from it by appending to
   * target
   */
  public void extractShapes(Script script, List<ScriptElement> target) {
    throw modelNotSupported();
  }

  /**
   * Raise exception if there is a mixture of rectangles and polygons in a
   * script
   */
  public final void assertNoMixing(Script script) {
    if (!ScriptUtil.rectElements(script).isEmpty() && !ScriptUtil.polygonElements(script).isEmpty())
      badArg("Cannot mix rectangles and polygons");
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

  public final void setImageStream(DataOutputStream imageStream) {
    mImageOutputStream = imageStream;
  }

  public final void setLabelStream(DataOutputStream labelStream) {
    mLabelOutputStream = labelStream;
  }

  public final DataOutputStream imageOutputStream() {
    return mImageOutputStream;
  }

  public final DataOutputStream labelOutputStream() {
    return mLabelOutputStream;
  }

  /**
   * Process an image and its annotations, converting to form suitable for
   * training
   */
  public abstract void accept(float[] image, ScriptElementList scriptElementList);

  /**
   * Fill in information fields. Some fields may have already been filled in
   */
  public abstract void storeImageSetInfo(ImageSetInfo.Builder imageSetInfo);

  /**
   * Parse model output to a Script
   */
  public void parseInferenceResult(byte[] modelOutput, Script.Builder script) {
    modelNotSupported();
  }

  /**
   * Get ImageSetInfo builder, constructing if necessary
   */
  public ImageSetInfo.Builder imageSetInfo() {
    if (mImageSetInfo == null) {
      mImageSetInfo = ImageSetInfo.newBuilder();
      storeImageSetInfo(mImageSetInfo);
      checkArgument(mImageSetInfo.imageLengthBytes() > 0 && mImageSetInfo.labelLengthBytes() > 0);
    }
    return mImageSetInfo;
  }

  // ------------------------------------------------------------------
  // Writing training images and labels
  // ------------------------------------------------------------------

  public final void writeImage(float[] imageFloats) {
    Files.S.write(DataUtil.floatsToBytesLittleEndian(imageFloats), imageOutputStream());
  }

  /**
   * Write labels associated with an image to the label's output stream
   */
  public final void writeLabels(byte[] labelBytes) {
    checkArgument(labelBytes.length == imageSetInfo().labelLengthBytes());
    Files.S.write(labelBytes, labelOutputStream());
  }

  /**
   * Write labels associated with an image to the label's output stream,
   * converting to bytes
   */
  public final void writeLabels(int[] labelInts) {
    writeLabels(DataUtil.intsToBytesLittleEndian(labelInts));
  }

  /**
   * Write labels associated with an image to the label's output stream,
   * converting to bytes
   */
  public final void writeLabels(float[] labelFloats) {
    writeLabels(DataUtil.floatsToBytesLittleEndian(labelFloats));
  }

  // ------------------------------------------------------------------

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
  private DataOutputStream mImageOutputStream;
  private DataOutputStream mLabelOutputStream;
  private ImageSetInfo.Builder mImageSetInfo;

}
