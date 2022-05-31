package ml;

import static js.base.Tools.*;

import java.io.DataOutputStream;
import java.io.File;
import java.util.List;

import js.base.BaseObject;
import js.base.BasePrinter;
import js.data.AbstractData;
import js.data.DataUtil;
import js.file.Files;
import js.geometry.IPoint;
import js.graphics.ScriptElement;
import js.json.JSMap;

import gen.Classifier;
import gen.ImageSetInfo;
import gen.Layer;
import gen.NetworkProjectType;
import gen.NeuralNetwork;
import gen.Vol;
import gen.Yolo;

/**
 * An intelligent wrapper around datagen model class (Yolo, etc.) Parses
 * information about a neural network, its input image dimensions, and whatnot
 * to provide a handier way of manipulating these things.
 */
public abstract class ModelWrapper<T extends AbstractData> extends BaseObject {

  /**
   * Construct an appropriate concrete ModelWrapper for a network
   */
  public static ModelWrapper constructFor(NeuralNetwork networkOrNull, File networkPathOrNull) {
    NeuralNetwork network = NetworkUtil.resolveNetwork(networkOrNull, networkPathOrNull);

    ModelWrapper handler = null;
    switch (network.projectType()) {

    case YOLO:
      handler = new ml.yolo.YoloModelWrapper();
      break;

    case CLASSIFIER:
      handler = new ml.classifier.ClassifierModelWrapper();
      break;

    default:
      throw notSupported(network.projectType());
    }
    handler.auxInit(network);
    return handler;
  }

  private void auxInit(NeuralNetwork network) {
    mNetwork = NetworkUtil.validateNetwork(network);
    mInputImageVolume = NetworkUtil.determineInputImageVolume(network);
    mInputImageChannels = network.modelConfig().getInt("image_channels");
    mInputImagePlanarSize = VolumeUtil.spatialDimension(mInputImageVolume);
    mInputImageVolumeProduct = VolumeUtil.product(mInputImageVolume);
    mModelConfig = (T) parseModelConfig(network.projectType(), network.modelConfig());
    init();
    mOutputLayer = constructLabelBuffer();
  }

  /**
   * Optional initialization of subclasses; default does nothing
   */
  public void init() {
  }

  /**
   * Get the default buffer used for storing an image's labels. At present there
   * is a single such buffer; but in the future, we may want to have a second
   * buffer, in case the input and output buffers have different sizes or types
   */
  public abstract Object constructLabelBuffer();

  /**
   * Throw an exception for an unsupported operation with this type of model
   */
  public final RuntimeException modelNotSupported(Object... messageObjects) {
    return die("Unsupported; project type:", projectType(), ";", BasePrinter.toString(messageObjects));
  }

  /**
   * Perform NetworkAnalyzer for custom layers involving this model type
   * 
   * Return true if we handled the layer, false otherwise
   */
  public boolean processLayer(NetworkAnalyzer networkAnalyzer, Layer.Builder layer) {
    return false;
  }

  /**
   * Describe custom layers associated with this network
   */
  public void describeLayer(NetworkAnalyzer an, Layer layer, StringBuilder sb) {
    modelNotSupported();
  }

  private static AbstractData parseModelConfig(NetworkProjectType projectType, JSMap jsMap) {
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
    return prototype.parse(jsMap);
  }

  /**
   * Get the AbstractMessage representing this model (e.g. Yolo, Classifier)
   */
  public T modelConfig() {
    return mModelConfig;
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
  public final void accept(LabelledImage labelledImage) {
    writeImage(labelledImage);
    transformScreditToModelInput(labelledImage.annotations());
    writeLabels();
  }

  /**
   * Fill in information fields. Some fields may have already been filled in
   */
  public abstract void storeImageSetInfo(ImageSetInfo.Builder imageSetInfo);

  public Object transformScreditToModelInput(List<ScriptElement> scriptElements) {
    throw modelNotSupported("transformScreditToModelInput");
  }

  /**
   * Transform labels from their form as passed to the model back to
   * ScriptElements; for inspection / debug purposes only
   */
  public List<ScriptElement> transformModelInputToScredit() {
    throw modelNotSupported("transformModelInputToScredit");
  }

  /**
   * Get ImageSetInfo builder, constructing if necessary
   */
  public final ImageSetInfo.Builder imageSetInfo() {
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

  public final void writeImage(LabelledImage image) {
    switch (network().imageDataType()) {
    default:
      throw notSupported("image_data_type:", network().imageDataType());

    case FLOAT32:
      Files.S.write(DataUtil.floatsToBytesLittleEndian(image.pixelsF()), imageOutputStream());
      break;

    case UNSIGNED_BYTE:
      Files.S.write(image.pixelsB(), imageOutputStream());
      break;
    }
  }

  private void writeLabels() {
    switch (network().labelDataType()) {
    default:
      throw notSupported("label_data_type:", network().labelDataType());

    case FLOAT32:
      writeLabels(labelBufferFloats());
      break;

    case UNSIGNED_BYTE:
      writeLabels(labelBufferBytes());
      break;
    }

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

  /**
   * Get the buffer used for storing an image's labels, as an array of floats
   */
  public final float[] labelBufferFloats() {
    return (float[]) mOutputLayer;
  }

  /**
   * Get the buffer used for storing an image's labels, as an array of bytes
   */
  public final byte[] labelBufferBytes() {
    return (byte[]) mOutputLayer;
  }

  // ------------------------------------------------------------------

  private NeuralNetwork mNetwork;
  private Vol mInputImageVolume;
  private T mModelConfig;
  private IPoint mInputImagePlanarSize;
  private int mInputImageChannels;
  private int mInputImageVolumeProduct;
  private DataOutputStream mImageOutputStream;
  private DataOutputStream mLabelOutputStream;
  private ImageSetInfo.Builder mImageSetInfo;
  private Object mOutputLayer;
}
