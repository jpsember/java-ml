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
import js.graphics.ScriptUtil;
import js.graphics.gen.Script;
import js.json.JSMap;

import gen.Classifier;
import gen.ImageSetInfo;
import gen.LabelForm;
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
public abstract class ModelWrapper<T extends AbstractData> extends BaseObject {

  /**
   * Construct an appropriate concrete ModelWrapper for a network
   */
  public static ModelWrapper constructFor(NeuralNetwork networkOrNull, File networkPathOrNull) {
    NeuralNetwork network = NetworkUtil.resolveNetwork(networkOrNull, networkPathOrNull);

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

  private void auxInit(NeuralNetwork network) {
    mNetwork = NetworkUtil.validateNetwork(network);
    mInputImageVolume = NetworkUtil.determineInputImageVolume(network);
    mInputImageChannels = network.modelConfig().getInt("image_channels");
    mInputImagePlanarSize = VolumeUtil.spatialDimension(mInputImageVolume);
    mInputImageVolumeProduct = VolumeUtil.product(mInputImageVolume);
    mModelConfig = (T) parseModelConfig(network.projectType(), network.modelConfig());
    init();
  }

  /**
   * Optional initialization of subclasses; default does nothing
   */
  public void init() {
  }

  /**
   * Apply a transformation to ScriptElements. Default implementation applies
   * the supplied transformation to the elements. Classifier wrapper overrides
   * this to copy the elements unchanged, as each element's geometry is not
   * used, only its category
   */
  public void transformAnnotations(List<ScriptElement> in, List<ScriptElement> out,
      TransformWrapper transform) {
    for (ScriptElement orig : in)
      out.add(orig.applyTransform(transform.matrix()));
  }

  /**
   * Throw an exception for an unsupported operation with this type of model
   */
  public final RuntimeException modelNotSupported(Object... messageObjects) {
    return die("Unsupported; project type:", projectType(), ";", BasePrinter.toString(messageObjects));
  }

  /**
   * Examine script and extract appropriate elements from it by appending to
   * target
   */
  @Deprecated // Not sure this is used...
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
  public abstract void accept(LabelledImage labelledImage);

  /**
   * Process an image and its annotations, converting to form suitable for
   * training
   * 
   * @param imagePixelsArray
   *          a primitive array of a datatype compatible with the network's
   *          imageDataType field
   */
  public abstract void accept(Object imagePixelsArray, List<ScriptElement> scriptElementList);

  /**
   * Fill in information fields. Some fields may have already been filled in
   */
  public abstract void storeImageSetInfo(ImageSetInfo.Builder imageSetInfo);

  /**
   * Convert image labels from one form to another
   */
  public Object transformLabels(LabelForm fromForm, Object input, LabelForm toForm) {
    if (fromForm == LabelForm.SCREDIT && toForm == LabelForm.MODEL_INPUT)
      return transformScreditToModelInput((List<ScriptElement>) input);

    if (fromForm == LabelForm.MODEL_INPUT && toForm == LabelForm.SCREDIT)
      return transformModelInputToScredit(input);

    throw notSupported("parseLabels not supported for project", projectType(), "from", fromForm, "to",
        toForm);
  }

  public Object transformScreditToModelInput(List<ScriptElement> scriptElements) {
    throw modelNotSupported("transformScreditToModelInput");
  }

  public List<ScriptElement> transformModelInputToScredit(Object input) {
    throw modelNotSupported("transformModelInputToScredit");
  }

  /**
   * Parse model output to a Script
   */
  @Deprecated
  public void parseInferenceResult(byte[] modelOutput, int confidencePct, Script.Builder script) {
    modelNotSupported();
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

  public final void writeImage(Object imagePixelArray) {
    switch (network().imageDataType()) {
    default:
      throw notSupported("image_data_type:", network().imageDataType());
    case FLOAT32: {
      float[] imageFloats = (float[]) imagePixelArray;
      Files.S.write(DataUtil.floatsToBytesLittleEndian(imageFloats), imageOutputStream());
    }
      break;

    case UNSIGNED_BYTE: {
      byte[] imageBytes = (byte[]) imagePixelArray;
      Files.S.write(imageBytes, imageOutputStream());
    }
      break;

    }
  }

  /**
   * Write labels associated with an image to the label's output stream
   */
  public final void writeLabels(byte[] labelBytes) {
    checkArgument(labelBytes.length == imageSetInfo().labelLengthBytes());
    mLastLabelBytesWritten = labelBytes;
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
   * For inspection purposes, get the last bytes written via writeLabels()
   */
  @Deprecated // Add support to optionally convert to model's label format (floats)
  public final byte[] lastLabelBytesWritten() {
    return checkNotNull(mLastLabelBytesWritten, "no lastLabelBytesWritten available");
  }

  public Object getLabelBuffer() {
    throw modelNotSupported("getLabelBuffer");
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
  private byte[] mLastLabelBytesWritten;

}
