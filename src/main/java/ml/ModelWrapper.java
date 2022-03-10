package ml;

import static js.base.Tools.*;

import gen.NeuralNetwork;
import js.base.BaseObject;
import js.data.AbstractData;
import js.geometry.IPoint;
import js.json.JSMap;
import ml.VolumeUtil;
import gen.*;

/**
 * An intelligent wrapper around pojo model class (Yolo, etc.) Parses
 * information about a neural network, its input image dimensions, and whatnot
 * to provide a handier way of manipulating these things.
 */
public final class ModelWrapper extends BaseObject {

  public ModelWrapper(NeuralNetwork network) {
    if (network.projectType() == NetworkProjectType.UNKNOWN)
      throw die("Unknown project type for network:", INDENT, network);
    mNetwork = network.build();
    mInputImageVolume = determineInputImageVolume(network);
    mInputImageChannels = network.modelConfig().getInt("image_channels");
    mInputImagePlanarSize = VolumeUtil.spatialDimension(mInputImageVolume);
    mInputImageVolumeProduct = VolumeUtil.product(mInputImageVolume);
    mModelConfig = parseModelConfig(network.projectType(), network.modelConfig());
  }

  @SuppressWarnings("unchecked")
  public static <T extends AbstractData> T parseModelConfig(NetworkProjectType projectType, JSMap jsMap) {
    AbstractData prototype;
    switch (projectType) {
    default:
      throw die("unsupported project type:", projectType);
    case YOLO:
      prototype = Yolo.DEFAULT_INSTANCE;
      break;
    }
    return (T) prototype.parse(jsMap);
  }

  /**
   * Get the AbstractMessage representing this model (e.g. Yolo, ImageGrid)
   */
  @SuppressWarnings("unchecked")
  public <T extends AbstractData> T modelConfig() {
    return (T) mModelConfig;
  }

  public NeuralNetwork network() {
    return mNetwork;
  }

  public Vol inputImageVolume() {
    return mInputImageVolume;
  }

  public IPoint inputImagePlanarSize() {
    return mInputImagePlanarSize;
  }

  public int inputImageChannels() {
    return mInputImageChannels;
  }

  public int inputImageVolumeProduct() {
    return mInputImageVolumeProduct;
  }

  public NetworkProjectType projectType() {
    return network().projectType();
  }

  public long[] inputImageTensorShape() {
    long[] shape = new long[3];
    Vol v = inputImageVolume();
    shape[0] = v.depth();
    shape[1] = v.height(); // Note order of y,x
    shape[2] = v.width();
    return shape;
  }

  public IPoint blockGrid() {
    if (mBlockGrid == null) {
      JSMap m = mModelConfig.toJson().asMap();
      mBlockSize = IPoint.get(m, "block_size");
      mBlockGrid = new IPoint(mInputImagePlanarSize.x / mBlockSize.x, mInputImagePlanarSize.y / mBlockSize.y);
    }
    return mBlockGrid;
  }

  public IPoint blockSize() {
    blockGrid();
    return mBlockSize;
  }

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

  private final NeuralNetwork mNetwork;
  private final Vol mInputImageVolume;
  private final AbstractData mModelConfig;
  private final IPoint mInputImagePlanarSize;
  private final int mInputImageChannels;
  private final int mInputImageVolumeProduct;

  private IPoint mBlockGrid;
  private IPoint mBlockSize;
}
