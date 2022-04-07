package ml;

import js.base.BaseObject;
import js.base.BasePrinter;
import js.geometry.IPoint;

import static js.base.Tools.*;
import java.util.List;

import ml.ModelHandler;
import js.json.JSList;
import js.json.JSMap;
import ml.ModelWrapper;
import ml.NetworkUtil;
import ml.VolumeUtil;
import gen.*;

/**
 * Analyze a NeuralNetwork, and apply default values where appropriate
 */
public final class NetworkAnalyzer extends BaseObject {

  public static NetworkAnalyzer build(NeuralNetwork network) {
    NetworkAnalyzer analyzer = new NetworkAnalyzer(network);
    analyzer.result();
    return analyzer;
  }

  private NetworkAnalyzer(NeuralNetwork network) {
    mNetwork = NetworkUtil.validateNetwork(network);
    mHandler = ModelHandler.construct(network());
    setVolume(model().inputImageVolume());
  }

  private ModelWrapper model() {
    return mHandler.model();
  }

  public NeuralNetwork result() {
    if (mResult != null)
      return mResult;

    mLayerBuilders = arrayList();
    for (Layer layer : network().layers())
      mLayerBuilders.add(layer.toBuilder());

    // If last layer isn't OUTPUT, add OUTPUT
    if (layerCount() == 0 || layer(layerCount() - 1).type() != LayerType.OUTPUT)
      mLayerBuilders.add(Layer.newBuilder().type(LayerType.OUTPUT));

    for (int i = 0; i < layerCount(); i++) {
      mLayerIndex = i;
      if (problemsFound())
        continue;
      analyzeLayer();
    }

    if (layerCount() <= 1)
      addProblem("too few layers");

    if (network().alpha() < 0 || network().dropoutHidden() < 0 || network().dropoutInput() < 0)
      addProblem("Deprecated negative values found in network fields");

    NeuralNetwork.Builder b = network().toBuilder();

    List<Layer> built = arrayList();
    for (Layer.Builder bd : mLayerBuilders)
      built.add(bd.build());
    b.layers(built);

    b.weightCount(weightCount());
    mResult = b.build();
    mLayerBuilders = null;
    return mResult;
  }

  public List<String> problemList() {
    return mProblems;
  }

  public boolean problemsFound() {
    return !problemList().isEmpty();
  }

  public void verifyNoProblemsFound() {
    if (problemsFound())
      throw die("Problems were found with network:", INDENT, problemList());
  }

  public ModelHandler handler() {
    return mHandler;
  }

  /**
   * Get the total number of weights defined so far
   */
  public long weightCount() {
    long sum = 0;
    for (int i = 0; i < layerCount(); i++)
      sum += layer(i).numWeights();
    return sum;
  }

  /**
   * Update the analysis by applying another layer
   */
  private void analyzeLayer() {

    Layer.Builder builder = layerBuilder(mLayerIndex);

    builder.inputVolume(volume());
    // Set output volume to same as input, in case it's a pass-through layer that doesn't explictly change it
    builder.outputVolume(volume());
    applyDefaults(builder);

    switch (builder.type()) {

    default:
      // Give model handler an opportunity to process this layer type
      // TODO: we may want to provide the layer index and layer count
      if (!handler().processLayer(this, mLayerIndex))
        addProblem("Unsupported layer:", builder.type());
      break;

    case CONV: {

      // Each filter has a small 'radius' that slides over the spatial extent of the input volume,
      // but extends through the ENTIRE depth of the input volume.

      // We are free to choose the *number* of such filters to apply, and this determines the
      // depth of the outut volume.

      // The output volume's spatial extent is that of the input volume, but optionally reduced
      // via having a stride greater than 1.  The 'pool' flag indicates we are to use a stride of 2,
      // which will decrease the output volume's width and height by a factor of 2.

      Vol inBox = volume();
      Vol workVolume = inBox;
      int numFilters = builder.filters();
      {
        IPoint stride = builder.stride();
        if (stride.x <= 0 || stride.y <= 0)
          throw badArg("unexpected stride, layer", mLayerIndex, INDENT, builder);
        // We should be able to call this even if stride is (1,1)
        workVolume = reduceVolumeForPooling(workVolume, builder.stride());
      }

      Vol outBox = VolumeUtil.withDepth(workVolume, numFilters);
      NetworkUtil.calcWeightsForConv(builder,
          VolumeUtil.build(builder.kernelWidth(), builder.kernelWidth(), inBox.depth()), numFilters, outBox);
      builder.outputVolume(outBox);
    }
      break;

    case LEAKY_RELU:
      break;

    case MAXPOOL: {
      IPoint stride = builder.stride();
      if (stride.x <= 0 || stride.y <= 0)
        throw badArg("unexpected stride, layer", mLayerIndex, INDENT, builder);
      builder.stride(stride);
      builder.outputVolume(reduceVolumeForPooling(volume(), stride));
    }
      break;

    case FC: {
      Vol newVolume = VolumeUtil.fibre(builder.filters());
      NetworkUtil.calcWeightsForFC(builder, VolumeUtil.product(volume()), VolumeUtil.product(newVolume));
      builder.outputVolume(newVolume);
    }
      break;

    case OUTPUT: {
      builder.outputVolume(VolumeUtil.fibre(VolumeUtil.product(volume())));
    }
      break;
    }

    if (problemsFound())
      return;

    setVolume(builder.outputVolume());
  }

  public JSMap toJson() {
    result();
    JSMap m = map();
    JSList lst = list();
    m.put("layers", lst);
    for (int i = 0; i < layerCount(); i++)
      lst.add(layer(i).toJson());
    if (!mProblems.isEmpty()) {
      m.put("problems", JSList.withUnsafeList(mProblems));
    }
    return m;
  }

  public void addProblem(Object... messages) {
    mProblems.add(BasePrinter.toString(messages));
  }

  public int layerCount() {
    if (mLayerBuilders != null)
      return mLayerBuilders.size();
    return mResult.layers().size();
  }

  public Layer layer(int index) {
    if (mLayerBuilders != null)
      return mLayerBuilders.get(index);
    return mResult.layers().get(index);
  }

  public Layer.Builder layerBuilder(int index) {
    ensureAnalysis();
    return mLayerBuilders.get(index);
  }

  private void ensureAnalysis() {
    checkState(mLayerBuilders != null, "not in analysis stage");
  }

  /**
   * Examine layer feeding into the current one being analyzed, and if it is
   * CONV or FC, disable dropout and batch normalization
   */
  public void applyNextToOutputFiltering() {
    ensureAnalysis();
    int i = mLayerIndex - 1;
    if (i >= 0) {
      Layer.Builder b = layerBuilder(i);
      if (b.type() == LayerType.FC || b.type() == LayerType.CONV) {
        b.dropout(null);
      }
    }
  }

  private NeuralNetwork network() {
    return mNetwork;
  }

  /**
   * Get the current output volume (or input volume, if no layers added yet)
   */
  private Vol volume() {
    return mVolume;
  }

  private void setVolume(Vol volume) {
    mVolume = VolumeUtil.ensureValid(volume);
  }

  /**
   * Fill in any default values for a Layer that are otherwise missing
   */
  private void applyDefaults(Layer.Builder layer) {

    switch (layer.type()) {

    default:
      break;

    case CONV:
      if (layer.filters() == 0)
        layer.filters(layer.inputVolume().depth());
      if (layer.kernelWidth() == null)
        layer.kernelWidth(network().kernelWidth());

      IPoint stride = layer.stride();
      if (stride == null) {
        if (layer.pool())
          stride = network().stride();
        else
          stride = IPoint.with(1, 1);
        layer.stride(stride);
      }
      applyDropoutAndBatchNormDefaults(layer);
      break;

    case LEAKY_RELU:
      if (layer.alpha() == null)
        layer.alpha(network().alpha());
      break;

    case MAXPOOL:
      if (layer.stride() == null) {
        layer.stride(network().stride());
      }
      break;

    case FC:
      applyDropoutAndBatchNormDefaults(layer);
      break;

    case OUTPUT:
      applyNextToOutputFiltering();
      break;
    }

    if (layer.dropout() != null && layer.dropout() > 0.5f)
      throw die("Suspected incorrect dropout value:", layer);
  }

  private void applyDropoutAndBatchNormDefaults(Layer.Builder layer) {
    if (layer.dropout() == null) {
      float defValue = (mLayerIndex == 0) ? network().dropoutInput() : network().dropoutHidden();
      layer.dropout(defValue);
    }
  }

  /**
   * Wraps VolumeUtil method, reporting exceptions as problems
   */
  private Vol reduceVolumeForPooling(Vol volume, IPoint stride) {
    try {
      return VolumeUtil.reducedForPooling(volume, stride.x, stride.y);
    } catch (Throwable t) {
      addProblem(t.getMessage());
      return volume;
    }
  }

  private final NeuralNetwork mNetwork;
  private final ModelHandler mHandler;
  private final List<String> mProblems = arrayList();
  private NeuralNetwork mResult;
  private int mLayerIndex;
  private List<Layer.Builder> mLayerBuilders;
  private Vol mVolume;
}
