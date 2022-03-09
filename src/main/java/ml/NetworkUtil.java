package ml;

import static js.base.Tools.*;

import gen.Layer;
import gen.NetworkProjectType;
import gen.NeuralNetwork;
import gen.Vol;

public final class NetworkUtil {

  public final static int VERSION = 0;

  public static NeuralNetwork applyDefaults(NeuralNetwork network) {
    if (network.version() != VERSION)
      throw die("Unexpected version:", network.version(), "Expected:", VERSION);
    if (network.projectType() == NetworkProjectType.UNKNOWN)
      throw die("Unknown project type");
    return network.build();
  }

  public static void calcWeightsForConv(Layer.Builder layer, Vol filterBox, int numFilters, Vol outputBox) {
    int filterVolume = VolumeUtil.product(filterBox);
    // I'm going to assume I can fold any batch norm variables so there is no difference
    // Add one for bias
    layer.numWeights((filterVolume + 1) * numFilters);
  }

  public static void calcWeightsForFC(Layer.Builder layer, int inputVolume, int outputVolume) {
    // Add one for bias
    layer.numWeights((inputVolume + 1) * outputVolume);
  }

  public static NeuralNetwork ensureValid(NeuralNetwork n) {
    if (n.alpha() < 0 || n.dropoutConv() < 0 || n.dropoutFc() < 0)
      throw die("no-longer-supported negative values:", INDENT, n);
    return n.build();
  }

}
