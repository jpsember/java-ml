package ml;

import static js.base.Tools.*;

import java.io.File;

import gen.Layer;
import gen.NetworkProjectType;
import gen.NeuralNetwork;
import gen.Vol;
import js.data.DataUtil;

public final class NetworkUtil {

  public final static int VERSION = 0;

  public static ModelHandler constructModelHandler(File baseDirectoryOrNull, NeuralNetwork networkOrNull,
      File networkPath) {
    NeuralNetwork network = DataUtil.resolveField(baseDirectoryOrNull, NeuralNetwork.DEFAULT_INSTANCE,
        networkOrNull, networkPath);
    if (network == null)
      throw badArg("Cannot construct model handler");
    return ModelHandler.construct(network);
  }

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

  public static double ensureFinite(double value, double argument, String prompt) {
    if (!Double.isFinite(value)) {
      throw die("Failed to produce finite value for argument:", argument, "Location:", prompt);
    }
    return value;
  }

  private static final float MAX_EXPONENT = 12;
  private static final float EPSILON = (float) Math.exp(-MAX_EXPONENT);

  public static float logit(float value) {
    if (value < EPSILON)
      return -MAX_EXPONENT;
    if (value > 1 - EPSILON)
      return MAX_EXPONENT;
    double result = (float) Math.log(value / (1 - value));
    return (float) ensureFinite(result, value, "logit");
  }

  public static float sigmoid(float value) {
    return (float) ensureFinite((1 / (1 + Math.exp(-value))), value, "sigmoid");
  }

  public static float tanh(float value) {
    // see https://en.wikipedia.org/wiki/Activation_function
    float exp = (float) Math.exp(value);
    float exp2 = (float) Math.exp(-value);
    return (float) ensureFinite((exp - exp2) / (exp + exp2), value, "tanh");
  }

  public static float ln(float value) {
    if (value < EPSILON)
      return -MAX_EXPONENT;
    return (float) ensureFinite(Math.log(value), value, "ln");
  }

  public static float exp(float value) {
    return (float) ensureFinite(Math.exp(value), value, "ln");
  }

}
