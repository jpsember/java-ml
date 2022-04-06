package ml;

import static js.base.Tools.*;

import java.io.File;

import gen.CompileNetworkConfig;
import gen.NeuralNetwork;
import js.file.Files;
import js.app.AppOper;

public class CompileNetworkOper extends AppOper {

  @Override
  public String userCommand() {
    return "compilenetwork";
  }

  @Override
  public String getHelpDescription() {
    return "parse a network and fill in any missing fields";
  }

  @Override
  public CompileNetworkConfig defaultArgs() {
    return CompileNetworkConfig.DEFAULT_INSTANCE;
  }

  @Override
  public void perform() {
    mConfig = config();

    NeuralNetwork netIn = network();
    NetworkAnalyzer analyzer = NetworkAnalyzer.build(netIn);
    NeuralNetwork netOut = analyzer.result();
    if (netIn.equals(netOut)) {
      pr("...no changes");
      return;
    }
    mNetwork = netOut;

    pr("...writing changed version");
    files().writePretty(networkArgs().path(), mNetwork);
  }

  private CompileNetworkConfig networkArgs() {
    return mConfig;
  }

  private NeuralNetwork network() {
    if (mNetwork == null) {
      File path = Files.assertExists(mConfig.path(), "network path argument");
      mNetwork = Files.parseAbstractData(NeuralNetwork.DEFAULT_INSTANCE, path);
    }
    return mNetwork;
  }

  private NeuralNetwork mNetwork;
  private CompileNetworkConfig mConfig;

}
