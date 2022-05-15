package ml.feed;

import static js.base.Tools.*;

import gen.FeedConfig;
import js.app.AppOper;

public class FeedOper extends AppOper {

  @Override
  public String userCommand() {
    loadTools();
    return "feed";
  }

  @Override
  public String getHelpDescription() {
    return "Investigate strategies for feeding training data to Python code";
  }

  @Override
  public void perform() {
    FeedAlg alg;
    alg = new Alg1();
    alg.setConfig(config());
    alg.perform();
  }

  @Override
  public FeedConfig defaultArgs() {
    return FeedConfig.DEFAULT_INSTANCE;
  }

  @Override
  public FeedConfig config() {
    return super.config();
  }

}
