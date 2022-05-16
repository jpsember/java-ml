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
    FeedAlg alg = null;
    String algName = "ml.feed.Alg" + config().alg();
    Class klass = null;
    try {
      klass = Class.forName(algName);
      alg = (FeedAlg) klass.newInstance();
    } catch (Throwable e) {
      throw badArg("Can't construct algorithm! Error:", e.getMessage());
    }
    alg.setVerbose(verbose());
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
