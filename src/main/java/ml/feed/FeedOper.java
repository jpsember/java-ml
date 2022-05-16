package ml.feed;

import static js.base.Tools.*;

import java.util.List;

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
    if (alert("temporary")) {
      exp();
      return;
    }

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

  private void exp() {
    List<Integer> vals = arrayList();
    int targetMax = 10;
    vals.add(targetMax);

    while (targetMax < 500) {
      targetMax += 10;
    }

    int tMax = 1200;
    double power = 0.5f;

    int s = 8;
    for (int i = 1; i <= s; i++) {
      double ni = i / (double) s;
      double ei = tMax * Math.pow(ni, power);
      pr(i, ei);
      int ti = target(ni, power, tMax);
      double nic = fraction(ti, tMax, power);
      pr(ei, ti, nic);
    }
  }

  /**
   * Calculate target for a particular fraction
   */
  private static int target(double fraction, double power, int targetMax) {
    return (int) Math.round(targetMax * Math.pow(fraction, power));
  }

  /**
   * Calculate fraction for a particular target
   */
  private static double fraction(int target, int targetMax, double power) {
    return Math.pow(target / (double) targetMax, 1/power);
  }

 
}
