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

    int capacity = 8;
    float power = 0.5f;

    for (int j = 100; j < 1000; j += 10) {
      vals.add(j);

      if (vals.size() > capacity) {
        // Throw out value whose neighbors have fractions closest together
        double[] frac = new double[vals.size()];
        int i = INIT_INDEX;
        for (int target : vals) {
          i++;
          frac[i] = fraction(target, targetMax, power);
        }
        double minDiff = 0;
        int minIndex = -1;
        for (int q = 1; q < vals.size() - 1; q++) {
          double diff = frac[q + 1] - frac[q - 1];
          if (minIndex < 0 || minDiff > diff) {
            minIndex = q;
            minDiff = diff;
          }
        }
        vals.remove(minIndex);
      }
      pr("values:", vals);
    }
  }

  /**
   * Calculate epoch for a particular fraction
   */
  /* private */ static int epoch(double fraction, double power, int epochMax) {
    return (int) Math.round(epochMax * Math.pow(fraction, power));
  }

  /**
   * Calculate fraction for a particular epoch
   */
  private static double fraction(int epoch, int maxEpoch, double power) {
    return Math.pow(epoch / (double) maxEpoch, 1 / power);
  }

}
