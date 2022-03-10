package ml;

import static js.base.Tools.*;

import js.geometry.MyMath;
import js.json.JSMap;
import gen.Stats;

public final class StatCalculator {

  public static StatCalculator forValues(float[] floats) {
    StatCalculator s = new StatCalculator();
    for (float f : floats)
      s.add(f);
    return s;
  }

  public static Stats verify(Stats stats) {
    if (stats == null || stats.standardDeviation() == 0f)
      throw die("Stats is null or undefined:", INDENT, stats);
    return stats;
  }

  public StatCalculator withName(String name) {
    checkState(mName == null);
    mName = name;
    return this;
  }

  public String name() {
    return mName;
  }

  public void add(float value) {
    verifyFinite(value);

    if (mSampleCount == 0) {
      mMin = value;
      mMax = value;

      // Apparently it is sufficient to choose a location parameter of any value within the range
      mLocationParameter = value;
    }
    if (value < mMin)
      mMin = value;
    if (value > mMax)
      mMax = value;

    float adjustedValue = value - mLocationParameter;

    mSampleSum += adjustedValue;
    mSampleSquaredSum += adjustedValue * adjustedValue;
    mSampleCount++;
  }

  public int sampleCount() {
    return mSampleCount;
  }

  public float mean() {
    ensureSamplesExist();
    return mSampleSum / mSampleCount + mLocationParameter;
  }

  public float variance() {
    ensureSamplesExist();
    float offsetMean = mSampleSum / mSampleCount;
    float offsetMeanSquared = offsetMean * offsetMean;
    float offsetMeanSampleSquared = mSampleSquaredSum / mSampleCount;
    return offsetMeanSampleSquared - offsetMeanSquared;
  }

  public float standardDeviation() {
    return MyMath.sqrtf(variance());
  }

  @Override
  public String toString() {
    return toJson().prettyPrint();
  }

  public JSMap toJson() {
    JSMap m = map();
    if (mName != null)
      m.put("", mName);
    m.put("count", mSampleCount);
    if (mSampleCount != 0) {
      m.put("min", mMin);
      m.put("max", mMax);
      m.put("mean", mean());
      float var = variance();
      m.put("var", var);
      m.put("dev", MyMath.sqrtf(var));
    }
    return m;
  }

  private void ensureSamplesExist() {
    if (mSampleCount == 0)
      throw new IllegalStateException("Sample count is zero");
  }

  private String mName;
  private int mSampleCount;
  private float mMin;
  private float mMax;
  private float mSampleSum;
  private float mSampleSquaredSum;

  // An offset to make the variance calculation numerically stable
  //
  private float mLocationParameter;

}
