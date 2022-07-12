package ml;

import java.util.Comparator;

import js.data.DataUtil;

class StatRecord {

  public static final String EPOCH = "epoch";
  public static final String LOSS = "loss";

  public StatRecord(String name) {
    mName = name;
  }

  public void update(float value) {
    mValue = value;
    if (mValueCount == 0)
      mSmoothedValue = value;
    else {
      float tau = 0.1f;
      mSmoothedValue = tau * value + (1 - tau) * mSmoothedValue;
    }
    mValueCount++;
  }

  public void printTo(StringBuilder sb) {
    if (sb.length() > 0 && sb.charAt(sb.length() - 1) > ' ')
      sb.append("  ");

    String nm = DataUtil.capitalizeFirst(mName);
    if (!isFloat()) {
      sb.append(String.format("%s: %d", nm, intValue()));
    } else {
      sb.append(String.format("%s: %6.3f", nm, mSmoothedValue));
    }
  }

  public float smoothedValue() {
    return mSmoothedValue;
  }

  public int intValue() {
    return Math.round(mValue);
  }

  private boolean isFloat() {
    return !mName.equals(EPOCH);
  }

  private int order() {
    if (mOrder == null) {
      int ord = 1000;
      for (int i = 0; i < LogProcessor.sStatOrder.length; i++) {
        if (LogProcessor.sStatOrder[i].equals(mName))
          ord = i;
      }
      mOrder = ord;
    }
    return mOrder;
  }

  private String mName;
  private float mValue;
  private float mSmoothedValue;
  private int mValueCount;
  private Integer mOrder;

  public static final Comparator<StatRecord> COMPARATOR = (StatRecord x, StatRecord y) -> {
    int diff = x.order() - y.order();
    if (diff == 0)
      diff = x.mName.compareTo(y.mName);
    return diff;
  };

}