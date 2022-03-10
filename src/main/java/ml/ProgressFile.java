package ml;

import js.file.Files;
import js.json.JSMap;
import ml.Util;
import ml.ModelHandler;
import gen.TrainStream;

import static js.base.Tools.*;

import java.io.BufferedWriter;
import java.io.Closeable;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.List;
import java.util.Map;

public final class ProgressFile implements Closeable {

  public ProgressFile(ModelHandler model, TrainStream config) {
    mModel = model;
    mConfig = config;
    Files.S.deletePeacefully(exampleFile());
  }

  private void write(String content) {
    writer().println(content);
  }

  private void flush() {
    writer().flush();
  }

  @Override
  public void close() {
    if (mProgressFile == null)
      return;
    try {
      Files.close(writer());
    } finally {
      mProgressFile = null;
      mPrintWriter = null;
    }
  }

  public void processUpdate(JSMap m) {
    int epoch = m.getInt("epoch");
    int batchTotal = m.opt("batch_total", 1);
    int batch = m.opt("batch", 0);

    float trainLoss = m.getFloat("train_loss");
    boolean hasTestLoss = m.containsKey("test_loss");
    float testLoss = 0;
    if (hasTestLoss)
      testLoss = m.getFloat("test_loss");
    Float learningRate = m.opt("learning_rate", 0f);
    mSmoothedTrainLoss.update(trainLoss);
    if (hasTestLoss)
      mSmoothedTestLoss.update(testLoss);

    sb().setLength(0);

    append("Epoch ");

    String epochExpr = "" + epoch;
    if (batchTotal > 1) {
      String batchFormatExpr = Util.formatStringForTotal(batchTotal);
      epochExpr += "." + String.format(batchFormatExpr, batch);
    }
    append(epochExpr);

    append("  Train loss: ");
    append(formatSized(trainLoss));
    append(mSmoothedTrainLoss);
    append("  Test loss: ");
    append(formatSized(mSmoothedTestLoss.recentValue()));
    append(mSmoothedTestLoss);

    appendIfChanged("  Learning rate: ", String.format("%7.5f", learningRate));

    mModel.updateTrainingProgress(this, m);

    int origLength = sb().length();
    if (origLength != sb().length()) {
      if (alert("Generating sample that includes inspections"))
        writeExample(m);
    }

    String message = sb().toString().trim();

    write(message);
    pr(message);
    flush();
  }

  private void append(SmoothedFloat smoothed) {
    append(" (");
    append(formatSized(smoothed.value()));
    append(")");
  }

  private static String formatSized(float value) {
    if (value > Integer.MAX_VALUE)
      return " (^^^)";
    if (value < 0)
      return String.format("%7.5f", value);
    float v = value;
    if (v >= 100)
      return String.format("%3d", Math.round(v));
    else if (v >= 3)
      return String.format("%4.1f", v);
    else if (v >= 0.5f)
      return String.format("%4.2f", v);
    else if (v >= 0.1f)
      return String.format("%5.3f", v);
    else if (v >= 0.01f)
      return String.format("%7.5f", v);
    else
      return String.format("%8.6f", value);
  }

  public void displayMessage(JSMap m) {
    String content = m.get("text");
    write(content);
    pr(content);
    flush();
  }

  private void appendIfChanged(String heading, String value) {
    String oldValue = mPrevValuesMap.put(heading, value);
    appendIf(heading, value, !value.equals(oldValue));
  }

  private void appendIf(String heading, String value, boolean flag) {
    if (flag) {
      append(heading);
      append(value);
    } else
      append(spaces(heading.length() + value.length()));
  }

  private void append(Object msg) {
    sb().append(msg);
  }

  private File file() {
    if (mProgressFile == null) {
      File checkpointPath = mConfig.checkpointsDir();
      checkArgument(Files.nonEmpty(checkpointPath));
      mProgressFile = new File(checkpointPath, "progress.txt");
    }
    return mProgressFile;
  }

  private PrintWriter writer() {
    if (mPrintWriter == null) {
      try {
        mPrintWriter = new PrintWriter(new BufferedWriter(new FileWriter(file(), true)));
      } catch (Throwable t) {
        throw Files.asFileException(t);
      }
    }
    return mPrintWriter;
  }

  private StringBuilder sb() {
    return mStringBuilder;
  }

  private void writeExample(JSMap m) {
    if (mExampleGenerated)
      return;
    Files.S.writePretty(exampleFile(), m);
    pr("...wrote example to:", exampleFile());
    mExampleGenerated = true;
  }

  private File exampleFile() {
    return new File(mConfig.trainInputDir(), "example.json");
  }

  private boolean mExampleGenerated;

  private final ModelHandler mModel;
  private final TrainStream mConfig;
  private File mProgressFile;
  private PrintWriter mPrintWriter;
  private StringBuilder mStringBuilder = new StringBuilder();

  private Map<String, String> mPrevValuesMap = hashMap();
  private SmoothedFloat mSmoothedTrainLoss = new SmoothedFloat();
  private SmoothedFloat mSmoothedTestLoss = new SmoothedFloat();

  private static class SmoothedFloat {

    private static final int MAX_SIZE = 40;

    public void update(float loss) {
      mHistory.add(loss);
      while (mHistory.size() > MAX_SIZE)
        mHistory.remove(0);

      float sum = 0;
      for (float value : mHistory) {
        sum += value;
      }
      mSmoothed = sum / mHistory.size();
    }

    public float recentValue() {
      if (mHistory.size() == 0)
        return 0;
      return last(mHistory);
    }

    public float value() {
      return mSmoothed;
    }

    private List<Float> mHistory = arrayList();
    private float mSmoothed;

  }

}
