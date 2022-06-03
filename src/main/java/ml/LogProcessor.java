package ml;

import gen.CompileImagesConfig;
import gen.FloatFormat;
import gen.TensorInfo;
import js.base.BaseObject;
import js.base.DateTimeTools;
import js.file.DirWalk;
import js.file.Files;

import static js.base.Tools.*;

import java.io.File;

public class LogProcessor extends BaseObject implements Runnable {

  public void start(CompileImagesConfig c) {
    checkState(mState == 0);
    mConfig = c;
    mState = 1;
    mThread = new Thread(this);
    mThread.setDaemon(true);
    mThread.start();
  }

  public void stop() {
    switch (mState) {
    case 1:
      break;
    }
    mState = 2;
  }

  @Override
  public void run() {
    log("starting");
    while (mState != 2) {
      File logDir = config().targetDirTrain();
      DirWalk w = new DirWalk(logDir).withRecurse(false).withExtensions("json");
      for (File infoFile : w.files())
        processFile(infoFile);
      DateTimeTools.sleepForRealMs(50);
    }
    log("stopping");
  }

  private void processFile(File infoFile) {
    File tensorFile = Files.setExtension(infoFile, "dat");
    // If no extension file exists, it may not have been renamed
    if (!tensorFile.exists())
      DateTimeTools.sleepForRealMs(100);
    if (!tensorFile.exists()) {
      pr("...logger, no corresponding tensor file found:", tensorFile.getName());
    } else {
      TensorInfo ti = Files.parseAbstractData(TensorInfo.DEFAULT_INSTANCE, infoFile);
      switch (ti.dataType()) {
      case FLOAT32: {
        float[] t = Files.readFloatsLittleEndian(tensorFile, "tensorFile");
        String s = formatTensor(ti, t);
        pr(s);
      }
        break;
      default:
        throw notSupported("Unsupported datatype:", ti);
      }
    }
    files().deletePeacefully(infoFile);
    files().deletePeacefully(tensorFile);
  }

  private Files files() {
    return Files.S;
  }

  private CompileImagesConfig config() {
    return mConfig;
  }

  private String formatTensor(TensorInfo ti, float[] t) {

    String effName = ti.name();
    String suffix = "";
    {
      int i = effName.indexOf(':');
      if (i >= 0) {
        suffix = effName.substring(i + 1);
        effName = effName.substring(0, i);
      }
    }

    FloatFormat fmt = null;
    if (nonEmpty(suffix)) {
      fmt = FLOAT_FORMATS[Integer.parseInt(suffix)];
    } else {
      fmt = getFloatFormatString(t);
    }

    StringBuilder sb = new StringBuilder();
    sb.append(ti.name());
    sb.append('\n');
    int[] shape = ti.shape();

    // View the tensor as two dimensional, by collapsing dimensions 2...n together into one.
    // More elaborate manipulations, cropping, etc., should be done within the Python code
    //
    if (shape.length <= 1 || shape[0] == 0) {
      int[] altShape = new int[2];
      altShape[0] = 1;
      altShape[1] = t.length;
      shape = altShape;
    }

    int rows = shape[0];
    int cols = shape[1];
    for (int i = 2; i < shape.length; i++) {
      cols *= shape[i];
    }
    checkArgument(rows * cols == t.length);
    int q = 0;
    for (int y = 0; y < rows; y++) {
      sb.append("[ ");
      for (int x = 0; x < cols; x++, q++) {
        if (x > 0)
          sb.append(" │ "); // Note: this is a unicode char taller than the vertical brace
        sb.append(fmt(fmt, t[q]));
      }
      sb.append(" ]\n");
    }
    return sb.toString();
  }

  private static final FloatFormat buildFmt(float maxVal, String fmt, float minVal, String zero) {
    FloatFormat.Builder b = FloatFormat.newBuilder();
    b.maxValue(maxVal).formatStr(fmt).minValue(minVal).zeroStr(zero);
    return b.build();
  }

  private static String blankField(int width) {
    if (false)
      return spaces(width - 1) + "◌";
    return spaces(width);
  }

  private static final FloatFormat[] FLOAT_FORMATS = { //
      buildFmt(0.1f, "%7.4f", 0.0001f, blankField(7)), //
      buildFmt(1, "%6.3f", .001f, blankField(6)), //
      buildFmt(10, "%5.2f", 0.01f, blankField(5)), //
      buildFmt(100, "%3.0f", 1f, blankField(3)), //
      buildFmt(1000, "%4.0f", 1f, blankField(4)), //
      buildFmt(Float.MAX_VALUE, "%7.0f", 1f, blankField(7)), //
  };

  private static FloatFormat getFloatFormatString(float[] floats) {
    checkArgument(floats.length > 0);
    float magMax = Math.abs(floats[0]);
    for (float f : floats) {
      float mag = Math.abs(f);
      if (mag > magMax)
        magMax = mag;
    }
    for (FloatFormat fmt : FLOAT_FORMATS) {
      if (magMax < fmt.maxValue())
        return fmt;
    }
    return FLOAT_FORMATS[FLOAT_FORMATS.length - 1];
  }

  private static String fmt(FloatFormat format, float value) {
    if (Math.abs(value) < format.minValue())
      return format.zeroStr();
    return String.format(format.formatStr(), value);
  }

  private CompileImagesConfig mConfig;
  private int mState;

  private Thread mThread;
}
