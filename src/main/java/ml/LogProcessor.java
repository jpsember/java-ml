package ml;

import gen.CompileImagesConfig;
import gen.FloatFormat;
import gen.NeuralNetwork;
import gen.TensorStats;
import gen.LogItem;
import gen.Vol;
import js.base.BaseObject;
import js.base.BasePrinter;
import js.base.DateTimeTools;
import js.data.DataUtil;
import js.file.DirWalk;
import js.file.Files;
import js.geometry.IPoint;
import js.graphics.ImgUtil;
import js.graphics.ScriptUtil;
import js.graphics.gen.Script;
import js.json.JSMap;
import ml.img.ImageCompiler;

import static js.base.Tools.*;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

public class LogProcessor extends BaseObject implements Runnable {

  public static final boolean ISSUE_61 = true;

  public void start(CompileImagesConfig compileImagesConfig, ModelWrapper model) {
    checkState(mState == STATE_READY);
    mConfig = compileImagesConfig;
    mModel = model;
    mNetwork = model.network();
    mImageVolume = NetworkUtil.determineInputImageVolume(mNetwork);
    mImageSize = VolumeUtil.spatialDimension(mImageVolume);
    mState = STATE_RUNNING;

    mThread = new Thread(this);
    mThread.setDaemon(true);
    mThread.start();
  }

  private static final int STATE_READY = 0, STATE_RUNNING = 1, STATE_STOPPED = 2;

  public void stop() {
    switch (mState) {
    case STATE_RUNNING:
      break;
    }
    mState = STATE_STOPPED;
  }

  public boolean errorFlag() {
    return mErrorFlag;
  }

  @Override
  public void run() {
    log("starting");
    pf().write("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
    prog(DASHES, CR, "Starting training session at:", DateTimeTools.humanTimeString());
    try {
      auxRun();
    } catch (Throwable t) {
      prog("LogProcessor caught exception:", t);
      mErrorFlag = true;
      stop();
    }
  }

  /**
   * Append to progress file
   */
  private void prog(Object... messages) {
    String result = BasePrinter.toString(messages);
    pf().write(result);
  }

  private ProgressFile pf() {
    if (mProgressFile == null)
      mProgressFile = new ProgressFile(config());
    return mProgressFile;
  }

  private void auxRun() {
    while (mState != STATE_STOPPED) {
      File logDir = config().targetDirTrain();
      DirWalk w = new DirWalk(logDir).withRecurse(false).withExtensions("json");
      for (File infoFile : w.files()) {
        if (ISSUE_61) {
          String ext = Files.getExtension(infoFile);
          if (!ext.equals("json")) {
            die("DirWalk is returning a file without a json ext:", Files.infoMap(infoFile));
          }
        }
        LogItem ti = null;

        if (ISSUE_61) {
          try {
            ti = parseLogItem(infoFile);
          } catch (Throwable t) {
            prog("*** failed to parseLogItem, file:", INDENT, Files.infoMap(infoFile));
            prog("logDir:", INDENT, Files.infoMap(logDir));
            prog("exception:", t);
            continue;
          }
        } else {
          ti = parseLogItem(infoFile);
        }

        if (ti.illegalValuesFound()) {
          String formatted = prettyPrint(ti);
          if (formatted.length() > 500)
            formatted = formatted.substring(0, 500) + "...";
          prog("======== Tensor name:", ti.message());
          prog("Illegal values found !");
          prog(formatted);
          prog("======== Tensor name:", ti.message());
        }
        int tensorTypeCount = 0;
        if (ti.tensorBytes() != null)
          tensorTypeCount++;
        if (ti.tensorFloats() != null)
          tensorTypeCount++;
        checkArgument(tensorTypeCount <= 1, "multiple tensor types stored in message", brief(ti));

        if (ti.id() <= mPrevId) {
          prog("*** log item not greater than prev:", mPrevId, INDENT, ti);
        } else {
          checkState(ti.id() > mPrevId, "LogItem ids not strictly increasing");
          mPrevId = ti.id();
          int effFamSize = Math.max(1, ti.familySize());
          bufferLogItem(ti, effFamSize);
        }
        files().deletePeacefully(infoFile);
      }
      cullFamilyBuffer();
      pf().flush();
      DateTimeTools.sleepForRealMs(50);
    }
    log("stopping");
  }

  /**
   * Parse LogItem from file, replacing NaN with -999 so it still parses
   */
  private LogItem parseLogItem(File file) {
    String content = Files.readString(file);
    boolean hasNan = content.contains("NaN");
    LogItem result;
    if (hasNan) {
      content = content.replace("NaN", "-999");
    }
    result = Files.parseAbstractData(LogItem.DEFAULT_INSTANCE, new JSMap(content));
    if (hasNan)
      result = result.toBuilder().illegalValuesFound(true).build();
    return result;
  }

  /**
   * Pretty print a LogItem, changing any "-999" (that presumably represent NaNs
   * found earlier) with "NaN"
   */
  private String prettyPrint(LogItem ti) {
    return ti.toString().replace("-999.0", "NaN").replace("-999", "NaN");
  }

  private void processLogItem(LogItem ti) {
    StringBuilder sb = new StringBuilder();
    if (ti.shape().length != 0) {
      formatTensor(ti, sb);
    } else {
      String msg = ti.message().trim();
      sb.append(msg);
      addLF(sb);
      if (ti.stats() != null)
        parseStats(ti.stats(), sb);
    }
    prog(sb.toString());
  }

  private void bufferLogItem(LogItem logItem, int familySize) {
    log("processing labelled image:", logItem);
    int key = logItem.familyId();
    LogItem[] familySet = mFamilyMap.get(key);
    if (familySet == null) {
      familySet = new LogItem[familySize];
      mFamilyMap.put(key, familySet);
    }

    checkState(familySet[logItem.familySlot()] == null, "family slot already taken:", INDENT, logItem);
    familySet[logItem.familySlot()] = logItem;

    boolean famComplete = true;
    for (LogItem famElement : familySet)
      if (famElement == null)
        famComplete = false;

    if (!famComplete)
      return;

    mFamilyMap.remove(logItem.familyId());

    switch (logItem.specialHandling()) {
    default:
      throw notSupported("special handling for:", INDENT, logItem);
    case NONE:
      for (LogItem itm : familySet)
        processLogItem(itm);
      break;
    case SNAPSHOT:
      processSnapshotItem(familySet);
      break;
    }
  }

  private void processSnapshotItem(LogItem[] family) {
    if (Files.empty(config().snapshotDir()))
      return;
    LogItem imgRec = family[0];
    LogItem lblRec = family[1];

    Vol imgVol = NetworkUtil.determineInputImageVolume(mNetwork);

    switch (mNetwork.imageDataType()) {
    default:
      throw notSupported("network.image_data_type", mNetwork.imageDataType());
    case UNSIGNED_BYTE: {
      checkNotNull(imgRec.tensorBytes(), "no bytes in tensor");
      int imgLength = imgRec.tensorBytes().length;

      // We have a stacked batch of images.
      int bytesPerImage = mImageSize.product() * mImageVolume.depth();

      int batchSize = imgLength / bytesPerImage;
      checkArgument(imgLength % bytesPerImage == 0, "images length", imgLength,
          "is not a multiple of image volume", bytesPerImage);
      String setName = String.format("%05d_", imgRec.familyId()) + "_%02d";

      for (int i = 0; i < batchSize; i++) {
        byte[] imgb = Arrays.copyOfRange(imgRec.tensorBytes(), bytesPerImage * i, bytesPerImage * (i + 1));
        if (ModelWrapper.ISSUE_42_PIXEL_ORDER) {
          imgb = ImageCompiler.pixelCYXtoYXC(mImageSize, imgb);
        }

        BufferedImage img = ImgUtil.bytesToBGRImage(imgb, VolumeUtil.spatialDimension(imgVol));
        File baseFile = new File(targetProjectDir(), String.format(setName, i));
        File imgPath = Files.setExtension(baseFile, ImgUtil.EXT_JPEG);
        ImgUtil.writeJPG(files(), img, imgPath, null);

        switch (mNetwork.labelDataType()) {
        case FLOAT32: {
          float[] targetBuffer = mModel.labelBufferFloats();
          float[] labelSets = checkNotNull(lblRec.tensorFloats(), "tensor floats");
          int imgLblLen = targetBuffer.length;
          checkArgument(batchSize * imgLblLen == labelSets.length, "label size * batch != labels length");
          System.arraycopy(labelSets, imgLblLen * i, targetBuffer, 0, imgLblLen);
        }
          break;
        default:
          throw notSupported("label data type:", mNetwork.labelDataType());
        }

        Script.Builder script = Script.newBuilder();
        script.items(mModel.transformModelOutputToScredit());
        ScriptUtil.write(files(), script, ScriptUtil.scriptPathForImage(imgPath));
      }
    }
      break;
    }
  }

  private JSMap brief(LogItem x) {
    JSMap m = x.toJson();
    if (x.tensorBytes() != null)
      m.put("tensor_bytes", x.tensorBytes().length);
    if (x.tensorFloats() != null)
      m.put("tensor_floats", x.tensorFloats().length);
    return m;
  }

  private File targetProjectDir() {
    if (mTargetProjectDir == null) {
      mTargetProjectDir = config().snapshotDir();
      checkArgument(Files.basename(mTargetProjectDir).contains("snapshot"),
          "for safety, must contain word 'snapshot'");
      files().remakeDirs(mTargetProjectDir);
      mTargetProjectScriptsDir = ScriptUtil.scriptDirForProject(mTargetProjectDir);
      files().mkdirs(mTargetProjectScriptsDir);
    }
    return mTargetProjectDir;
  }

  /**
   * If for some reason there are family sets still waiting after a long while,
   * delete them
   */
  private void cullFamilyBuffer() {
    List<Integer> keysToRemove = arrayList();
    for (LogItem[] set : mFamilyMap.values()) {
      for (LogItem r : set) {
        if (r != null && r.id() < mPrevId - 20) {
          keysToRemove.add(r.familyId());
          break;
        }
      }
    }
    if (keysToRemove.isEmpty())
      return;
    alert("Removing stale buffered records of size:", keysToRemove.size());
    mFamilyMap.keySet().removeAll(keysToRemove);
  }

  private Files files() {
    return Files.S;
  }

  private CompileImagesConfig config() {
    return mConfig;
  }

  private void formatTensor(LogItem ti, StringBuilder sb) {
    float[] coeff = ti.tensorFloats();
    String effName = ti.message();
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
      fmt = getFloatFormatString(coeff);
    }

    if (nonEmpty(ti.message())) {
      sb.append(ti.message());
      sb.append('\n');
    }
    int[] shape = ti.shape();

    // View the tensor as two dimensional, by collapsing dimensions 2...n together into one.
    // More elaborate manipulations, cropping, etc., should be done within the Python code
    //
    todo("have support for optionally flattening dimensions");
    boolean special = false;
    if (shape.length <= 1 || shape[0] == 0) {
      int[] altShape = new int[2];
      altShape[0] = 1;
      altShape[1] = coeff.length;
      shape = altShape;
    } else if (shape.length >= 3 && shape[0] == 32) {
      special = true;
    }

    int pages = 1;
    int rows, cols;
    if (special) {
      pages = shape[0];
      rows = shape[1];
      cols = shape[2];
      for (int i = 3; i < shape.length; i++) {
        cols *= shape[i];
      }
    } else {
      rows = shape[0];
      cols = shape[1];
      for (int i = 2; i < shape.length; i++) {
        cols *= shape[i];
      }
    }
    checkArgument(pages * rows * cols == coeff.length);
    int q = 0;
    for (int p = 0; p < pages; p++) {
      for (int y = 0; y < rows; y++) {
        sb.append("[ ");
        for (int x = 0; x < cols; x++, q++) {
          if (x > 0)
            sb.append(" │ "); // Note: this is a unicode char taller than the vertical brace
          sb.append(fmt(fmt, coeff[q]));
        }
        sb.append(" ]\n");
      }
    }

    if (pages > 1)
      sb.append("\n");
  }

  /* private */ TensorStats buildTensorStats(float[] tensorFloats) {
    if (tensorFloats.length == 0)
      return TensorStats.DEFAULT_INSTANCE;
    TensorStats.Builder b = TensorStats.newBuilder();

    double sum = 0;
    b.min(tensorFloats[0]).max(tensorFloats[0]);
    for (float f : tensorFloats) {
      sum += f;
      if (f < b.min())
        b.min(f);
      if (f > b.max())
        b.max(f);
    }
    b.population(tensorFloats.length);
    b.mean((float) (sum / b.population()));
    return b.build();
  }

  private static final FloatFormat buildFmt(float maxVal, String fmt, float minVal, String zero) {
    FloatFormat.Builder b = FloatFormat.newBuilder();
    b.maxValue(maxVal).formatStr(fmt).minValue(minVal).zeroStr(zero);
    return b.build();
  }

  private static String blankField(int width) {
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

  // ------------------------------------------------------------------
  // Statistics bookkeeping
  // ------------------------------------------------------------------

  private static final String STAT_EPOCH = "epoch";
  private static final String STAT_LOSS = "loss";

  private static String[] sStatOrder = { STAT_EPOCH, STAT_LOSS };

  private static class StatRecord {

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
        sb.append(String.format("%s: %d", nm, Math.round(mValue)));
      } else {
        sb.append(String.format("%s: %5.2f", nm, mSmoothedValue));
      }
    }

    private boolean isFloat() {
      return !mName.equals(STAT_EPOCH);
    }

    private int order() {
      if (mOrder == null) {
        int ord = 1000;
        for (int i = 0; i < sStatOrder.length; i++) {
          if (sStatOrder[i].equals(mName))
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

  private Map<String, StatRecord> mStatRecordMap = hashMap();

  private StatRecord statRecord(String name) {
    StatRecord r = mStatRecordMap.get(name);
    if (r == null) {
      r = new StatRecord(name);
      mStatRecordMap.put(name, r);
    }
    return r;
  }

  private void parseStats(JSMap stats, StringBuilder sb) {
    List<StatRecord> rec = arrayList();
    for (String key : stats.keySet()) {
      float value = stats.getFloat(key);
      StatRecord r = statRecord(key);
      r.update(value);
      rec.add(r);
    }
    rec.sort(StatRecord.COMPARATOR);
    for (StatRecord r : rec) {
      r.printTo(sb);
    }
  }

  // ------------------------------------------------------------------

  private Vol mImageVolume;
  private IPoint mImageSize;

  private CompileImagesConfig mConfig;
  private ModelWrapper mModel;
  private NeuralNetwork mNetwork;
  private int mState = STATE_READY;
  private Thread mThread;
  private int mPrevId;

  private Map<Integer, LogItem[]> mFamilyMap = hashMap();
  private File mTargetProjectDir;
  private File mTargetProjectScriptsDir;
  private boolean mErrorFlag;
  private ProgressFile mProgressFile;
}
