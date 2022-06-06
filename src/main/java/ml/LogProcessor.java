package ml;

import gen.CompileImagesConfig;
import gen.FloatFormat;
import gen.NeuralNetwork;
import gen.LogItem;
import gen.Vol;
import js.base.BaseObject;
import js.base.DateTimeTools;
import js.file.DirWalk;
import js.file.Files;
import js.geometry.IPoint;
import js.graphics.ImgUtil;
import js.graphics.ScriptUtil;
import js.graphics.gen.Script;

import static js.base.Tools.*;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Arrays;
import java.util.Map;

public class LogProcessor extends BaseObject implements Runnable {

  public void start(CompileImagesConfig c, ModelWrapper model) {
    checkState(mState == 0);
    mConfig = c;
    mModel = model;
    mNetwork = model.network();
    determineImageAndLabelInfo();
    mState = 1;
    mThread = new Thread(this);
    mThread.setDaemon(true);
    mThread.start();
  }

  private Vol mImageVolume;
  private IPoint mImageSize;

  private void determineImageAndLabelInfo() {
    mImageVolume = NetworkUtil.determineInputImageVolume(mNetwork);
    mImageSize = VolumeUtil.spatialDimension(mImageVolume);
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
      for (File infoFile : w.files()) {
        File tensorFile = Files.setExtension(infoFile, "dat");
        processFile(infoFile, tensorFile);
        files().deletePeacefully(infoFile);
        files().deletePeacefully(tensorFile);
      }

      DateTimeTools.sleepForRealMs(50);
    }
    log("stopping");
  }

  private void processFile(File infoFile, File tensorFile) {
    LogItem ti = Files.parseAbstractData(LogItem.DEFAULT_INSTANCE, infoFile);
    if (ti.id() <= mPrevId) {
      pr("*** log item not greater than prev:", mPrevId, INDENT, ti);
      return;
    }
    checkState(ti.id() > mPrevId, "LogItem ids not strictly increasing");
    mPrevId = ti.id();

    if (ti.infrequent()) {
      if (config().logEpochInterval() > 0 && ti.epoch() % config().logEpochInterval() != 0) 
        return;
    }

    InfoRecord rec = new InfoRecord(ti);

    if (rec.hasTensor()) {
      // If no extension file exists, it may not have been renamed
      if (!tensorFile.exists())
        DateTimeTools.sleepForRealMs(100);

      todo("InfoRecords won't always have corresponding tensors");
      if (!tensorFile.exists()) {
        pr("...logger, no corresponding tensor file found:", tensorFile.getName());
        return;
      }

      switch (ti.dataType()) {
      case FLOAT32: {
        float[] t = Files.readFloatsLittleEndian(tensorFile, "tensorFile");
        rec.setData(t);
      }
        break;
      case UNSIGNED_BYTE:
        rec.setData(Files.toByteArray(tensorFile, "tensorFile"));
        break;
      default:
        throw notSupported(ti);
      }
    }

    // If this belongs to a family, buffer it accordingly
    //
    if (ti.familySize() != 0) {
      todo("separate the buffering from the handling (printing, generating script)");
      processLabelledImage(rec);
      return;
    }

    StringBuilder sb = new StringBuilder();
    if (rec.hasTensor()) {
      checkArgument(rec.mFloats != null, "no floats found");
      String s = formatTensor(ti, rec.mFloats);
      sb.append(s.trim());
    } else {

      // Parse special sequences of the form ^x;
      //
      String msg = ti.message().trim();
      int cursor = 0;
      while (cursor < msg.length()) {
        if (msg.charAt(cursor) == '^') {
          checkArgument(cursor + 3 <= msg.length() && msg.charAt(cursor + 2) == ';', "ill formed message:",
              msg, CR, "cursor:", cursor, quote(msg.substring(cursor)));
          char code = msg.charAt(cursor + 1);
          switch (code) {
          case 'v':
            sb.append("\n\n\n");
            break;
          case 'd':
            sb.append("-----------------------------------------------------------------------");
            break;
          default:
            throw badArg("ill formed message:", msg);
          }
          cursor += 3;
        } else {
          int nextC = msg.indexOf('^', cursor);
          if (nextC < 0)
            nextC = msg.length();
          sb.append(msg.substring(cursor, nextC));
          cursor = nextC;
        }
      }
    }
    addLF(sb);
    System.out.print(sb.toString());
  }

  private void processLabelledImage(InfoRecord rec) {
    LogItem logItem = rec.logItem();
    if (todo("do this check only when the family is being processed") && //
        Files.empty(config().snapshotDir()))
      return;

    log("processing labelled image:", logItem);
    int key = logItem.familyId();
    InfoRecord[] familySet = mFamilyMap.get(key);
    if (familySet == null) {
      familySet = new InfoRecord[logItem.familySize()];
      mFamilyMap.put(key, familySet);
    }

    checkState(familySet[logItem.familySlot()] == null, "family slot already taken:", INDENT, logItem);
    familySet[logItem.familySlot()] = rec;

    boolean famComplete = true;
    for (InfoRecord famElement : familySet)
      if (famElement == null)
        famComplete = false;

    if (!famComplete) {
      todo("do culling in a more general location");
      return;
    }

    mFamilyMap.remove(logItem.familyId());

    InfoRecord imgRec = familySet[0];
    InfoRecord lblRec = familySet[1];

    Vol imgVol = NetworkUtil.determineInputImageVolume(mNetwork);

    switch (mNetwork.imageDataType()) {
    default:
      throw notSupported("network.image_data_type", mNetwork.imageDataType());
    case UNSIGNED_BYTE: {
      if (imgRec.mBytes == null)
        badArg("missing image bytes");
      int imgLength = imgRec.mBytes.length;

      // We have a stacked batch of images.
      int bytesPerImage = mImageSize.product() * mImageVolume.depth();

      int batchSize = imgLength / bytesPerImage;
      checkArgument(imgLength % bytesPerImage == 0, "images length", imgLength,
          "is not a multiple of image volume", bytesPerImage);
      String setName = "" + logItem.familyId() + "_%02d";
      for (int i = 0; i < batchSize; i++) {
        byte[] imgb = Arrays.copyOfRange(imgRec.mBytes, bytesPerImage * i, bytesPerImage * (i + 1));
        BufferedImage img = ImgUtil.bytesToBGRImage(imgb, VolumeUtil.spatialDimension(imgVol));
        File baseFile = new File(targetProjectDir(), String.format(setName, i));
        File imgPath = Files.setExtension(baseFile, ImgUtil.EXT_JPEG);
        ImgUtil.writeJPG(files(), img, imgPath, null);

        switch (mNetwork.labelDataType()) {
        case FLOAT32: {
          float[] targetBuffer = mModel.labelBufferFloats();
          float[] labelSets = lblRec.mFloats;
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

  //  /**
  //   * If for some reason there are orphaned images or labels, discard them
  //   */
  //  private void cullStalePendingRecords(int key) {
  //    List<Integer> keysToRemove = arrayList();
  //    for (int k : mOrphanInfoRecordMap.keySet()) {
  //      if (k < key - 10) {
  //        keysToRemove.add(k);
  //      }
  //    }
  //    if (keysToRemove.isEmpty())
  //      return;
  //    alert("Removing stale pending records of size:", keysToRemove.size());
  //    mOrphanInfoRecordMap.keySet().removeAll(keysToRemove);
  //  }

  private Files files() {
    return Files.S;
  }

  private CompileImagesConfig config() {
    return mConfig;
  }

  private String formatTensor(LogItem ti, float[] t) {
    todo("refactor to accept InfoRecord instead of ti");

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
      fmt = getFloatFormatString(t);
    }

    StringBuilder sb = new StringBuilder();
    if (nonEmpty(ti.message())) {
      sb.append(ti.message());
      sb.append('\n');
    }
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
      sb.append(" ]");
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

  private static class InfoRecord {

    public InfoRecord(LogItem tensorInfo) {
      mLogItem = tensorInfo.build();
    }

    public boolean hasTensor() {
      return logItem().shape().length != 0;
    }

    public void setData(byte[] bytes) {
      mBytes = bytes;
    }

    public void setData(float[] floats) {
      mFloats = floats;
    }

    public LogItem logItem() {
      return mLogItem;
    }

    private final LogItem mLogItem;
    byte[] mBytes;
    float[] mFloats;
  }

  private CompileImagesConfig mConfig;
  private ModelWrapper mModel;
  private NeuralNetwork mNetwork;
  private int mState;
  private Thread mThread;
  private int mPrevId;

  private Map<Integer, InfoRecord[]> mFamilyMap = hashMap();
  private File mTargetProjectDir;
  private File mTargetProjectScriptsDir;

}
