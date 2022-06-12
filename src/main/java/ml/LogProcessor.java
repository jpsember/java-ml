package ml;

import gen.CompileImagesConfig;
import gen.DataType;
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
import js.json.JSMap;
import ml.img.ImageCompiler;

import static js.base.Tools.*;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Arrays;
import java.util.List;
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

  public boolean errorFlag() {
    return mErrorFlag;
  }

  @Override
  public void run() {
    log("starting");
    try {
      auxRun();
    } catch (Throwable t) {
      pr("LogProcessor caught exception:", t);
      mErrorFlag = true;
      stop();
    }
  }

  private void auxRun() {
    while (mState != 2) {
      File logDir = config().targetDirTrain();
      DirWalk w = new DirWalk(logDir).withRecurse(false).withExtensions("json");
      for (File infoFile : w.files()) {
        LogItem ti;
        try {
          ti = Files.parseAbstractData(LogItem.DEFAULT_INSTANCE, infoFile);
        } catch (Throwable t) {
          throw badArg("Problem parsing file:", infoFile, "message:", t.getMessage());
        }
        int tensorTypeCount = 0;
        if (ti.tensorBytes() != null)
          tensorTypeCount++;
        if (ti.tensorFloats() != null)
          tensorTypeCount++;
        checkArgument(tensorTypeCount <= 1, "multiple tensor types stored in message", brief(ti));

        if (ti.id() <= mPrevId) {
          pr("*** log item not greater than prev:", mPrevId, INDENT, ti);
        } else {
          checkState(ti.id() > mPrevId, "LogItem ids not strictly increasing");
          mPrevId = ti.id();
          int effFamSize = Math.max(1, ti.familySize());
          bufferLogItem(ti, effFamSize);
        }
        files().deletePeacefully(infoFile);
      }
      cullFamilyBuffer();
      DateTimeTools.sleepForRealMs(50);
    }
    log("stopping");
  }

  private void processLogItem(LogItem ti) {
    StringBuilder sb = new StringBuilder();
    if (ti.shape().length != 0) {
      String s = formatTensor(ti);
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
    case 0:
      for (LogItem itm : familySet)
        processLogItem(itm);
      break;
    case 1:
      processSnapshotItem(familySet);
      break;
    case 2:
      processIssue42(familySet);
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

      final boolean show = false && alert("showing snapshot labels");
      JSMap m = null;
      if (show) {
        m = map();
      }
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

          if (show) {
            StringBuilder sb = new StringBuilder();
            for (float f : targetBuffer)
              sb.append(String.format("%4d ", (int) (f * 100)));
            m.put(String.format("img%02d", i), sb.toString());
          }
        }
          break;
        default:
          throw notSupported("label data type:", mNetwork.labelDataType());
        }

        Script.Builder script = Script.newBuilder();
        if (config().logLabels())
          pr("Model produced labels:", CR, mModel.renderLabels());
        script.items(mModel.transformModelOutputToScredit());
        ScriptUtil.write(files(), script, ScriptUtil.scriptPathForImage(imgPath));
      }
      if (show)
        pr("labels:", INDENT, m);
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

  private void show(String message, LogItem x) {
    pr(message, INDENT, brief(x));
  }

  private void processIssue42(LogItem[] family) {
    if (Files.empty(config().snapshotDir()))
      return;
    LogItem trainImagesRec = family[0];
    LogItem trainLabelsRec = family[1];
    show("trainImagesRec:", trainImagesRec);
    show("trainLabelsRec:", trainLabelsRec);

    LogItem imgLossRec = family[2];
    LogItem lblLossRec = family[3];
    if (false && imgLossRec == null && lblLossRec == null)
      pr("");

    Vol imgVol = NetworkUtil.determineInputImageVolume(mNetwork);
    checkArgument(mNetwork.imageDataType() == DataType.UNSIGNED_BYTE);
    checkArgument(mNetwork.labelDataType() == DataType.FLOAT32);
    int imgLength = trainImagesRec.tensorBytes().length;

    // We have a stacked batch of images.
    int bytesPerImage = mImageSize.product() * mImageVolume.depth();

    int batchSize = imgLength / bytesPerImage;
    checkArgument(imgLength % bytesPerImage == 0, "images length", imgLength,
        "is not a multiple of image volume", bytesPerImage);
    String setName = "" + trainImagesRec.familyId() + "_%02d";

    byte[] imgb = new byte[bytesPerImage];
    checkArgument(mImageVolume.depth() == 3, "not supported for channels != 3");

    for (int i = 0; i < batchSize; i++) {
      // The model wants the shape to be (image, channel, column, row)
      // which is different from the BufferedImage (row, column, channel),
      // so reverse this interleaving
      //
      // ....but what about the ordering of the rows and columns?
      //
      {
        byte[] src = trainImagesRec.tensorBytes();
        int j = bytesPerImage * i;
        int bytesPerChannel = bytesPerImage / 3;
        int q = 0;
        for (int k = 0; k < bytesPerImage; k += 3, q++) {
          imgb[k] = src[j + q];
          imgb[k + 1] = src[j + bytesPerChannel + q];
          imgb[k + 2] = src[j + bytesPerChannel * 2 + q];
        }
      }
      BufferedImage img = ImgUtil.bytesToBGRImage(imgb, VolumeUtil.spatialDimension(imgVol));
      File baseFile = new File(targetProjectDir(), String.format(setName, i));
      File imgPath = Files.setExtension(baseFile, ImgUtil.EXT_JPEG);
      ImgUtil.writeJPG(files(), img, imgPath, null);

      todo("do something with the predicted labels instead?");
      {
        float[] targetBuffer = mModel.labelBufferFloats();
        float[] labelSets = trainLabelsRec.tensorFloats();
        int imgLblLen = targetBuffer.length;
        checkArgument(batchSize * imgLblLen == labelSets.length, "label size * batch != labels length");
        System.arraycopy(labelSets, imgLblLen * i, targetBuffer, 0, imgLblLen);
      }

      Script.Builder script = Script.newBuilder();
      script.items(mModel.transformModelInputToScredit());
      ScriptUtil.write(files(), script, ScriptUtil.scriptPathForImage(imgPath));
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

  private String formatTensor(LogItem ti) {

    float[] coeff = ti.tensorFloats();
    todo("add support for byte tensors");
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

    StringBuilder sb = new StringBuilder();
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
      if (pages > 1)
        sb.append("\n");
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
  private ModelWrapper mModel;
  private NeuralNetwork mNetwork;
  private int mState;
  private Thread mThread;
  private int mPrevId;

  private Map<Integer, LogItem[]> mFamilyMap = hashMap();
  private File mTargetProjectDir;
  private File mTargetProjectScriptsDir;
  private boolean mErrorFlag;
}
