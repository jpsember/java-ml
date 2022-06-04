package ml;

import gen.CompileImagesConfig;
import gen.FloatFormat;
import gen.NeuralNetwork;
import gen.TensorInfo;
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
      InfoRecord rec = new InfoRecord(ti);
      switch (ti.dataType()) {
      case FLOAT32: {
        float[] t = Files.readFloatsLittleEndian(tensorFile, "tensorFile");
        rec.setData(t);
        if (ti.imageIndex() > 0) {
          todo("do something with indexed image");
          pr(ti);
          break;
        }
        if (ti.labelIndex() > 0) {
          todo("do something with indexed image/label");
          pr(ti);
          break;
        }
      }
        break;
      case UNSIGNED_BYTE:
        rec.setData(Files.toByteArray(tensorFile, "tensorFile"));
        break;
      default:
        throw notSupported(ti);
      }

      if (ti.imageIndex() > 0 || ti.labelIndex() > 0) {
        processLabelledImage(rec);
      } else {
        checkArgument(rec.mFloats != null, "no floats found");
        String s = formatTensor(ti, rec.mFloats);
        pr(s);
      }

    }
    files().deletePeacefully(infoFile);
    files().deletePeacefully(tensorFile);
  }

  private void processLabelledImage(InfoRecord rec) {
    log("processing labelled image:", rec.inf());
    int key = rec.key();
    InfoRecord alt = mOrphanInfoRecordMap.get(key);
    if (alt == null) {
      log("...companion not found; storing in map");
      mOrphanInfoRecordMap.put(key, rec);
      cullStalePendingRecords(key);
      return;
    }
    log("...companion found:", alt);
    mOrphanInfoRecordMap.remove(key);

    InfoRecord imgRec;
    InfoRecord lblRec;
    if (rec.imageIndex() != 0) {
      imgRec = rec;
      lblRec = alt;
      checkArgument(lblRec.imageIndex() == 0);
    } else {
      imgRec = alt;
      lblRec = rec;
      checkArgument(imgRec.labelIndex() == 0);
    }

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
      String setName = "" + imgRec.imageIndex() + "_%02d";
      for (int i = 0; i < batchSize; i++) {
        byte[] imgb = Arrays.copyOfRange(imgRec.mBytes, bytesPerImage * i, bytesPerImage * (i + 1));
        BufferedImage img = ImgUtil.bytesToBGRImage(imgb, VolumeUtil.spatialDimension(imgVol));
        File baseFile = new File(targetProjectDir(), String.format(setName, i));
        File imgPath = Files.setExtension(baseFile, ImgUtil.EXT_JPEG);
        ImgUtil.writeJPG(files(), img, imgPath, null);
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
      mTargetProjectDir = new File("snapshot");
      files().remakeDirs(mTargetProjectDir);
      mTargetProjectScriptsDir = ScriptUtil.scriptDirForProject(mTargetProjectDir);
      files().mkdirs(mTargetProjectScriptsDir);
    }
    return mTargetProjectDir;
  }

  /**
   * If for some reason there are orphaned images or labels, discard them
   */
  private void cullStalePendingRecords(int key) {
    List<Integer> keysToRemove = arrayList();
    for (int k : mOrphanInfoRecordMap.keySet()) {
      if (k < key - 10) {
        keysToRemove.add(k);
      }
    }
    if (keysToRemove.isEmpty())
      return;
    alert("Removing stale pending records of size:", keysToRemove.size());
    mOrphanInfoRecordMap.keySet().removeAll(keysToRemove);
  }

  private Files files() {
    return Files.S;
  }

  private CompileImagesConfig config() {
    return mConfig;
  }

  private String formatTensor(TensorInfo ti, float[] t) {
    todo("refactor to accept InfoRecord instead of ti");

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

  private static class InfoRecord {

    public InfoRecord(TensorInfo tensorInfo) {
      mTensorInfo = tensorInfo.build();
    }

    public int key() {
      int k1 = imageIndex();
      int k2 = labelIndex();
      int key = k1 ^ k2;
      checkArgument((k1 == 0 || k2 == 0) && key != 0);
      return key;
    }

    public void setData(byte[] bytes) {
      pr("storing byte array of length:", bytes.length);
      mBytes = bytes;
    }

    public void setData(float[] floats) {
      mFloats = floats;
    }

    public TensorInfo inf() {
      return mTensorInfo;
    }

    public int imageIndex() {
      return inf().imageIndex();
    }

    public int labelIndex() {
      return inf().labelIndex();
    }

    private final TensorInfo mTensorInfo;
    byte[] mBytes;
    float[] mFloats;
  }

  private CompileImagesConfig mConfig;
  private ModelWrapper mModel;
  private NeuralNetwork mNetwork;
  private int mState;
  private Thread mThread;

  private Map<Integer, InfoRecord> mOrphanInfoRecordMap = hashMap();
  private File mTargetProjectDir;
  private File mTargetProjectScriptsDir;

}
