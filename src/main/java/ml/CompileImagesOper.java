package ml;

import static js.base.Tools.*;

import java.io.File;
import java.util.List;
import java.util.SortedMap;

import gen.CompileImagesConfig;
import gen.FloatFormat;
import gen.NeuralNetwork;
import gen.TensorInfo;
import gen.TrainParam;
import js.app.AppOper;
import js.base.DateTimeTools;
import js.file.DirWalk;
import js.file.Files;
import js.graphics.Inspector;
import ml.img.ImageCompiler;

/**
 * Compiles images into a form to be consumed by pytorch. Partitions images into
 * train and test sets; optionally generates training sets to be consumed by an
 * external training session running in parallel
 */
public final class CompileImagesOper extends AppOper {

  @Override
  public String userCommand() {
    return "compileimages";
  }

  @Override
  public String getHelpDescription() {
    return "Compile sets of training or testing images";
  }

  @Override
  public CompileImagesConfig defaultArgs() {
    return CompileImagesConfig.DEFAULT_INSTANCE;
  }

  @Override
  public CompileImagesConfig config() {
    return super.config();
  }

  private TrainParam trainParam() {
    return config().trainParam();
  }

  @Override
  public void perform() {
    if (config().prepare()) {
      prepareTrainService();
      return;
    }

    writeModelData();
    ImageCompiler imageCompiler = new ImageCompiler(config(), network(), files());
    Inspector insp = Inspector.build(config().inspectionDir());
    imageCompiler.setInspector(insp);

    if (config().trainService())
      performTrainService(imageCompiler);
    else
      imageCompiler.compileTrainSet(config().targetDirTrain());

    insp.flush();
  }

  private File modelDataDir() {
    return Files.assertNonEmpty(config().targetDirModel(), "target_dir_model");
  }

  private void writeModelData() {
    File modelDataDir = modelDataDir();
    files().remakeDirs(modelDataDir);
    files().writePretty(new File(modelDataDir, "network.json"), network());
    files().writePretty(new File(modelDataDir, "train_param.json"), trainParam());
  }

  private NeuralNetwork network() {
    if (mCompiledNetwork == null) {
      NeuralNetwork netIn = NetworkUtil.resolveNetwork(config().network(), config().networkPath());
      NetworkAnalyzer analyzer = NetworkAnalyzer.build(netIn);
      mCompiledNetwork = analyzer.result();
    }
    return mCompiledNetwork;
  }

  private long currentTime() {
    return System.currentTimeMillis();
  }

  private void prepareTrainService() {
    // Delete any existing signature file, or 'stop' signal file;
    // If either exists, then pause a bit afterward to try to avoid race conditions
    files().deletePeacefully(sigFile());
    files().deletePeacefully(stopSignalFile());

    // Delete existing training set subdirectories, or any temporary file associated with them
    if (config().targetDirTrain().isDirectory()) {
      DirWalk w = new DirWalk(config().targetDirTrain()).includeDirectories().withRecurse(false);
      for (File f : w.files()) {
        if (!f.isDirectory()) {
          // If it is a python logging file (.json, .tmp, .dat), delete it
          String ext = Files.getExtension(f);
          if (ext.equals("json") || ext.equals("tmp") || ext.equals("dat"))
            files().deleteFile(f);
          continue;
        }

        if (config().retainExistingTrainingSets() && f.getName().startsWith(STREAM_PREFIX)) {
          alert("Retaining existing training sets");
          continue;
        }

        if (f.getName().equals("_temp_") || f.getName().startsWith(STREAM_PREFIX))
          files().deleteDirectory(f);
      }
    }

    // Write a new signature file with the current time
    files().writeString(sigFile(), "" + System.currentTimeMillis());
    validateCheckpoints();
  }

  private void performTrainService(ImageCompiler imageCompiler) {
    String signature = readSignature();
    checkState(nonEmpty(signature), "No signature file found; need to prepare?");

    // Choose a temporary filename that can be atomically renamed when it is complete
    //
    File tempDir = new File(config().targetDirTrain(), "_temp_");
    Files.assertDoesNotExist(tempDir, "Found old directory; need to prepare?");

    checkpointDir();

    while (true) {
      if (!signature.equals(readSignature())) {
        log("(CompileImagesOper: Signature file has changed or disappeared, stopping)");
        break;
      }

      if (countTrainSets() >= trainParam().maxTrainSets()) {
        if (stopIfInactive())
          break;
        updateLogging();
        DateTimeTools.sleepForRealMs(100);
        continue;
      }

      long startTime = System.currentTimeMillis();

      imageCompiler.compileTrainSet(tempDir);

      // Choose a name for the new set
      //
      File newDir = null;
      while (true) {
        newDir = new File(config().targetDirTrain(), STREAM_PREFIX + mNextStreamSetNumber);
        mNextStreamSetNumber++;
        checkState(!newDir.exists(), "Stream directory already exists; need to prepare?", newDir);
        break;
      }
      log("Generated set:", newDir.getName());
      files().moveDirectory(tempDir, newDir);
      mLastGeneratedFilesTime = currentTime();

      if (verbose()) {
        long elapsed = mLastGeneratedFilesTime - startTime;
        float sec = elapsed / 1000f;
        if (mAvgGeneratedTimeSec < 0)
          mAvgGeneratedTimeSec = sec;
        mAvgGeneratedTimeSec = (0.1f * sec) + (1 - 0.1f) * mAvgGeneratedTimeSec;
        if (mAvgReportedCounter < 20) {
          mAvgReportedCounter++;
          pr("Time to generate training set:", sec, "sm:", mAvgGeneratedTimeSec);
        }
      }

      updateCheckpoints();
    }
  }

  /**
   * Count the number of subdirectories with prefix "set_"
   */
  private int countTrainSets() {
    int count = 0;
    DirWalk w = new DirWalk(config().targetDirTrain()).includeDirectories().withRecurse(false);
    for (File f : w.files()) {
      if (f.isDirectory() && f.getName().startsWith(STREAM_PREFIX))
        count++;
    }
    return count;
  }

  private boolean stopIfInactive() {
    long curr = currentTime();
    if (mLastGeneratedFilesTime == 0)
      mLastGeneratedFilesTime = curr;
    if (curr - mLastGeneratedFilesTime > DateTimeTools.MINUTES(5)) {
      pr("...a lot of time has elapsed since we had to generate files; assuming client is not running");
      return true;
    }
    return false;
  }

  // ------------------------------------------------------------------
  // Tensor logging
  // ------------------------------------------------------------------

  private void updateLogging() {
    File logDir = config().targetDirTrain();
    DirWalk w = new DirWalk(logDir).withRecurse(false).withExtensions("json");
    for (File infoFile : w.files()) {
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
  }

  private String formatTensor(TensorInfo ti, float[] t) {
    if (false && alert("verifying formatting"))
      verifyFormatting();

    FloatFormat fmt = getFloatFormatString(t);
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
          sb.append(" │ ");  // Note: this is a unicode char taller than the vertical brace
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
    return spaces(width ) ;
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

  private static void verifyFormatting() {
    float[] x = new float[1];
    float[] vals = { 0.0001f, 0.00001f, 0.000001f, 0.001f, 0.01f, 0.1f, 1.0f, 10.0f, 100.0f, 1000f, 10000f };
    for (float z2 : vals) {
      float z = -z2;
      x[0] = z;
      FloatFormat fmt = getFloatFormatString(x);
      pr("z:", z, quote(fmt(fmt, z)), quote(fmt(fmt, z2)));
    }
  }

  // ------------------------------------------------------------------
  // Signature file, a signal sent by client to stop service
  // ------------------------------------------------------------------

  private String readSignature() {
    return Files.readString(sigFile(), "");
  }

  private File sigFile() {
    return new File(config().targetDirTrain(), "sig.txt");
  }

  private File stopSignalFile() {
    return new File(config().targetDirTrain(), "stop.txt");
  }

  //------------------------------------------------------------------
  // Checkpoint management
  // ------------------------------------------------------------------

  /**
   * Get checkpoint directory. If none exists, then constructs it if in prepare
   * mode; otherwise, fails
   */
  private File checkpointDir() {
    if (mCheckpointDir == null) {
      File d = Files.assertNonEmpty(config().targetDirCheckpoint());
      if (config().prepare())
        files().mkdirs(d);
      else
        Files.assertDirectoryExists(d, "No checkpoint directory found; need to prepare?");
      mCheckpointDir = d;
    }
    return mCheckpointDir;
  }

  private File mCheckpointDir;

  /**
   * Determine if the existing checkpoints are still valid. If not, delete them.
   * We look at a checksum of the network parameters to determine this.
   */
  private void validateCheckpoints() {
    File networkChecksumFile = new File(checkpointDir(), "network_checksum.txt");
    String savedChecksum = Files.readString(networkChecksumFile, "");
    String currentChecksum = "" + network().toJson().toString().hashCode();
    if (!currentChecksum.equals(savedChecksum)) {
      SortedMap<Integer, File> epochMap = getCheckpointEpochs();
      if (!epochMap.isEmpty()) {
        pr("...deleting existing checkpoints, since network has changed");
      }
      for (File f : epochMap.values())
        files().deleteFile(f);
      files().writeString(networkChecksumFile, currentChecksum);
    }
  }

  private void updateCheckpoints() {
    SortedMap<Integer, File> epochMap = getCheckpointEpochs();
    List<Integer> epochs = arrayList();
    epochs.addAll(epochMap.keySet());

    while (epochs.size() > trainParam().maxCheckpoints()) {

      final double power = 0.5f;
      int maxEpoch = last(epochs);

      // Throw out value whose neighbors have fractions closest together
      double[] coeff = new double[epochs.size()];
      for (int i = 0; i < epochs.size(); i++)
        coeff[i] = checkpointCoefficient(epochs.get(i), maxEpoch, power);

      double minDiff = 0;
      int minIndex = -1;
      for (int i = 1; i < epochs.size() - 1; i++) {
        double diff = coeff[i + 1] - coeff[i - 1];
        if (minIndex < 0 || minDiff > diff) {
          minIndex = i;
          minDiff = diff;
        }
      }

      int discardEpoch = epochs.get(minIndex);
      log("trimming checkpoints:", INDENT, epochs);
      log("discarding checkpoint for epoch:", discardEpoch);
      epochs.remove(minIndex);
      File checkpointFile = Files.assertExists(epochMap.remove(discardEpoch), "checkpoint file");
      files().deleteFile(checkpointFile);
      log("after trim checkpoints:", INDENT, epochs);
    }
  }

  /**
   * Calculate coefficient for a particular epoch, where max has 1.0
   * 
   * We maintain a finite set of checkpoints, so that the epoch numbers try to
   * approximate a nonlinear curve such that the epochs are clustered more
   * towards the highest epoch number reached.
   * 
   * The nonlinearity is determined by the power parameter
   */
  private static double checkpointCoefficient(int epoch, int maxEpoch, double power) {
    return Math.pow(epoch / (double) maxEpoch, 1 / power);
  }

  /**
   * Look at the checkpoint files and construct a sorted map of epoch numbers =>
   * filenames
   */
  private SortedMap<Integer, File> getCheckpointEpochs() {
    SortedMap<Integer, File> map = treeMap();
    for (File file : new DirWalk(checkpointDir()).withRecurse(false).withExtensions("pt").files()) {
      int key = Integer.parseInt(Files.basename(file));
      map.put(key, file);
    }
    return map;
  }

  // ------------------------------------------------------------------

  private static final String STREAM_PREFIX = "set_";

  private int mNextStreamSetNumber;
  private long mLastGeneratedFilesTime;
  private float mAvgGeneratedTimeSec = -1;
  private int mAvgReportedCounter;
  private NeuralNetwork mCompiledNetwork;
}
