package ml;

import static js.base.Tools.*;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.List;
import java.util.SortedMap;

import gen.CmdItem;
import gen.CompileImagesConfig;
import gen.CompileOper;
import gen.ImageSetInfo;
import gen.NeuralNetwork;
import gen.TrainParam;
import js.app.AppOper;
import js.base.DateTimeTools;
import js.file.DirWalk;
import js.file.Files;
import js.graphics.ImgUtil;
import js.graphics.Inspector;
import js.graphics.ScriptUtil;
import js.graphics.gen.Script;
import ml.img.ImageCompiler;
import static gen.CompileOper.*;

/**
 * Compiles images into a form to be consumed by pytorch. Partitions images into
 * train and test sets; optionally generates training sets to be consumed by an
 * external training session running in parallel, and processes log files
 * received from the training session.
 * 
 * Actually, performs one of several operations related to these tasks
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

  private boolean oper(CompileOper oper) {
    return config().oper() == oper;
  }

  private TrainParam trainParam() {
    return config().trainParam();
  }

  @Override
  public void perform() {
    // The logger runs in a separate thread, so this adds some complexity
    try {
      performSubOperation();
    } finally {
      stopLogging();
    }
  }

  private void performSubOperation() {
    switch (config().oper()) {
    default:
      throw notSupported("operation not supported:", config().oper());
    case PREPARE_TRAIN:
      prepareTrainService();
      break;
    case TRAIN_SERVICE:
      performTrainService();
      break;
    case COMPILE_INFERENCE_IMAGES:
      compileInferenceImages();
      break;
    case PROCESS_INFERENCE_RESULT:
      processInferenceResult();
      break;
    }
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

  private ModelWrapper model() {
    if (mModel == null) {
      mModel = ModelWrapper.constructFor(network(), null);
    }
    return mModel;
  }

  private NeuralNetwork network() {
    if (mCompiledNetwork == null) {
      NeuralNetwork netIn = NetworkUtil.resolveNetwork(config().network(), config().networkPath());
      NetworkAnalyzer analyzer = NetworkAnalyzer.build(netIn);
      mCompiledNetwork = analyzer.result();
    }
    return mCompiledNetwork;
  }

  /**
   * Construct a modified version of the network for determining whether it has
   * changed from last time. We clear out some keys that shouldn't trigger a
   * purge of existing checkpoints
   */
  private NeuralNetwork getNetworkForChecksum() {
    return network().build().toBuilder().options(null);
  }

  private static long currentTime() {
    return System.currentTimeMillis();
  }

  private void compileInferenceImages() {
    writeModelData();
    ImageCompiler imageCompiler = new ImageCompiler(config(), model(), files());
    File inferenceDir = Files.assertNonEmpty(config().inferenceDir(), "inference_dir");
    files().remakeDirs(inferenceDir);
    files().remakeDirs(inferenceInspectionDir());
    imageCompiler.compileSet(inferenceDir, (x) -> writeInferenceImage(x));
  }

  /**
   * Write BufferedImage (produced by ImageCompiler) to inference inspection
   * directory
   */
  private void writeInferenceImage(BufferedImage image) {
    File imgFile = nextInferenceImageName(inferenceInspectionDir(), ImgUtil.EXT_JPEG);
    ImgUtil.writeImage(files(), image, imgFile);
  }

  private File nextInferenceImageName(File directory, String ext) {
    String result = String.format("%04d.%s", mInfImageNumber++, ext);
    return new File(directory, result);
  }

  private int mInfImageNumber;

  private void processInferenceResult() {
    File inferenceDir = config().inferenceDir();
    File resultsFile = new File(inferenceDir, "results.bin");
    Files.assertExists(resultsFile);

    ImageSetInfo imageSetInfo = Files.parseAbstractData(ImageSetInfo.DEFAULT_INSTANCE,
        new File(inferenceDir, "image_set_info.json"));

    //      switch (network().imageDataType()) {
    //      default:
    //        throw notSupported("network.image_data_type", mNetwork.imageDataType());
    //      case UNSIGNED_BYTE: {
    //        checkNotNull(imgRec.tensorBytes(), "no bytes in tensor");
    //        int imgLength = imgRec.tensorBytes().length;
    //
    //        // We have a stacked batch of images.
    //        int bytesPerImage = mImageSize.product() * mImageVolume.depth();
    //
    //        int batchSize = imgLength / bytesPerImage;
    //        checkArgument(imgLength % bytesPerImage == 0, "images length", imgLength,
    //            "is not a multiple of image volume", bytesPerImage);
    //        String setName = String.format("%05d_", imgRec.familyId()) + "_%02d";

    //       for (int i = 0; i < imageSetInfo.imageCount(); i++) {
    //          byte[] imgb = Arrays.copyOfRange(imgRec.tensorBytes(), bytesPerImage * i, bytesPerImage * (i + 1));
    //          BufferedImage img = ImgUtil.bytesToBGRImage(imgb, VolumeUtil.spatialDimension(imgVol));
    //          File baseFile = new File(targetProjectDir(), String.format(setName, i));
    //          File imgPath = Files.setExtension(baseFile, ImgUtil.EXT_JPEG);
    todo("when generating inference info, write images to a results subdirectory");

    //          ImgUtil.writeJPG(files(), img, imgPath, null);

    switch (network().labelDataType()) {
    case FLOAT32: {
      float[] results = Files.readFloatsLittleEndian(resultsFile, "inference_results");
      int ic = imageSetInfo.imageCount();
      int labelCount = model().imageSetInfo().labelLengthBytes() * Float.BYTES;
      checkArgument(ic * labelCount == results.length, "label size * batch != labels length");

      File scriptDir = ScriptUtil.scriptDirForProject(inferenceInspectionDir());

      for (int i = 0; i < ic; i++) {
        float[] targetBuffer = model().labelBufferFloats();
        int imgLblLen = targetBuffer.length;
        System.arraycopy(results, imgLblLen * i, targetBuffer, 0, imgLblLen);

        Script.Builder script = Script.newBuilder();
        script.items(model().transformModelOutputToScredit());
        ScriptUtil.write(files(), script, nextInferenceImageName(scriptDir, Files.EXT_JSON));
      }
    }
      break;
    default:
      throw notSupported("label data type:", network().labelDataType());
    }

  }

  private File inferenceInspectionDir() {
    if (mInferenceInspectionDir == null) {
      mInferenceInspectionDir = new File("inference_results");
    }
    return mInferenceInspectionDir;
  }

  private void prepareTrainService() {
    startLogging();
    // Delete any existing signature file, or 'stop' signal file;
    // If either exists, then pause a bit afterward to try to avoid race conditions
    files().deletePeacefully(sigFile());
    files().deletePeacefully(stopSignalFile());

    // Delete existing training set subdirectories, or any temporary file associated with them
    if (trainParam().targetDirTrain().isDirectory()) {
      DirWalk w = new DirWalk(trainParam().targetDirTrain()).includeDirectories().withRecurse(false);
      for (File f : w.files()) {
        if (!f.isDirectory()) {
          // If it is a python logging file (.json, .tmp, .dat), or a python command file, delete it
          String ext = Files.getExtension(f);
          if (ext.equals(Files.EXT_JSON) || ext.equals(Files.EXT_JSON) || ext.equals("dat")
              || ext.equals(PYTHON_CMD_EXT))
            files().deleteFile(f);
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

  private void performTrainService() {
    writeModelData();
    ImageCompiler imageCompiler = new ImageCompiler(config(), model(), files());
    Inspector insp = Inspector.build(config().inspectionDir());
    imageCompiler.setInspector(insp);
    String signature = readSignature();
    checkState(nonEmpty(signature), "No signature file found; need to prepare?");

    // Choose a temporary filename that can be atomically renamed when it is complete
    //
    File tempDir = new File(trainParam().targetDirTrain(), "_temp_");
    Files.assertDoesNotExist(tempDir, "Found old directory; need to prepare?");
    long startServiceTime = System.currentTimeMillis();

    checkpointDir();

    startLogging();
    while (true) {
      if (lp().errorFlag()) {
        sendStopCommand("Error in LogProcessor");
        break;
      }

      if (!signature.equals(readSignature())) {
        log("(CompileImagesOper: Signature file has changed or disappeared, stopping)");
        break;
      }

      int recentCheckpoint = trimCheckpoints();
      addCheckpoint(recentCheckpoint);
      if (trainTargetReached())
        break;
      if (stopFlagFound())
        break;

      if (countTrainSets() >= trainParam().maxTrainSets()) {
        if (stopIfInactive())
          break;
        DateTimeTools.sleepForRealMs(100);
        continue;
      }

      long startTime = System.currentTimeMillis();
      imageCompiler.compileSet(tempDir, null);

      // Choose a name for the new set
      //
      File newDir = null;
      while (true) {
        newDir = new File(trainParam().targetDirTrain(), STREAM_PREFIX + mNextStreamSetNumber);
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
          lp().println("Time to generate training set:", sec, "sm:", mAvgGeneratedTimeSec);
        }
      }
    }
    lp().println("Elapsed time training:",
        DateTimeTools.humanDuration(System.currentTimeMillis() - startServiceTime));
    insp.flush();
  }

  private boolean stopFlagFound() {
    if (stopSignalFile().exists()) {
      files().deletePeacefully(stopSignalFile());
      sendStopCommand("Stop signal detected");
      return true;
    }
    return false;
  }

  /**
   * Count the number of subdirectories with prefix "set_"
   */
  private int countTrainSets() {
    int count = 0;
    DirWalk w = new DirWalk(trainParam().targetDirTrain()).includeDirectories().withRecurse(false);
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
    if (curr - mLastGeneratedFilesTime > DateTimeTools.MINUTES(15)) {
      lp().println(
          "...a lot of time has elapsed since we had to generate files; assuming client is not running");
      return true;
    }
    return false;
  }

  private boolean trainTargetReached() {
    float targetLoss = trainParam().targetLoss();
    if (targetLoss > 0) {
      StatRecord loss = lp().findStat(StatRecord.LOSS);
      if (loss != null && loss.smoothedValue() <= targetLoss) {
        sendStopCommand("Target loss reached, stopping training");
        return true;
      }
    }
    int targetEpoch = trainParam().targetEpoch();
    if (targetEpoch > 0) {
      StatRecord epoch = lp().findStat(StatRecord.EPOCH);
      if (epoch != null && epoch.intValue() >= targetEpoch) {
        sendStopCommand("Target epoch reached, stopping training");
        return true;
      }
    }

    return false;
  }

  private void sendStopCommand(String optionalMessage) {
    if (nonEmpty(optionalMessage))
      lp().println(optionalMessage);
    sendCommand("stop");
  }

  // ------------------------------------------------------------------
  // Logging
  // ------------------------------------------------------------------

  private void startLogging() {
    lp().start(config(),
        // The logger runs in a different thread, so give it its own model 
        ModelWrapper.constructFor(network(), null));
  }

  private void stopLogging() {
    lp().stop();
  }

  private LogProcessor lp() {
    if (mLogProcessor == null)
      mLogProcessor = new LogProcessor();
    return mLogProcessor;
  }

  private LogProcessor mLogProcessor;

  // ------------------------------------------------------------------
  // Signature file, a signal sent by client to stop service
  // ------------------------------------------------------------------

  private String readSignature() {
    return Files.readString(sigFile(), "");
  }

  private File sigFile() {
    return new File(trainParam().targetDirTrain(), "sig.txt");
  }

  private File stopSignalFile() {
    return new File(trainParam().targetDirTrain(), "stop.txt");
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
      File d = Files.assertNonEmpty(trainParam().targetDirCheckpoint());
      if (oper(PREPARE_TRAIN))
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
    String currentChecksum = "" + getNetworkForChecksum().toJson().toString().hashCode();
    if (!currentChecksum.equals(savedChecksum)) {
      SortedMap<Integer, File> epochMap = getCheckpointEpochs();
      if (!epochMap.isEmpty()) {
        lp().println("...deleting existing checkpoints, since network has changed");
      }
      for (File f : epochMap.values())
        files().deleteFile(f);
      files().writeString(networkChecksumFile, currentChecksum);
    }
  }

  /**
   * Trim checkpoints to reasonable size; return most recent checkpoint epoch
   * (or -1 if there are none)
   */
  private int trimCheckpoints() {
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

    int mostRecentCheckpoint = -1;
    if (!epochs.isEmpty())
      mostRecentCheckpoint = last(epochs);
    return mostRecentCheckpoint;
  }

  /**
   * Instruct Python code to store a new checkpoint if appropriate
   */
  private void addCheckpoint(int recentCheckpoint) {
    long currTime = System.currentTimeMillis();
    if (mCheckpointIntervalMs == 0L) {
      mCheckpointIntervalMs = 30000;
      mLastCheckpointTime = currTime;
    }
    long msUntilSave = (mLastCheckpointTime + mCheckpointIntervalMs) - currTime;
    if (msUntilSave > 0)
      return;

    mCheckpointIntervalMs = Math.min((long) (mCheckpointIntervalMs * 1.2f), 10 * 60 * 1000);
    mLastCheckpointTime = currTime;
    sendCommand("checkpoint");
  }

  private long mLastCheckpointTime;
  private long mCheckpointIntervalMs;

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
      checkArgument(key > 0, "unexpected checkpoint:", file);
      map.put(key, file);
    }
    return map;
  }

  // ------------------------------------------------------------------
  // Commands sent to Python 
  // ------------------------------------------------------------------

  private static final String PYTHON_CMD_EXT = "pcmd";

  private void sendCommand(CmdItem.Builder cmdItem) {
    mOutCommandId++;
    cmdItem.id(mOutCommandId);
    File cmdFile = new File(trainParam().targetDirTrain(),
        String.format("cmd_%07d." + PYTHON_CMD_EXT, cmdItem.id()));
    File tmpFile = Files.addTempSuffix(cmdFile);
    Files.assertDoesNotExist(cmdFile, "sendCommand file");
    Files.assertDoesNotExist(tmpFile, "sendCommand temporary file");
    files().write(tmpFile, cmdItem);
    files().moveFile(tmpFile, cmdFile);
  }

  private void sendCommand(String... args) {
    CmdItem.Builder cmdItem = CmdItem.newBuilder();
    cmdItem.args(arrayList(args));
    sendCommand(cmdItem);
  }

  private int mOutCommandId;

  // ------------------------------------------------------------------

  private static final String STREAM_PREFIX = "set_";

  private int mNextStreamSetNumber;
  private long mLastGeneratedFilesTime;
  private float mAvgGeneratedTimeSec = -1;
  private int mAvgReportedCounter;
  private NeuralNetwork mCompiledNetwork;
  private ModelWrapper mModel;
  private File mInferenceInspectionDir;
}
