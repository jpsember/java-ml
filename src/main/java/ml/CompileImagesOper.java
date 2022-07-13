package ml;

import static js.base.Tools.*;

import java.io.File;
import java.util.List;
import java.util.SortedMap;

import gen.CmdItem;
import gen.CompileImagesConfig;
import gen.NeuralNetwork;
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
 * external training session running in parallel, and processes log files
 * received from the training session
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
    ImageCompiler imageCompiler = new ImageCompiler(config(), model(), files());
    Inspector insp = Inspector.build(config().inspectionDir());
    imageCompiler.setInspector(insp);

    try {
      if (config().trainService())
        performTrainService(imageCompiler);
      else
        imageCompiler.compileTrainSet(trainParam().targetDirTrain());

      insp.flush();
    } finally {
      stopLogging();
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

  private long currentTime() {
    return System.currentTimeMillis();
  }

  private void prepareTrainService() {
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

      imageCompiler.compileTrainSet(tempDir);

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

}
