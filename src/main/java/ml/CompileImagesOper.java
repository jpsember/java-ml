package ml;

import static js.base.Tools.*;

import java.io.File;
import java.util.List;
import java.util.SortedMap;

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
      NeuralNetwork netIn = NetworkUtil.resolveNetwork(null, config().network(), config().networkPath());
      NetworkAnalyzer analyzer = NetworkAnalyzer.build(netIn);
      mCompiledNetwork = analyzer.result();
    }
    return mCompiledNetwork;
  }

  private long currentTime() {
    return System.currentTimeMillis();
  }

  private void prepareTrainService() {

    // Delete any existing signature file
    File sigFile = sigFile();
    files().deletePeacefully(sigFile);

    // Delete existing training set subdirectories, or any temporary file associated with them
    {
      DirWalk w = new DirWalk(config().targetDirTrain()).includeDirectories().withRecurse(false);
      for (File f : w.files()) {
        if (!f.isDirectory())
          continue;
        if (f.getName().equals("_temp_") || f.getName().startsWith(STREAM_PREFIX))
          files().deleteDirectory(f);
      }
    }

    // Write a new signature file with the current time
    files().writeString(sigFile, "" + System.currentTimeMillis());

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
        pr("signature file has changed or disappeared, stopping");
        break;
      }

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
  // Signature file, a signal sent by client to stop service
  // ------------------------------------------------------------------

  private String readSignature() {
    return Files.readString(sigFile(), "");
  }

  private File sigFile() {
    return new File(config().targetDirTrain(), "sig.txt");
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
