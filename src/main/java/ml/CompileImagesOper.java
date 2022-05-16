package ml;

import static js.base.Tools.*;

import java.io.File;

import gen.CompileImagesConfig;
import gen.NeuralNetwork;
import js.app.AppOper;
import js.base.DateTimeTools;
import js.file.DirWalk;
import js.file.Files;

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
  public void perform() {
    if (config().prepare()) {
      prepareTrainService();
      return;
    }

    writeModelData();

    if (config().modelDataOnly())
      return;

    mImageCompiler = new ImageCompiler(config(), network());
    mImageCompiler.setFiles(files());

    if (config().trainService()) {
      performTrainService();
    } else {
      if (includeTestSets())
        mImageCompiler.compileTestSet(config().targetDirTest());
      mImageCompiler.compileTrainSet(config().targetDirTrain());
    }
  }

  private boolean includeTestSets() {
    return Files.nonEmpty(config().targetDirTest());
  }

  private File modelDataDir() {
    return Files.assertNonEmpty(config().targetDirModel(), "target_dir_model");
  }

  private void writeModelData() {
    File modelDataDir = modelDataDir();
    files().remakeDirs(modelDataDir);
    files().writePretty(new File(modelDataDir, "network.json"), network());
  }

  private NeuralNetwork network() {
    if (mCompiledNetwork == null) {
      NeuralNetwork netIn = NetworkUtil.resolveNetwork(null, config().network(), config().networkPath());
      NetworkAnalyzer analyzer = NetworkAnalyzer.build(netIn);
      mCompiledNetwork = analyzer.result();
    }
    return mCompiledNetwork;
  }

  @Override
  public CompileImagesConfig defaultArgs() {
    return CompileImagesConfig.DEFAULT_INSTANCE;
  }

  @Override
  public CompileImagesConfig config() {
    return super.config();
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
    
    // Construct checkpoint subdirectory if necessary
    //
    File cp = Files.assertNonEmpty(config().targetDirCheckpoint());
    files().mkdirs(cp);
  }
 
  private void performTrainService() {
    String signature = readSignature();
    checkState(nonEmpty(signature), "No signature file found; need to prepare?");

    // Choose a temporary filename that can be atomically renamed when it is complete
    //
    File tempDir = new File(config().targetDirTrain(), "_temp_");
    Files.assertDoesNotExist(tempDir, "Found old directory; need to prepare?");

    File cp = Files.assertNonEmpty(config().targetDirCheckpoint());
    Files.assertDirectoryExists(cp,"No checkpoints directory found; need to prepare?");
    
    while (true) {
      if (!signature.equals(readSignature())) {
        pr("signature file has changed or disappeared, stopping");
        break;
      }

      if (countTrainSets() >= config().maxTrainSets()) {
        if (stopIfInactive())
          break;
        DateTimeTools.sleepForRealMs(100);
        continue;
      }

      long startTime = System.currentTimeMillis();

      mImageCompiler.compileTrainSet(tempDir);

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

      {
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

  // ------------------------------------------------------------------

  private static final String STREAM_PREFIX = "set_";

  private ImageCompiler mImageCompiler;
  private int mNextStreamSetNumber;
  private long mLastGeneratedFilesTime;
  private float mAvgGeneratedTimeSec = -1;
  private int mAvgReportedCounter;
  private NeuralNetwork mCompiledNetwork;
}
