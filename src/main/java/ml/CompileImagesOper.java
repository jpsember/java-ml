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

  private void performTrainService() {

    createSignatureFile();

    // Choose a temporary filename that can be atomically renamed when it is complete
    //
    File tempDir = new File(config().targetDirTrain(), "_temp_");

    // Clean up any old directory
    //
    files().deleteDirectory(tempDir);

    while (!stopDetected()) {
      if (countTrainSets() >= config().maxTrainSets()) {
        if (stopIfInactive())
          break;
        DateTimeTools.sleepForRealMs(100);
        continue;
      }

      mImageCompiler.compileTrainSet(tempDir);

      // Choose a name for the new set
      //
      File newDir = null;
      while (true) {
        newDir = new File(config().targetDirTrain(), STREAM_PREFIX + mNextStreamSetNumber);
        mNextStreamSetNumber++;
        if (!newDir.exists())
          break;
        alert("Stream directory already exists:", newDir);
      }
      log("Generated set:", newDir.getName());
      files().moveDirectory(tempDir, newDir);
      mLastGeneratedFilesTime = currentTime();
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

  private void createSignatureFile() {

    // Write a new signature file
    mSignatureFile = new File(config().targetDirTrain(), System.currentTimeMillis() + "." + EXT_SIG);
    files().writeString(mSignatureFile, "");

    boolean staleFilesFound = false;

    Integer oldestIndexFound = null;

    {// Delete any other signature files and generated directories; try to do this as quickly as possible
     //
      DirWalk w = new DirWalk(config().targetDirTrain()).withRecurse(false).includeDirectories();
      for (File f : w.files()) {
        String name = f.getName();
        if (name.startsWith("set_")) {
          staleFilesFound = true;
          log("deleting stale training set:", name);
          files().deleteDirectory(f);
          // Avoid reusing indexes close to ones we found
          oldestIndexFound = Integer.parseInt(chompPrefix(name, "set_"));
        }
      }
    }
    if (oldestIndexFound != null)
      mNextStreamSetNumber = oldestIndexFound + 100;

    {
      pr("signature file:", mSignatureFile);
      DirWalk w = new DirWalk(config().targetDirTrain()).withRecurse(false).withExtensions(EXT_SIG);
      if (!w.files().isEmpty()) {
        for (File f : w.files()) {
          if (f.getName().equals(mSignatureFile.getName()))
            continue;
          pr("f:", f);
          pr("s:", mSignatureFile);
          log("deleting stale signature file:", f.getName());
          files().deleteFile(f);
        }
        staleFilesFound = true;
      }
    }

    if (staleFilesFound) {
      // Sleep for a bit to let some other service notice we've deleted their signature file
      log("sleeping for a bit to let other services shut down gracefully");
      DateTimeTools.sleepForRealMs(5000);
    }
  }

  private boolean stopDetected() {
    boolean detected = !mSignatureFile.exists();
    if (detected)
      pr("...stop signal detected, file has disappeared:", mSignatureFile.getName());
    return detected;
  }

  private static final String EXT_SIG = "sig";
  private File mSignatureFile;
  // ------------------------------------------------------------------

  private static final String STREAM_PREFIX = "set_";

  private ImageCompiler mImageCompiler;
  private int mNextStreamSetNumber;
  private long mLastGeneratedFilesTime;
  private NeuralNetwork mCompiledNetwork;
}
