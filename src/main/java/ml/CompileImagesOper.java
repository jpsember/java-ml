package ml;

import static js.base.Tools.*;

import java.io.File;

import gen.CompileImagesConfig;
import js.app.AppOper;
import js.base.DateTimeTools;
import js.file.DirWalk;

public class CompileImagesOper extends AppOper {

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

    ImageCompiler c = new ImageCompiler(config());
    c.setFiles(files());
    c.compileTestSet(config().targetDirTest());
    if (config().trainService()) {
      performTrainService(c);
    } else {
      c.compileTrainSet(config().targetDirTrain());
    }
  }

  @Override
  public CompileImagesConfig defaultArgs() {
    return CompileImagesConfig.DEFAULT_INSTANCE;
  }

  @SuppressWarnings("unchecked")
  @Override
  public CompileImagesConfig config() {
    return super.config();
  }

  private void performTrainService(ImageCompiler c) {

    // Choose a temporary filename that can be atomically renamed when it is complete
    File tempDir = new File(config().targetDirTrain(), "_temp_");
    // Clean up any old directory
    files().deleteDirectory(tempDir);

    long endTimestamp = System.currentTimeMillis() + 120000;
    while (true) {
      if (System.currentTimeMillis() > endTimestamp)
        break;

      int k = countTrainSets();
      if (k >= config().maxTrainSets()) {
        DateTimeTools.sleepForRealMs(100);
        continue;
      }

      c.compileTrainSet(tempDir);

      // Choose a name for the new set
      //
      File newSetDir = null;
      while (true) {
        newSetDir = new File(config().targetDirTrain(), STREAM_PREFIX + mSetNumber);
        mSetNumber++;
        if (!newSetDir.exists())
          break;
        pr("??? directory already exists:", newSetDir);
      }
      files().moveDirectory(tempDir, newSetDir);
      //      if (alert("sleeping a bit"))
      //        DateTimeTools.sleepForRealMs(1500);
    }
  }

  private static final String STREAM_PREFIX = "set_";

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

  private int mSetNumber;
}
