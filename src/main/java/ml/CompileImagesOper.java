package ml;

import static js.base.Tools.*;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.InputStream;

import gen.CompileImagesConfig;
import gen.ImageSetInfo;
import js.app.AppOper;
import js.base.DateTimeTools;
import js.file.DirWalk;
import js.file.Files;
import js.graphics.ImgUtil;
import js.graphics.ScriptUtil;
import js.graphics.gen.Script;

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

    mImageCompiler = new ImageCompiler(config());
    mImageCompiler.setFiles(files());
    mImageCompiler.compileTestSet(config().targetDirTest());
    if (Files.nonEmpty(config().targetDirInspect())) {
      generateInspection();
      if (alert("temporarily skipping training stuff"))
        return;
    }

    if (config().trainService()) {
      performTrainService();
    } else {
      mImageCompiler.compileTrainSet(config().targetDirTrain());
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

  private long currentTime() {
    return System.currentTimeMillis();
  }

  private void performTrainService() {

    // Choose a temporary filename that can be atomically renamed when it is complete
    //
    File tempDir = new File(config().targetDirTrain(), "_temp_");

    // Clean up any old directory
    //
    files().deleteDirectory(tempDir);

    while (true) {
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

  private void generateInspection() {

    todo("bad assumption being made here: the output labels are NOT directly equivalent to the input labels.");
    
    File inspectDir = files().remakeDirs(config().targetDirInspect());
    File sourceDir = config().targetDirTest();

    File imagesPath = new File(sourceDir, "images.bin");
    File labelsPath = new File(sourceDir, "labels.bin");
    File infoPath = new File(sourceDir, "image_set_info.json");
    ImageSetInfo imageSetInfo = Files.parseAbstractData(ImageSetInfo.DEFAULT_INSTANCE, infoPath);

    ModelServiceProvider p = mImageCompiler.buildModelServiceProvider();
    InputStream imageStream = Files.openInputStream(imagesPath);
    InputStream labelStream = Files.openInputStream(labelsPath);

    for (int imageNumber = 0; imageNumber < imageSetInfo.imageCount(); imageNumber++) {
      BufferedImage image = p.decodeImage(Files.readBytes(imageStream, imageSetInfo.imageLengthBytes()));
      File targetImageFile = new File(inspectDir,
          Files.setExtension(String.format("%03d", imageNumber), ImgUtil.EXT_JPEG));
      ImgUtil.writeImage(files(), image, targetImageFile);

      Script.Builder script = Script.newBuilder();

      byte[] trainingLabels = Files.readBytes(labelStream, imageSetInfo.labelLengthBytes());
      p.parseTrainingLabels(trainingLabels, script);
      files().writePretty(ScriptUtil.scriptPathForImage(targetImageFile), script.build());
    }
    Files.close(labelStream, imageStream);
  }

  private static final String STREAM_PREFIX = "set_";

  private ImageCompiler mImageCompiler;
  private int mNextStreamSetNumber;
  private long mLastGeneratedFilesTime;
}
