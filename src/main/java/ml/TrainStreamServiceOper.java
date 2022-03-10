package ml;

import static js.base.Tools.*;

import java.awt.image.BufferedImage;
import java.io.DataOutputStream;
import java.io.File;
import java.util.List;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import js.app.AppOper;
import js.base.DateTimeTools;
import js.base.TaskProcessor;
import js.data.DataUtil;
import js.file.DirWalk;
import js.file.Files;
import js.geometry.MyMath;
import js.graphics.Inspector;
import js.json.JSMap;
import js.parsing.RegExp;
import ml.ModelHandler;
import ml.ModelInputReceiver;
import ml.ProgressFile;
import ml.ModelWrapper;
import js.system.SystemUtil;
import gen.AugmentationConfig;
import gen.TrainConfig;
import gen.TrainStream;
import gen.TransformWrapper;
import gen.AnnotationFile;
import js.graphics.gen.ScriptElementList;

/**
 * Generates images for training a network
 * 
 * Designed to be run as a separate process, to provide data to a train
 * operation (i.e. Python).
 */
public final class TrainStreamServiceOper extends AppOper {

  @Override
  public String userCommand() {
    return "train-stream";
  }

  @Override
  public String getHelpDescription() {
    return "start streaming service for training images";
  }

  @Override
  public TrainStream defaultArgs() {
    return TrainStream.DEFAULT_INSTANCE;
  }

  @Override
  public void perform() {
    log("performance starts");
    mConfig = config();

    try {
      auxPerform();
    } finally {
      Files.closePeacefully(mProgressFile);
    }
  }

  // ------------------------------------------------------------------
  // Enforcing a single instance of the stream service running at once
  // ------------------------------------------------------------------

  private void writeProcessSignature() {
    deleteAllSignatures();
    mSigFile = new File(streamDir(), SIGNATURE_PREFIX + System.currentTimeMillis() + "." + SIGNATURE_EXT);
    Files.S.writeString(mSigFile, "");
    log("...wrote signature file:", mSigFile.getName());
  }

  private void deleteAllSignatures() {
    for (File f : new DirWalk(streamDir()).withExtensions(SIGNATURE_EXT).withRecurse(false).files())
      Files.S.deletePeacefully(f);
  }

  private static final String SIGNATURE_PREFIX = "signature-";
  private static final String SIGNATURE_EXT = "sig";

  private File mSigFile;

  //------------------------------------------------------------------

  private AugmentationConfig augmentationConfig() {
    return mTrainConfig.augmentationConfig();
  }

  private Random random() {
    return mImageHandler.random();
  }

  private File streamDir() {
    return mConfig.trainStreamDir();
  }

  /**
   * Extracted from perform() so it can catch any exceptions thrown and clean
   * things up
   */
  private void auxPerform() {
    Files.S.mkdirs(streamDir());
    readTrainConfig();
    mInspectionManager = Inspector.build(mConfig.inspectionsDir());
    mInspectionManager.maxSamples(200);
    mHandler = ModelHandler.construct(mTrainConfig.architecture());

    mProgressFile = new ProgressFile(mHandler, mConfig);

    constructImageHandler();
    compileTestFiles();

    constructImageHandler();

    mRecords = recallTrainImages();

    // If the 'one_shot' option is set, we don't enter an endless streaming loop.
    // Instead, we generate one set of files and exit.
    //
    if (mConfig.oneShot()) {
      MyMath.permute(mRecords, mImageHandler.random());
      generateStreamingSet(streamDir(), "train", mRecords);
      return;
    }

    writeProcessSignature();

    while (!stopFlagDetected())
      streamMainLoop();

    log("...stream operation exiting");
    mSigFile = Files.S.deletePeacefully(mSigFile);
  }

  private void streamMainLoop() {
    mSetsGenerated = 0;
    updateCurrentTime();

    // We generate the sets in a background thread, but only one at a time... we use the
    // idle time to process any additional python commands that may have appeared
    //
    processPythonCommands();
    mTaskProcessor.submit(() -> {
      performGenerateSetsTask();
      return null;
    });
    mTaskProcessor.fetchAllResults(() -> processPythonCommands());

    generateLogInfo();

    stopIfInactive();

    if (mSetsGenerated == 0)
      DateTimeTools.sleepForRealMs(mConfig.devMode() ? 2000 : 30);
  }

  /**
   * This method is run in a background thread, but is guaranteed to not be
   * reentrant
   */
  private void performGenerateSetsTask() {
    checkState(!mPerformingGenerateSetsTask);
    mPerformingGenerateSetsTask = true;

    int currentSets = countStreamSets();
    int targetSets = determineMinimumBufferedSetCount();
    int setsToGenerate = targetSets - countStreamSets();

    while (mSetsGenerated < setsToGenerate) {
      log("desired sets:", targetSets, "current:", currentSets);
      File subdir = null;
      File finalSubdir = null;

      // We add a suffix representing the current timestamp, so the client
      // can read them in the order they were created.  This makes unit tests deterministic

      long modTime = DateTimeTools.getRealMs();
      for (long timestamp = modTime;; timestamp++) {
        String newName = String.format("set_%d", timestamp);
        subdir = new File(streamDir(), "temp_" + newName);
        finalSubdir = new File(streamDir(), newName);
        if (!subdir.exists() && !finalSubdir.exists())
          break;
      }
      Files.S.mkdirs(subdir);
      MyMath.permute(mRecords, mImageHandler.random());
      generateStreamingSet(subdir, "train", mRecords);

      subdir.renameTo(finalSubdir);
      mLastGeneratedFilesTime = currentTime();
      mSetsGenerated++;
    }
    mPerformingGenerateSetsTask = false;
  }

  private void stopIfInactive() {
    StatCalculator stats = mElapsedTimeStats;
    long elapsed = currentTime() - mLastGeneratedFilesTime;
    float elapsedSec = elapsed / 1000f;
    stats.add(elapsedSec);
    if (elapsed > DateTimeTools.SECONDS(mConfig.maxInactivitySeconds())) {
      pr("...", elapsedSec, "seconds elapsed since we had to generate files; assuming client is not running");
      pr("Stats:", INDENT, stats);
      mStopFlag = true;
    }
  }

  private long currentTime() {
    return mCurrentTime;
  }

  private long updateCurrentTime() {
    mCurrentTime = System.currentTimeMillis();
    if (mStartTime == 0) {
      mStartTime = mCurrentTime;
      mLastGeneratedFilesTime = mCurrentTime;
    }
    return currentTime();
  }

  /**
   * We don't generate a large number of sets right at the start, since this
   * will create unnecessary delays
   */
  private int determineMinimumBufferedSetCount() {
    int maxSets = mConfig.maxSets();
    if (maxSets <= 0)
      maxSets = 2;
    return MyMath.clamp((int) ((currentTime() - mStartTime) / DateTimeTools.SECONDS(8)), 1,
        mConfig.maxSets());
  }

  private void generateLogInfo() {
    if (!verbose() || mSetsGenerated == 0)
      return;

    float elapsedMs = System.currentTimeMillis() - currentTime();
    long bytesUsed = SystemUtil.memoryUsed();
    float mbUsed = bytesUsed / (float) DataUtil.ONE_MB;
    if (mPrevMbUsed == 0)
      mPrevMbUsed = mbUsed;
    float mbDiff = mbUsed - mPrevMbUsed;
    mPrevMbUsed = mbUsed;
    log("Generated", mSetsGenerated, "sets; each took", (elapsedMs / mSetsGenerated) / 1000, "s", "MB used:",
        mbUsed, "Change:", mbDiff);
  }

  private boolean stopFlagDetected() {
    if (!mStopFlag) {
      File stopFile = new File(streamDir(), "stop.txt");
      File auxStopFile = new File(streamDir(), ".stop_signal_java.txt");
      if (stopFile.exists() || auxStopFile.exists()) {
        log("stop flag detected, quitting stream");
        mStopFlag = true;
      }
    }
    if (!mStopFlag) {
      if (!mSigFile.exists()) {
        log("signature file has disappeared, quitting stream");
        mStopFlag = true;
      }
    }
    return mStopFlag;
  }

  /**
   * Look for commands from the Python script, and process them. These are files
   * in the stream directory with the form "javacmd_[0-9]*.cmd".
   */
  private void processPythonCommands() {
    Pattern commandPattern = RegExp.pattern("javacmd_(\\d+)\\.cmd");

    List<File> commandFiles = arrayList();
    for (File file : Files.files(streamDir())) {
      String filename = file.getName();
      Matcher match = commandPattern.matcher(filename);
      if (!match.matches())
        continue;
      commandFiles.add(file);
    }

    if (commandFiles.size() >= 100) {
      pr("There are a lot of Python command files:", commandFiles.size());
    }

    for (File file : commandFiles) {
      // If Python generates some NaN values, they will be written to the json text file as such;
      // and we can't parse them
      String content = Files.readString(file);
      JSMap m;
      try {
        m = new JSMap(content);
      } catch (Throwable t) {
        String origContent = content;
        pr("*** Failed to parse JSMap from", file);
        final String nanString = "NaN";
        final String repString = "9999.0";
        content = content.replace(nanString, repString);
        if (!content.equals(origContent)) {
          int numNans = (content.length() - origContent.length()) / (repString.length() - nanString.length());
          pr("***", numNans, "Nan(s) encountered in file");
          if (origContent.length() < 3000)
            pr("Content:", origContent);
        }
        throw asRuntimeException(t);
      }
      Files.S.deleteFile(file);
      processPythonCommand(m);
    }
  }

  private void processPythonCommand(JSMap cmd) {
    String c = cmd.opt("cmd", "<none>");
    switch (c) {
    case "progress":
      mProgressFile.processUpdate(cmd);
      break;
    case "message":
      mProgressFile.displayMessage(cmd);
      break;
    default:
      throw die("Unrecognized command:", c, INDENT, cmd);
    }
  }

  private int countStreamSets() {
    int count = 0;
    for (File file : Files.files(streamDir())) {
      String filename = file.getName();
      if (filename.startsWith("set_"))
        count++;
    }
    return count;
  }

  private void readTrainConfig() {
    mTrainConfig = Files.parseAbstractData(TrainConfig.DEFAULT_INSTANCE,
        fileWithinTrainInputDir("train_config.json"));
  }

  private void constructImageHandler() {
    ImageHandler p = new ImageHandler(trainConfig().architecture(), augmentationConfig());
    p.setVerbose(verbose());
    mImageHandler = p;
  }

  private void compileTestFiles() {
    List<ImageRecord> records = recallTestImages();
    File streamDir = mConfig.testStreamDir();
    Files.S.remakeDirs(streamDir);
    generateStreamingSet(streamDir, "test", records);
  }

  private void generateStreamingSet(File directory, String filenamePrefix, List<ImageRecord> records) {
    log("generateStreamingSet, prefix:", filenamePrefix, "record count:", records.size());

    DataOutputStream imagesStream = Util.outputDataStream(directory, filenamePrefix + "_images.bin");
    DataOutputStream labelsStream = Util.outputDataStream(directory, filenamePrefix + "_labels.bin");
    ModelInputReceiver modelInputReceiver = mHandler.buildModelInputReceiver(imagesStream, labelsStream);
    modelInputReceiver.setInspector(mInspectionManager);

    boolean cacheImagesInMemory = calculateCacheImagesFlag(records.size());

    for (ImageRecord rec : records) {
      TransformWrapper aug = mImageHandler.buildAugmentTransform();
      ImageTransformer<BufferedImage> transformer = mHandler.buildImageTransformer(augmentationConfig(),
          random(), mTrainConfig.stats(), rec);
      transformer.setInspector(mInspectionManager);
      mImageHandler.applyCompileImagePipeline(rec.bufferedImage(), rec.annotations(), aug, transformer,
          modelInputReceiver, rec);
      if (!cacheImagesInMemory)
        rec.discardImage();
    }

    Files.close(imagesStream, labelsStream);
  }

  private ModelWrapper model() {
    return mHandler.model();
  }

  /**
   * Determine whether we should cache the images in memory. We don't do this if
   * there's a lot of them
   */
  private boolean calculateCacheImagesFlag(int numRecords) {
    long totalPixels = numRecords * (long) (model().inputImageVolumeProduct());
    boolean result = totalPixels < DataUtil.ONE_MB * 600;
    if (false && !mCacheReported) {
      pr("Total pixels (Mb):", totalPixels / DataUtil.ONE_MB, "Cache in memory:", result);
      mCacheReported = true;
    }
    return result;
  }

  private List<ImageRecord> recallTrainImages() {
    return recallImages(mConfig.trainInfoName(), mConfig.trainDirName(), false);
  }

  private List<ImageRecord> recallTestImages() {
    return recallImages(mConfig.testInfoName(), mConfig.testDirName(), true);
  }

  private List<ImageRecord> recallImages(String infoName, String imageDirName, boolean testFlag) {
    checkArgument(!nullOrEmpty(infoName), "infoName is empty");
    checkArgument(!nullOrEmpty(imageDirName), "imageDirName is empty");

    File imageDir = fileWithinTrainInputDir(imageDirName);
    File annotationPath = fileWithinTrainInputDir(infoName);

    AnnotationFile annotationFile = Files.parseAbstractData(AnnotationFile.DEFAULT_INSTANCE, annotationPath);

    List<ImageRecord> records = arrayList();
    int index = -1;
    int limit = testFlag ? mConfig.maxTestImages() : mConfig.maxTrainImages();

    for (ScriptElementList annotation : annotationFile.annotations()) {
      index++;
      if (limit > 0 && index >= limit)
        break;

      // We changed the format of the annotations file, and some test files haven't been updated
      if (testMode()) {
        annotation = Util.validate(annotation);
      }

      File file = new File(imageDir, annotationFile.filenames().get(index));
      ImageRecord record = new ImageRecord(mImageHandler, file);
      record.setAnnotations(annotation);
      records.add(record);
    }
    checkState(!records.isEmpty(), "no images found in " + mConfig.trainInputDir());
    return records;
  }

  private File fileWithinTrainInputDir(String relativePath) {
    return new File(mConfig.trainInputDir(), relativePath);
  }

  private TrainConfig trainConfig() {
    return mTrainConfig;
  }

  private TrainStream mConfig;
  private TrainConfig mTrainConfig;
  private ModelHandler mHandler;
  private ImageHandler mImageHandler;
  private ProgressFile mProgressFile;
  private List<ImageRecord> mRecords;
  private long mLastGeneratedFilesTime;
  private long mStartTime;
  private long mCurrentTime;
  private boolean mStopFlag;
  private int mSetsGenerated;
  private float mPrevMbUsed;
  private Inspector mInspectionManager;
  private StatCalculator mElapsedTimeStats = new StatCalculator();
  private boolean mCacheReported;
  private TaskProcessor<Boolean> mTaskProcessor = new TaskProcessor<>();
  private boolean mPerformingGenerateSetsTask;
}
