package ml;

import static js.base.Tools.*;

import java.awt.image.BufferedImage;
import java.io.DataOutputStream;
import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.Random;

import gen.CompileImagesConfig;
import gen.Vol;
import js.base.BaseObject;
import js.file.DirWalk;
import js.file.Files;
import js.geometry.IPoint;
import js.geometry.MyMath;
import js.graphics.ImgUtil;
import js.graphics.ScriptElement;
import js.graphics.ScriptUtil;
import js.graphics.gen.Script;

public class ImageCompiler extends BaseObject {

  public ImageCompiler() {
    this(null);
  }

  public ImageCompiler(CompileImagesConfig config) {
    mConfig = nullTo(config, CompileImagesConfig.DEFAULT_INSTANCE).build();
    int seed = config().seed();
    if (seed <= 0)
      seed = 1965;
    mRandom = new Random(seed);
  }

  public void compileTrainSet(File targetDir) {
    auxCompile(targetDir, trainEntries(), true);
  }

  public void compileTestSet(File targetDir) {
    auxCompile(targetDir, testEntries(), false);
  }

  private void auxCompile(File targetDir, List<Entry> entries, boolean training) {
    files().mkdirs(targetDir);
    File imagePath = new File(targetDir, "images.bin");
    File labelsPath = new File(targetDir, "labels.bin");

    DataOutputStream imagesStream = new DataOutputStream(files().outputStream(imagePath));
    DataOutputStream labelsStream = new DataOutputStream(files().outputStream(labelsPath));

    for (Entry entry : entries) {
      BufferedImage img = ImgUtil.read(entry.imageFile);
      checkImageSizeAndType(entry.imageFile, img, config().imageVol());

      todo("transform image randomly if training image");
      mWorkArray = ImgUtil.floatPixels(img, config().imageVol().depth(), mWorkArray);
      files().writeFloatsLittleEndian(mWorkArray, imagesStream);
      todo("write something to labels", entry.scriptElements.size());
    }
    Files.close(imagesStream, labelsStream);
  }

  private List<Entry> entries() {
    if (mEntries == null) {
      List<Entry> ents = arrayList();
      File imageDir = Files.assertDirectoryExists(config().sourceDir());
      File scriptDir = ScriptUtil.scriptDirForProject(imageDir);
      Files.assertDirectoryExists(scriptDir, "script directory");
      DirWalk w = new DirWalk(imageDir).withRecurse(false).withExtensions(ImgUtil.EXT_JPEG);
      for (File f : w.files()) {
        Entry ent = new Entry();
        ent.imageFile = f;
        File scriptFile = ScriptUtil.scriptPathForImage(f);
        if (scriptFile.exists()) {
          Script script = ScriptUtil.from(scriptFile);
          ent.scriptElements = script.items();
        } else
          ent.scriptElements = arrayList();
      }
      int testCount = Math.min(config().maxTestImagesCount(),
          (config().maxTestImagesPct() * ents.size()) / 100);
      int trainCount = ents.size() - testCount;
      checkArgument(Math.min(testCount, trainCount) > 0, "insufficient images:", ents.size(), "train:",
          trainCount, "test:", testCount);
      MyMath.permute(ents, random());

      mEntries = ents;
      mTrainEntries = ents.subList(0, trainCount);
      mTestEntries = ents.subList(trainCount, ents.size());
    }
    return mEntries;
  }

  private List<Entry> trainEntries() {
    entries();
    return mTrainEntries;
  }

  private List<Entry> testEntries() {
    entries();
    return mTestEntries;
  }

  private CompileImagesConfig config() {
    return mConfig;
  }

  private Files files() {
    return mFiles;
  }

  private Random random() {
    return mRandom;
  }

  private void checkImageSizeAndType(File imageFile, BufferedImage img, Vol expectedVol) {
    IPoint imgSize = ImgUtil.size(img);
    if (mExpectedImageSize == null) {
      mExpectedImageSize = new IPoint(expectedVol.width(), expectedVol.height());
      Integer channels = sImgChannelsMap.get(img.getType());
      if (channels == null)
        throw badArg("Unsupported image type:", INDENT, ImgUtil.toJson(img));
      if (channels != expectedVol.depth())
        throw badArg("Unsupported image type; wanted depth:", expectedVol.depth(), "got:", INDENT,
            ImgUtil.toJson(img));
      mExpectedImageType = img.getType();
    }
    if (img.getType() != mExpectedImageType)
      badArg("Unexpected image type, wanted:", mExpectedImageType, "but got:", INDENT, ImgUtil.toJson(img));
    if (!imgSize.equals(mExpectedImageSize))
      badArg("Unexpected image size, wanted:", mExpectedImageSize, "but got:", INDENT, ImgUtil.toJson(img));
  }

  private static final Map<Integer, Integer> sImgChannelsMap = mapWith(//
      BufferedImage.TYPE_INT_RGB, 3, //
      BufferedImage.TYPE_INT_BGR, 3, //
      BufferedImage.TYPE_BYTE_GRAY, 1, //
      BufferedImage.TYPE_USHORT_GRAY, 1 //
  );

  private static class Entry {
    File imageFile;
    List<ScriptElement> scriptElements;
  }

  private final CompileImagesConfig mConfig;
  private final Random mRandom;
  private Files mFiles = Files.S;
  private List<Entry> mEntries;
  private List<Entry> mTestEntries;
  private List<Entry> mTrainEntries;
  private int mExpectedImageType;
  private IPoint mExpectedImageSize = null;
  private float[] mWorkArray;
}
