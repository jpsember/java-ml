package ml.img;

import static js.base.Tools.*;
import static ml.Util.*;

import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.io.DataOutputStream;
import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.Random;

import gen.AugmentationConfig;
import gen.CompileImagesConfig;
import gen.ImageSetInfo;
import gen.NeuralNetwork;
import gen.TransformWrapper;
//import gen.TransformWrapper;
import js.base.BaseObject;
import js.file.DirWalk;
import js.file.Files;
import js.geometry.IPoint;
import js.geometry.Matrix;
import js.geometry.MyMath;
import js.graphics.ImgEffects;
import js.graphics.ImgUtil;
import js.graphics.Inspector;
import js.graphics.MonoImageUtil;
import js.graphics.ScriptElement;
import js.graphics.ScriptUtil;
import js.graphics.gen.MonoImage;
import ml.ModelHandler;
import ml.ModelServiceProvider;
import ml.ModelWrapper;
import ml.Util;

/**
 * Used by CompileImagesOper to process images
 */
public final class ImageCompiler extends BaseObject {

  public ImageCompiler(CompileImagesConfig config, NeuralNetwork network, Files files) {
    mConfig = nullTo(config, CompileImagesConfig.DEFAULT_INSTANCE).build();
    mFiles = nullTo(files, Files.S);
    int seed = config().seed();
    if (seed <= 0)
      seed = 1965;
    mRandom = new Random(seed);
    mModelHandler = ModelHandler.construct(network);
  }

  public void setInspector(Inspector inspector) {
    mInspector = Inspector.orNull(inspector);
  }

  public void compileTrainSet(File targetDir) {
    files().remakeDirs(targetDir);
    File imagePath = new File(targetDir, "images.bin");
    File labelsPath = new File(targetDir, "labels.bin");
    File infoPath = new File(targetDir, "image_set_info.json");
    ImageSetInfo.Builder imageSetInfo = ImageSetInfo.newBuilder();
    imageSetInfo.imageCount(entries().size());

    DataOutputStream imagesStream = new DataOutputStream(files().outputStream(imagePath));
    DataOutputStream labelsStream = new DataOutputStream(files().outputStream(labelsPath));

    ModelWrapper model = modelHandler().model();

    ModelServiceProvider provider = buildModelServiceProvider();
    provider.setImageStream(imagesStream);
    provider.setLabelStream(labelsStream);
    provider.storeImageSetInfo(imageSetInfo);
    checkArgument(imageSetInfo.imageLengthBytes() > 0 && imageSetInfo.labelLengthBytes() > 0);

    float[] imageFloats = null;

    for (ImageEntry entry : entries()) {
      BufferedImage img = ImgUtil.read(entry.imageFile());
      // We don't need to validate the images except on the first pass through them
      if (!mEntriesValidated) {
        checkImageSizeAndType(entry.imageFile(), img, model.inputImagePlanarSize(),
            model.inputImageChannels());
      }
      mInspector.create("orig").image(img).elements(entry.scriptElementList());
      entry.setTransform(buildAugmentTransform());

      BufferedImage targetImage = ImgUtil.build(model.inputImagePlanarSize(), img.getType());
      AugmentationConfig config = config().augmentationConfig();
      AffineTransformOp op = new AffineTransformOp(entry.transform().matrix().toAffineTransform(),
          AffineTransformOp.TYPE_BILINEAR);
      List<ScriptElement> tfm = arrayList();
      modelHandler().transformAnnotations(entry.scriptElementList().elements(), tfm, entry.transform());
      op.filter(img, targetImage);
      mInspector.create("tfm").image(targetImage).elements(tfm);

      if (false) {
        // Investigating transformations involving TYPE_USHORT_GRAY monochrome BufferedImages.

        // *** NOTE: Transforming a TYPE_USHORT_GRAY BufferedImage has strange effects that
        // look like overflow if the full 16 bit range is used.
        // Converting an 8-bit image to a  15-bit one seems to be ok.
        //
        BufferedImage img2 = ImgEffects.makeMonochrome1Channel(img);
        MonoImage monoImage = MonoImageUtil.convert8BitBufferedImageMonoImage(img2);
        img2 = MonoImageUtil.to15BitBufferedImage(monoImage);
        mInspector.create("mono").image(img2);
        BufferedImage targetImage2 = ImgUtil.build(model.inputImagePlanarSize(), img2.getType());
        op.filter(img2, targetImage2);
        mInspector.create("monotfm").image(targetImage2);
      }

      imageFloats = ImgUtil.floatPixels(targetImage, model.inputImageChannels(), imageFloats);

      if (config.adjustBrightness())
        Util.applyRandomBrightness(random(), imageFloats, config.brightShiftMin(), config.brightShiftMax());

      mInspector.create("float").imageSize(model.inputImagePlanarSize()).channels(model.inputImageChannels())
          .image(imageFloats);

      provider.accept(imageFloats, entry.scriptElementList());
      entry.releaseResources();
    }
    mEntriesValidated = true;
    Files.close(imagesStream, labelsStream);

    files().writePretty(infoPath, imageSetInfo.build());
  }

  /**
   * Construct a ModelServiceProvider for the compiler's model type
   */
  public ModelServiceProvider buildModelServiceProvider() {
    ModelServiceProvider provider = modelHandler().buildModelServiceProvider();
    provider.setModel(modelHandler().model());
    return provider;
  }

  private List<ImageEntry> entries() {
    if (mEntries == null) {
      List<ImageEntry> ents = arrayList();
      File imageDir = Files.assertDirectoryExists(config().sourceDir());
      File scriptDir = ScriptUtil.scriptDirForProject(imageDir);
      Files.assertDirectoryExists(scriptDir, "script directory");
      DirWalk w = new DirWalk(imageDir).withRecurse(false).withExtensions(ImgUtil.EXT_JPEG);
      for (File f : w.files())
        ents.add(new ImageEntry(f));
      checkArgument(ents.size() > 3, "insufficient images:", ents.size());
      MyMath.permute(ents, random());
      mEntries = ents;
    }
    return mEntries;
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

  /**
   * This is only called when constructing the first set (i.e. not on the
   * subsequent sets during streaming)
   */
  private void checkImageSizeAndType(File imageFile, BufferedImage img, IPoint expectedImageSize,
      int expectedImageChannels) {

    IPoint imgSize = ImgUtil.size(img);
    if (!imgSize.equals(expectedImageSize))
      badArg("Unexpected image size, wanted:", expectedImageSize, "but got:", INDENT, ImgUtil.toJson(img));

    Integer channels = sImgChannelsMap.get(img.getType());
    if (channels == null)
      throw badArg("Unsupported image type:", INDENT, ImgUtil.toJson(img));
    if (channels != expectedImageChannels) {
      // Special case for using color images to produce monochrome
      if (expectedImageChannels == 1 && img.getType() == BufferedImage.TYPE_3BYTE_BGR)
        ;
      else
        throw badArg("Unsupported image type; wanted channels:", expectedImageChannels, "got:", INDENT,
            ImgUtil.toJson(img));
    }
  }

  private static final Map<Integer, Integer> sImgChannelsMap = mapWith(//
      BufferedImage.TYPE_3BYTE_BGR, 3, //
      BufferedImage.TYPE_BYTE_GRAY, 1, //
      BufferedImage.TYPE_USHORT_GRAY, 1 //
  );

  private ModelHandler modelHandler() {
    return mModelHandler;
  }

  private TransformWrapper buildAugmentTransform() {
    AugmentationConfig ac = config().augmentationConfig();
    boolean horizFlip = ac.horizontalFlip() && random().nextBoolean();

    IPoint sourceImageSize = modelHandler().model().inputImagePlanarSize();
    Matrix tfmTranslateToCenter = Matrix.getTranslate(sourceImageSize.x * -.5f, sourceImageSize.y * -.5f);
    Matrix tfmTranslateFromCenter = Matrix.getTranslate(sourceImageSize.x * .5f, sourceImageSize.y * .5f);

    Matrix tfmShear = Matrix.IDENTITY;
    if (!ac.shearDisable()) {
      float sh = ac.shearMax();
      float shear = random(-sh, sh);
      if (random().nextBoolean()) {
        tfmShear = new Matrix(1, 0, shear, 1, 0, 0);
      } else {
        tfmShear = new Matrix(1, shear, 0, 1, 0, 0);
      }
    }

    Matrix tfmScale = Matrix.IDENTITY;
    if (!ac.scaleDisable()) {
      // Scale the horizontal and vertical axes independently
      float scaleMax = ac.scaleMax();
      float scaleMin = ac.scaleMin();
      if (scaleMin <= 0)
        scaleMin = scaleMax * 0.65f;
      float xScale = random(scaleMin, scaleMax);
      float yScale = random(scaleMin, scaleMax);
      if (horizFlip)
        xScale = -xScale;
      tfmScale = Matrix.getScale(xScale, yScale);
    }

    int rotateDegrees = 0;
    Matrix tfmRotate = Matrix.IDENTITY;
    if (!ac.rotateDisable()) {
      final float MAX_ROT_DEG = ac.rotateDegreesMax();
      float rotate = random(MyMath.M_DEG * -MAX_ROT_DEG, MyMath.M_DEG * MAX_ROT_DEG);
      tfmRotate = Matrix.getRotate(rotate);
      rotateDegrees = Math.round(rotate / MyMath.M_DEG);
    }

    Matrix tfmTranslate = Matrix.IDENTITY;
    if (!ac.translateDisable()) {
      float W = ac.translateMax();
      tfmTranslate = Matrix.getTranslate(random(-W, W), random(-W, W));
    }

    // Apply matrix multiplications in right-to-left order to get the effect we want

    // Note: we are sometimes doing an unnecessary multiply of the identity matrix, but
    // to keep things simple, don't bother optimizing that (yet)

    Matrix tfm = tfmTranslateToCenter //
        .pcat(tfmShear)//
        .pcat(tfmScale)//
        .pcat(tfmRotate)//
        .pcat(tfmTranslateFromCenter)//
        .pcat(tfmTranslate);

    return transformWrapper(tfm, rotateDegrees);
  }

  private float random(float min, float max) {
    checkArgument(max >= min);
    if (max == min)
      return min;
    return random().nextFloat() * (max - min) + min;
  }

  private final CompileImagesConfig mConfig;
  private final Random mRandom;
  private final ModelHandler mModelHandler;
  private final Files mFiles;
  private Inspector mInspector = Inspector.NULL_INSPECTOR;
  private List<ImageEntry> mEntries;
  private boolean mEntriesValidated;
}
