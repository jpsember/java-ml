package ml.img;

import static js.base.Tools.*;

import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.io.DataOutputStream;
import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.Random;

import gen.AugmentationConfig;
import gen.CompileImagesConfig;
import gen.DataType;
import gen.LabelForm;
import gen.NeuralNetwork;
import gen.TransformWrapper;
import js.base.BaseObject;
import js.file.DirWalk;
import js.file.Files;
import js.geometry.IPoint;
import js.geometry.Matrix;
import js.geometry.MyMath;
import js.graphics.ImgUtil;
import js.graphics.Inspector;
import js.graphics.ScriptElement;
import js.graphics.ScriptUtil;
import ml.ModelWrapper;

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
    mModel = ModelWrapper.constructFor(network, null);
  }

  public void setInspector(Inspector inspector) {
    mInspector = Inspector.orNull(inspector);
  }

  public void compileTrainSet(File targetDir) {
    ModelWrapper model = model();
    files().remakeDirs(targetDir);
    File imagePath = new File(targetDir, "images.bin");
    File labelsPath = new File(targetDir, "labels.bin");
    File infoPath = new File(targetDir, "image_set_info.json");
    model.imageSetInfo().imageCount(entries().size());

    DataOutputStream imagesStream = new DataOutputStream(files().outputStream(imagePath));
    DataOutputStream labelsStream = new DataOutputStream(files().outputStream(labelsPath));

    model.setImageStream(imagesStream);
    model.setLabelStream(labelsStream);

    float[] imageFloats = null;

    DataType imageDataType = model.network().imageDataType();

    for (ImageEntry entry : entries()) {
      BufferedImage img = ImgUtil.read(entry.imageFile());
      // We don't need to validate the images except on the first pass through them
      if (!mEntriesValidated) {
        checkImageSizeAndType(entry.imageFile(), img, model.inputImagePlanarSize(),
            model.inputImageChannels());
      }
      mInspector.create("orig").image(img).elements(entry.scriptElementList());
      entry.setTransform(buildAugmentTransform());
      List<ScriptElement> annotations = entry.scriptElementList().elements();

      BufferedImage targetImage = ImgUtil.build(model.inputImagePlanarSize(), img.getType());
      AugmentationConfig config = config().augmentationConfig();
      AffineTransformOp op = new AffineTransformOp(entry.transform().matrix().toAffineTransform(),
          AffineTransformOp.TYPE_BILINEAR);
      {
        List<ScriptElement> transformed = arrayList();
        model().transformAnnotations(annotations, transformed, entry.transform());
        // We don't want to mistakenly use the untransformed elements from this point on...
        annotations = transformed;
      }
      op.filter(img, targetImage);
      mInspector.create("tfm").image(targetImage).elements(annotations);

      Object imagePixels = null;

      switch (imageDataType) {
      case FLOAT32: {
        imageFloats = ImgUtil.floatPixels(targetImage, model.inputImageChannels(), imageFloats);

        if (config.adjustBrightness())
          applyRandomBrightness(imageFloats, config.brightShiftMin(), config.brightShiftMax());

        mInspector.create("float").imageSize(model.inputImagePlanarSize())
            .channels(model.inputImageChannels()).image(imageFloats);
        imagePixels = imageFloats;
      }
        break;
      case UNSIGNED_BYTE: {
        if (config.adjustBrightness())
          notSupported("adjust_brightness is not supported for data type", imageDataType);
        checkArgument(model.inputImageChannels() == 3, "not supported yet for channels != 3");
        imagePixels = ImgUtil.bgrPixels(targetImage);
      }
        break;
      default:
        throw notSupported("ImageDataType:", imageDataType);
      }
      model.accept(imagePixels, annotations);

      if (mInspector.used()) {
        alert("inspector is used");
        // Parse the labels we generated, and write as the annotations to an inspection image
        mInspector.create("parsed").image(targetImage);
        // Script.Builder script = Script.newBuilder();
        List<ScriptElement> elements = (List<ScriptElement>) model.transformLabels(LabelForm.MODEL_INPUT,
            model.getLabelBuffer(), LabelForm.SCREDIT);
        mInspector.elements(elements);
      }

      entry.releaseResources();
    }
    mEntriesValidated = true;
    Files.close(imagesStream, labelsStream);
    files().writePretty(infoPath, model.imageSetInfo());
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

  private ModelWrapper model() {
    return mModel;
  }

  private TransformWrapper buildAugmentTransform() {
    AugmentationConfig ac = config().augmentationConfig();
    boolean horizFlip = ac.horizontalFlip() && random().nextBoolean();

    IPoint sourceImageSize = model().inputImagePlanarSize();
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
      float wx = ac.translateRatioMax() * sourceImageSize.x;
      float wy = ac.translateRatioMax() * sourceImageSize.y;
      tfmTranslate = Matrix.getTranslate(random(-wx, wx), random(-wy, wy));
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

  private void applyRandomBrightness(float[] pixels, float minShift, float maxShift) {
    float scale = 1 + random(minShift, maxShift);
    for (int i = 0; i < pixels.length; i++)
      pixels[i] = pixels[i] * scale;
  }

  private static TransformWrapper transformWrapper(Matrix matrix, int rotationDegrees) {
    TransformWrapper.Builder b = TransformWrapper.newBuilder();
    b.matrix(matrix);
    b.inverse(matrix.invert());
    b.rotationDegrees(rotationDegrees);
    return b.build();
  }

  private final CompileImagesConfig mConfig;
  private final Random mRandom;
  private final ModelWrapper mModel;
  private final Files mFiles;
  private Inspector mInspector = Inspector.NULL_INSPECTOR;
  private List<ImageEntry> mEntries;
  private boolean mEntriesValidated;
}
