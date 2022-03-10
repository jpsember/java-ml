package ml;

import static js.base.Tools.*;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.List;
import java.util.Random;

import js.file.Files;
import js.base.BaseObject;
import js.geometry.IPoint;
import js.geometry.Matrix;
import js.geometry.MyMath;
import ml.ModelHandler;
import ml.ModelInputReceiver;
import ml.ModelWrapper;
import gen.AugmentationConfig;
import gen.NeuralNetwork;
import gen.TransformWrapper;
import js.graphics.gen.ScriptElementList;
import static ml.Util.*;

public final class ImageHandler extends BaseObject {

  public ImageHandler(NeuralNetwork network, AugmentationConfig augmentationConfig) {
    mNetwork = network;
    mHandler = ModelHandler.construct(mNetwork);
    mDestFloatPixels = new float[model().inputImageVolumeProduct()];
    mAugmentationConfig = augmentationConfig;
    mRandom = new Random(1965);
  }

  public ModelWrapper model() {
    return mHandler.model();
  }

  public ModelHandler modelHandler() {
    return mHandler;
  }

  /**
   * Generate annotated images for training or inference
   */
  public final void applyCompileImagePipeline(BufferedImage srcImage, ScriptElementList annotations,
      TransformWrapper aug, ImageTransformer<BufferedImage> imageTransformer, ModelInputReceiver receiver,
      ImageRecord sourceImage) {
    annotations = Util.transform(annotations, aug.inverse(), -aug.rotationDegrees()).build();
    imageTransformer.withPendingAnnotation(annotations);
    imageTransformer.transform(aug.inverse(), aug.matrix(), srcImage, mDestFloatPixels);
    receiver.accept(mDestFloatPixels, annotations);
  }

  public TransformWrapper buildAugmentTransform() {
    AugmentationConfig ac = mAugmentationConfig;
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
      float xScale = random(ac.scaleMin(), ac.scaleMax());
      float yScale = random(ac.scaleMin(), ac.scaleMax());
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

  /**
   * Examine source directory, and generate ImageRecords from suitable image
   * files found
   */
  public final List<ImageRecord> processInputDir(File inputDir) {
    log("processInputDir");
    List<ImageRecord> imageRecords = arrayList();

    List<File> sourceImageFiles = Util.getSourceImageFiles(inputDir);

    for (File imageFile : sourceImageFiles) {
      ImageRecord rec = new ImageRecord(this, imageFile);
      rec.readScriptIfExists();
      for (ImageRecord.Filter filter : mFilters) {
        filter.apply(rec);
        if (!rec.rejectionReason().isEmpty())
          break;
      }

      if (rec.rejectionReason().isEmpty()) {
        log("...accepting", rec.getLogDisplayFilename());
        imageRecords.add(rec);
      } else {
        log("...rejecting", rec.getLogDisplayFilename(), ";", rec.rejectionReason());
      }
    }
    checkState(!imageRecords.isEmpty(), "No (unfiltered) input data found within: " + inputDir
        + "; current dir: " + Files.currentDirectory());
    return imageRecords;
  }

  public final Random random() {
    return mRandom;
  }

  public ImageHandler withFilter(ImageRecord.Filter filter) {
    mFilters.add(filter);
    return this;
  }

  private float random(float min, float max) {
    checkArgument(max >= min);
    if (max == min)
      return min;
    return random().nextFloat() * (max - min) + min;
  }

  private final NeuralNetwork mNetwork;
  private final AugmentationConfig mAugmentationConfig;
  private Random mRandom;
  private float[] mDestFloatPixels;
  private List<ImageRecord.Filter> mFilters = arrayList();
  private ModelHandler mHandler;
}
