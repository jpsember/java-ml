package ml;

import static js.base.Tools.*;

import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.util.Random;

import js.geometry.Matrix;
import js.graphics.ImgUtil;
import gen.AugmentationConfig;

/**
 * An implementation of ImageTransformer to work with color inputs
 */
public final class ColorImageTransformer extends ImageTransformer<BufferedImage> {

  private static final int INSPECTION_IMAGE_TYPE = BufferedImage.TYPE_INT_RGB;

  public ColorImageTransformer(ModelWrapper modelConfig, AugmentationConfig augConfig, Random random) {
    loadTools();
    mModelConfig = modelConfig;
    mAugConfig = augConfig;
    mRandom = random;
  }

  @Override
  public void transform(Matrix sourceToDestTransform, Matrix destToSourceTransform, BufferedImage sourceImage,
      float[] destination) {

    AffineTransformOp op = new AffineTransformOp(sourceToDestTransform.toAffineTransform(),
        AffineTransformOp.TYPE_BILINEAR);

    BufferedImage targetImage = ImgUtil.imageOfSameSize(sourceImage, INSPECTION_IMAGE_TYPE);
    op.filter(sourceImage, targetImage);

    inspector().create("tfm");
    applyPendingAnnotations();
    inspector().image(targetImage);

    ImgUtil.bufferedImageToFloat(targetImage, mModelConfig.inputImageVolume().depth(), destination);
    if (mAugConfig.adjustBrightness()) {
      Util.applyRandomBrightness(mRandom, destination, mAugConfig.brightShiftMin(),
          mAugConfig.brightShiftMax());
    }
  }

  private final ModelWrapper mModelConfig;
  private final AugmentationConfig mAugConfig;
  private final Random mRandom;
}
