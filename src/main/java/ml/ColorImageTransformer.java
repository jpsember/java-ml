package ml;

import static js.base.Tools.*;

import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;

import js.geometry.Matrix;
import js.graphics.ImgUtil;
import gen.AugmentationConfig;

/**
 * An implementation of ImageTransformer to work with color inputs
 */
public final class ColorImageTransformer extends ImageTransformer<BufferedImage> {

  private static final int INSPECTION_IMAGE_TYPE = BufferedImage.TYPE_INT_RGB;

  @Override
  public void transform(Matrix sourceToDestTransform, Matrix destToSourceTransform, BufferedImage sourceImage,
      float[] destination) {
    loadTools();
    AugmentationConfig config = augmentationConfig();
    AffineTransformOp op = new AffineTransformOp(sourceToDestTransform.toAffineTransform(),
        AffineTransformOp.TYPE_BILINEAR);

    BufferedImage targetImage = ImgUtil.imageOfSameSize(sourceImage, INSPECTION_IMAGE_TYPE);
    op.filter(sourceImage, targetImage);

    inspector().create("tfm");
    applyPendingAnnotations();
    inspector().image(targetImage);

    ImgUtil.bufferedImageToFloat(targetImage, model().inputImageVolume().depth(), destination);
    if (config.adjustBrightness()) {
      Util.applyRandomBrightness(random(), destination, config.brightShiftMin(),
          config.brightShiftMax());
    }
  }

}
