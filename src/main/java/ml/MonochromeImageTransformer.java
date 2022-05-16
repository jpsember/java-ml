package ml;

import java.awt.Font;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.awt.image.BufferedImageOp;
import java.awt.image.ConvolveOp;
import java.awt.image.Kernel;
import java.util.List;

import static js.base.Tools.*;

import js.geometry.IPoint;
import js.geometry.Matrix;
import js.geometry.MyMath;
import js.graphics.ImgUtil;
import js.graphics.MonoImageUtil;
import js.graphics.gen.ImageStats;
import js.graphics.gen.MonoImage;
import js.json.JSList;
import js.json.JSMap;
import gen.AugmentationConfig;

/**
 * An implementation of ImageTransformer to work with 16-bit grayscale
 * BufferedImage inputs
 */
public class MonochromeImageTransformer extends ImageTransformer<BufferedImage> {

  // ------------------------------------------------------------------
  // Construction and initialization
  // ------------------------------------------------------------------

  private void prepare() {
    if (mPrepared)
      return;

    // Calculate the scale and offset to apply to 16-bit monochrome pixel values
    // to convert to floats

    mM = 1.0f / 0xffff;
    mB = 0;
    checkArgument(mM != 0f, "scale is zero");

    mPrepared = true;
  }

  // ------------------------------------------------------------------

  @Override
  public void transform(Matrix sourceToDestTransform, Matrix destToSourceTransform, BufferedImage sourceImage,
      float[] destinationOrNull) {

    inspector().create("source").normalize().sharpen().image(sourceImage);

    IPoint imageSize = ImgUtil.size(sourceImage);
    mDestination = destinationOrNull;
    if (mDestination == null)
      mDestination = new float[imageSize.product()];
    prepare();

    createTargetImage();
    transformSourceToTarget(sourceImage, sourceToDestTransform);

    applyBlurFactor();

    transformPixelsToFloat();

    adjustBrightness();

    applyPendingAnnotations();
    inspector().create("tfm");
    applyPendingAnnotations();
    inspector().imageSize(imageSize).image(mDestination);
    if (inspector().used()) {
      // Construct a monoimage from the float pixels
      short[] shortPixels = transformFloatPixelsToShort();
      MonoImage monoImage = MonoImageUtil.construct(imageSize, shortPixels);
      ImageStats stats = MonoImageUtil.generateStats(monoImage);
      JSMap m = map();
      m.put("cdf_2", stats.cdf()[2]);
      m.put("cdf_98", stats.cdf()[98]);
      m.put("mean", stats.mean());
      inspector().create("stats").json(m);
    }
  }

  private void createTargetImage() {
    mTargetImage = ImgUtil.build16BitGrayscaleImage(model().inputImagePlanarSize());
  }

  private static List<Font> sLabelFonts;
  static {
    sLabelFonts = arrayList();
    String[] names = { "TimesRoman", "Serif", "Helvetica", "SansSerif", "Courier", "Monospaced", "Dialog" };
    for (String name : names) {
      for (int sz = 0; sz < 4; sz++) {
        int fontSize = 22 + sz * 9;
        sLabelFonts.add(new Font(name, Font.PLAIN, fontSize));
        sLabelFonts.add(new Font(name, Font.BOLD, fontSize));
      }
    }
  }

  private void transformSourceToTarget(BufferedImage sourceImage, Matrix sourceToDestTransform) {
    AffineTransformOp op = new AffineTransformOp(sourceToDestTransform.toAffineTransform(),
        AffineTransformOp.TYPE_BILINEAR);
    op.filter(sourceImage, mTargetImage);
  }

  private void transformPixelsToFloat() {
    short[] pixels = ImgUtil.grayPixels(mTargetImage);
    float[] dest = mDestination;
    float m = mM;
    float b = mB;

    // Leave zero pixels as zero
    //
    // TODO: we can enhance this to produce a similar effect as our custom transformation, 
    // by scanning linearly until we hit a non-zero pixel, then copying that pixel value back along the path;
    // but we don't want to do that for the health project
    //

    for (int i = 0; i < pixels.length; i++) {
      short sourcePixel = pixels[i];
      float targetPixel = 0;
      if (sourcePixel != 0)
        targetPixel = m * sourcePixel + b;
      dest[i] = targetPixel;
    }
  }

  private short[] transformFloatPixelsToShort() {
    short[] pixels = new short[mDestination.length];

    float m = 1 / mM;
    float b = -mB;

    for (int i = 0; i < pixels.length; i++) {
      float sourcePixel = mDestination[i];
      if (sourcePixel != 0) {
        float tf = m * sourcePixel + b;
        if (tf < 0)
          tf = 0;
        else if (tf > 0x7fff)
          tf = 0x7fff;
        pixels[i] = (short) tf;
      }
    }
    return pixels;
  }

  private void adjustBrightness() {
    AugmentationConfig config = augmentationConfig();
    if (config.adjustBrightness()) {
      Util.applyRandomBrightness(random(), mDestination, config.brightShiftMin(), config.brightShiftMax());
    }
  }

  /* private */ void dumpPixelSample() {
    // TODO: make 'dumpPixelSample' a utility function somewhere
    StringBuilder sb = new StringBuilder();
    IPoint sz = model().inputImagePlanarSize();
    float scalex = 0.3f;
    float scaley = 0.1f;
    int y0 = (int) (sz.y * (.5f - scaley));
    int y1 = (int) (sz.y * (.5f + scaley));
    int x0 = (int) (sz.x * (.5f - scalex));
    int x1 = (int) (sz.x * (.5f + scalex));

    String pixels = " .:-=+*#%@";
    int stepx = 2;
    int stepy = 2;
    for (int y = y0; y < y1; y += stepy) {
      sb.append('[');
      for (int x = x0; x < x1; x += stepx) {
        float f = verifyFinite(mDestination[y * sz.x + x]);
        int j = Math.round(f * 10);
        j = MyMath.clamp(j, 0, 9);
        sb.append(pixels.charAt(j));
      }
      sb.append("]\n");
    }
    pr("pixels:", INDENT, sb);
  }

  private void applyBlurFactor() {
    AugmentationConfig config = augmentationConfig();
    int factor = 1 + config.blurFactor();
    if (factor == 0)
      return;

    checkArgument(factor <= sKernels.size());

    // We only apply blurring some of the time
    if (random().nextInt(2) != 0)
      return;

    // Choose random kernel from desired intensity
    int intensity = random().nextInt(factor);

    mTargetImage = applyBlurKernel(mTargetImage, intensity);
  }

  private BufferedImage applyBlurKernel(BufferedImage image, int kernelNumber) {
    Kernel kernel = sKernels.get(kernelNumber);
    BufferedImageOp op = new ConvolveOp(kernel);
    BufferedImage out = op.filter(image, null);
    return out;
  }

  // ------------------------------------------------------------------
  // Blurring kernels
  // ------------------------------------------------------------------

  private static final List<Kernel> sKernels;

  private static float[] calculateGuassianKernelCoefficients(int kernelSize, double sigma) {
    // TODO: this is giving different values than http://dev.theomader.com/gaussian-kernel-calculator/
    // but still look pretty reasonable

    int radius = kernelSize / 2;
    double sigmaSq = sigma * sigma;
    float[] terms = new float[kernelSize * kernelSize];

    double sum = 0;

    for (int i = 0; i < terms.length; i++) {
      int y = (i / kernelSize) - radius;
      int x = (i % kernelSize) - radius;

      double val = (1 / (2 * Math.PI * sigmaSq)) * Math.exp(-(x * x + y * y) / (2 * sigmaSq));

      terms[i] = (float) val;
      sum += val;
    }

    float scaleFactor = (float) (1 / sum);

    JSMap m = map();
    m.put("kernel size", kernelSize);
    m.put("radius", radius);
    m.put("sigma", sigma);

    JSList lst = null;
    for (int i = 0; i < kernelSize * kernelSize; i++) {
      if (i % kernelSize == 0) {
        lst = list();
        m.put("k" + (i / kernelSize), lst);
      }

      terms[i] *= scaleFactor;
      lst.add(String.format("%6.4f ", terms[i]));
    }
    return terms;
  }

  private static void genKernel(int radius, float sigma) {
    Kernel k = new Kernel(radius, radius, calculateGuassianKernelCoefficients(radius, sigma));
    sKernels.add(k);
  }

  static {
    sKernels = arrayList();

    genKernel(3, 1.0f);
    genKernel(5, 1.3f);
    genKernel(7, 1.8f);
  }

  // ------------------------------------------------------------------

  private float mB;
  private float mM;
  private boolean mPrepared;
  private float[] mDestination;
  private BufferedImage mTargetImage;

}
