package ml;

import static js.base.Tools.*;

import java.awt.Color;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics2D;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.Random;

import gen.AugmentationConfig;
import gen.Classifier;
import gen.GenerateImagesConfig;
import gen.NetworkProjectType;
import gen.Yolo;
import js.app.AppOper;
import js.file.Files;
import js.geometry.IPoint;
import js.geometry.IRect;
import js.geometry.Matrix;
import js.geometry.MyMath;
import js.graphics.ImgUtil;
import js.graphics.Paint;
import js.graphics.Plotter;
import js.graphics.RectElement;
import js.graphics.ScriptElement;
import js.graphics.ScriptUtil;
import js.graphics.gen.ElementProperties;
import js.graphics.gen.Script;

import static gen.SpecialOption.*;

public class GenerateImageSetOper extends AppOper {

  public static final boolean YOLO_DEV = false && alert("YOLO_DEV in effect");

  @Override
  public String userCommand() {
    return "genimages";
  }

  @Override
  public String getHelpDescription() {
    return "Generate annotated images procedurally";
  }

  @Override
  public void perform() {
    if (YOLO_DEV)
      die("temporarily disabled to work with known YOLO images");
    mModel = ModelWrapper.constructFor(config().network(), config().networkPath());

    switch (projectType()) {
    default:
      throw setError("unsupported project type", projectType());

    case YOLO: {
      Yolo yolo = (Yolo) model().modelConfig();
      checkArgument(yolo.categoryCount() == config().categories().length(), "Yolo category count",
          yolo.categoryCount(), "disagrees with categories string length", config().categories());
    }
      break;

    case CLASSIFIER: {
      Classifier c = (Classifier) model().modelConfig();
      checkArgument(c.categoryCount() == config().categories().length(), "Classifier category count",
          c.categoryCount(), "disagrees with categories string length", config().categories());
    }
      break;

    }

    ModelWrapper model = model();
    mImageSize = model.inputImagePlanarSize();
    if (model.isSpecial(OBVIOUS)) {
      checkArgument(projectType() == NetworkProjectType.CLASSIFIER,
          "obvious mode only supported in CLASSIFIER");
      checkArgument(config().categories().length() <= Plotter.rgbColorList().size(),
          "obvious mode doesn't support that many categories");
    }

    File targetDir = files().remakeDirs(config().targetDir());
    File annotationDir = files().mkdirs(ScriptUtil.scriptDirForProject(targetDir));

    for (int i = 0; i < config().imageTotal(); i++) {

      Plotter p = Plotter.build();
      p.withCanvas(mImageSize);
      Script.Builder script = Script.newBuilder();
      String imageBaseName = String.format("image_%05d", i);
      generateImage(p, script, imageBaseName);

      {
        String path = Files.setExtension(imageBaseName, ImgUtil.EXT_JPEG);
        File f = new File(targetDir, path);
        ImgUtil.writeImage(files(), p.image(), f);
      }
      {
        String path = Files.setExtension(imageBaseName, Files.EXT_JSON);
        File f = new File(annotationDir, path);
        ScriptUtil.write(files(), script, f);
      }
    }
  }

  private void generateImage(Plotter p, Script.Builder script, String imageBaseName) {
    ModelWrapper model = model();

    FontInfo fi = randomElement(fonts(), config().fontLimit());
    p.with(PAINT_BGND).fillRect();
    AffineTransform origTransform = p.graphics().getTransform();

    plotBgndImage(p);
    plotNoise(p);

    List<IRect> rectList = arrayList();

    if (model.inputImageChannels() == 1)
      alert(
          "monochrome isn't necessarily supported yet; clients should just use single channel of generated images, e.g. green");

    Integer firstCat = null;

    int totalObjects = 1;
    if (projectType() != NetworkProjectType.CLASSIFIER)
      totalObjects = 1 + random().nextInt(config().maxObjects());
    List<ScriptElement> scriptElements = arrayList();

    for (int objIndex = 0; objIndex < totalObjects; objIndex++) {
      p.with(randomElement(paints(), config().colorLimit()).toBuilder().font(fi.mFont, 1f));
      String categoriesString = config().categories();
      int category = random().nextInt(categoriesString.length());
      if (firstCat == null)
        firstCat = category;
      String text = categoriesString.substring(category, category + 1);

      FontMetrics m = fi.metrics(p.graphics());

      Matrix objectTfm = null;
      Matrix tfmFontOrigin = null;
      IRect tfmRect = null;

      boolean choseValidRectangle = false;

      AugmentationConfig aug = config().augmentationConfig();

      for (int attempt = 0; attempt < 5; attempt++) {

        // We may want to avoid placing things right in the center if we're restricting
        // the transformations, since this might be right on the border of a grid cell...
        int mx = mImageSize.x / 2;
        int my = mImageSize.y / 2;

        int charWidth = m.charWidth(categoriesString.charAt(0));
        int charHeight = (int) (m.getAscent() * ASCENT_SCALE_FACTOR);

        // This is the offset in the y coordinate to apply when actually rendering the character
        // using Java, so the render location is in terms of the baseline (not our center of the character)j
        IPoint fontRenderOffset = IPoint.with(-charWidth / 2, charHeight / 2);
        tfmFontOrigin = Matrix.getTranslate(fontRenderOffset);

        Matrix tfmImageCenter;
        if (aug.translateDisable())
          tfmImageCenter = Matrix.getTranslate(mx, my);
        else {
          float rangex = mImageSize.x * aug.translateRatioMax();
          float rangey = mImageSize.y * aug.translateRatioMax();
          tfmImageCenter = Matrix.getTranslate(randGuassian(mx - rangex, mx + rangex),
              randGuassian(my - rangey, my + rangey));
        }

        Matrix tfmRotate;
        if (aug.rotateDisable())
          tfmRotate = Matrix.IDENTITY;
        else {
          float maxRad = aug.rotateDegreesMax() * MyMath.M_DEG;
          tfmRotate = Matrix.getRotate(randGuassian(-maxRad, maxRad));
        }

        Matrix tfmScale;
        if (aug.scaleDisable()) {
          tfmScale = Matrix.getScale(aug.scaleMax());
        } else {
          float scaleMax = aug.scaleMax();
          float scaleMin = aug.scaleMin();
          if (scaleMin <= 0)
            scaleMin = scaleMax * 0.65f;
          tfmScale = Matrix.getScale(randGuassian(scaleMin, scaleMax));
        }

        objectTfm = Matrix.postMultiply(tfmImageCenter, tfmScale, tfmRotate);

        IPoint topLeft = IPoint.with(-charWidth / 2, -charHeight / 2);
        IPoint size = IPoint.with(charWidth, charHeight);
        IRect origRect = IRect.withLocAndSize(topLeft, size);

        tfmRect = RectElement.applyTruncatedHeuristicTransform(origRect, objectTfm);

        // If this rectangle overlaps too much with a previous one, keep searching
        choseValidRectangle = true;

        for (IRect prevRect : rectList) {
          IRect intersection = IRect.intersection(prevRect, tfmRect);
          if (intersection == null)
            continue;
          float intersectionFactor = intersection.area() / (float) tfmRect.area();
          if (intersectionFactor < 0.25f)
            continue;
          choseValidRectangle = false;
          break;
        }
      }
      if (!choseValidRectangle)
        continue;

      RectElement rectElement = new RectElement(ElementProperties.newBuilder().category(category), tfmRect);
      rectList.add(tfmRect);
      scriptElements.add(rectElement);

      Matrix tfm = Matrix.postMultiply(objectTfm, tfmFontOrigin);
      p.graphics().setTransform(tfm.toAffineTransform());
      p.graphics().drawString(text, 0, 0);
    }

    // If we're doing a classifier, append the class number to the filename
    if (projectType() == NetworkProjectType.CLASSIFIER) {
      imageBaseName += String.format("_%d", first(scriptElements).properties().category());
    }

    script.items(scriptElements);

    p.graphics().setTransform(origTransform);

    switch (model.network().specialOption()) {
    default:
      break;
    case OBVIOUS:
      p.graphics().setColor(Plotter.rgbColorList().get(firstCat));
      p.fillRect();
      break;

    case BLUE:
      p.graphics().setColor(Color.blue);
      p.fillRect();
      break;

    case PIXEL_ALIGNMENT:
      renderPixelOrderModeImage(p);
      break;
    }

  }

  private void renderPixelOrderModeImage(Plotter p) {
    int[] pixels = ImgUtil.rgbPixels(p.image());
    ModelWrapper model = model();
    IPoint imgSize = model.inputImagePlanarSize();

    if (model.inputImageChannels() != 3)
      throw notSupported("expected 3 channels");

    int pi = 0;
    for (int y = 0; y < imgSize.y; y++) {
      for (int x = 0; x < imgSize.x; x++, pi++) {
        int xyValue = y * 7 + x * 13;
        int rv = xyValue + 1;
        int gv = xyValue + 2;
        int bv = xyValue + 3;
        // RGB pixels have format:
        //       24       16       8        0
        //  unused |  red    | blue  |  green
        //
        // We want the red component to lie in channel 0, green in 1, and blue in 2:
        //
        pixels[pi] = ImgUtil.compileRGB((byte) bv, (byte) gv, (byte) rv);
      }
    }
  }

  private IPoint rndPoint() {
    return IPoint.with(random().nextInt(mImageSize.x), random().nextInt(mImageSize.y));
  }

  private void plotBgndImage(Plotter p) {
    String imageName = config().bgndImage();
    if (nullOrEmpty(imageName))
      return;
    BufferedImage bgndImage = bgndImage(imageName);
    IPoint imgSize = ImgUtil.size(bgndImage);
    IPoint slack = IPoint.difference(imgSize, mImageSize);
    checkArgument(Math.min(slack.x, slack.y) > 0, "image isn't big enough");
    IPoint originWithinBgndImage = new IPoint(random().nextFloat() * slack.x, random().nextFloat() * slack.y).negate();
    p.graphics().drawImage(bgndImage, originWithinBgndImage.x, originWithinBgndImage.y, null);
  }

  private void plotNoise(Plotter p) {
    int nf = config().noiseFactor();
    if (nf <= 0)
      return;

    IPoint loc = rndPoint();
    int k = random().nextInt(nf);

    for (int i = 0; i < k; i++) {
      IPoint loc2 = rndPoint();
      p.with(randomElement(paints(), config().colorLimit()));
      p.graphics().drawLine(loc.x, loc.y, loc2.x, loc2.y);
      loc = loc2;
    }

  }

  @Override
  public GenerateImagesConfig defaultArgs() {
    return GenerateImagesConfig.DEFAULT_INSTANCE;
  }

  @Override
  public GenerateImagesConfig config() {
    return super.config();
  }

  // ------------------------------------------------------------------
  // Paints
  // ------------------------------------------------------------------

  private List<Paint> paints() {
    if (mColors == null) {
      int[] sc = sColors;
      if (model().inputImageChannels() == 1)
        sc = sColorsMono;
      mColors = arrayList();
      for (int i = 0; i < sc.length; i += 3) {
        mColors.add(Paint.newBuilder().width(1f).color(sc[i], sc[i + 1], sc[i + 2]).build());
      }
    }
    return mColors;
  }

  private int[] sColors = { //
      74, 168, 50, //
      50, 107, 168, //
      168, 101, 50, //
      25, 25, 25, //
      127, 3, 252,//
  };
  private int[] sColorsMono = { //
      100, 100, 100, //
      80, 80, 80, //
      40, 40, 40, //
      20, 20, 20, //
  };

  private Paint PAINT_BGND = Paint.newBuilder().color(220, 220, 220).build();

  // ------------------------------------------------------------------
  // Fonts
  // ------------------------------------------------------------------

  private static final float ASCENT_SCALE_FACTOR = 0.85f;

  private List<FontInfo> fonts() {
    if (mFonts == null) {
      mFonts = arrayList();
      addFont("Dialog");
      addFont("DialogInput");
      addFont("Monospaced");
      addFont("SansSerif");
    }
    return mFonts;
  }

  private void addFont(String family) {
    FontInfo fi = new FontInfo();
    fi.mFont = new Font(family, Font.PLAIN, 12);
    fonts().add(fi);

    fi = new FontInfo();
    fi.mFont = new Font(family, Font.BOLD, 12);
    fonts().add(fi);
  }

  private static class FontInfo {
    Font mFont;
    FontMetrics mMetrics;

    public FontMetrics metrics() {
      checkState(mMetrics != null);
      return mMetrics;
    }

    public FontMetrics metrics(Graphics2D g) {
      if (mMetrics == null)
        mMetrics = g.getFontMetrics(mFont);
      return metrics();
    }
  }

  // ------------------------------------------------------------------
  // Utilities
  // ------------------------------------------------------------------

  private Random random() {
    if (mRandom == null)
      mRandom = new Random(config().seed());
    return mRandom;
  }

  private float randGuassian(float min, float max) {
    if (min == max)
      return max;
    if (min > max)
      badArg("min > max");
    float scl = (max - min) * 0.35f;
    float center = (max + min) * 0.5f;
    while (true) {
      float g = (float) (random().nextGaussian() * scl + center);
      if (g >= min && g <= max)
        return g;
    }
  }

  private <T> T randomElement(List<T> elements, int limitSize) {
    if (limitSize > 0)
      elements = elements.subList(0, Math.min(elements.size(), limitSize));
    return elements.get(random().nextInt(elements.size()));
  }

  private NetworkProjectType projectType() {
    return model().projectType();
  }

  private ModelWrapper model() {
    return mModel;
  }

  private BufferedImage bgndImage(String name) {
    BufferedImage img = mImageMap.get(name);
    if (img == null) {
      img = ImgUtil.read(Files.openResource(getClass(), name));
      mImageMap.put(name, img);
    }
    return img;
  }

  private Map<String, BufferedImage> mImageMap = hashMap();

  private ModelWrapper mModel;
  private IPoint mImageSize;
  private Random mRandom;
  private List<FontInfo> mFonts;
  private List<Paint> mColors;

}
