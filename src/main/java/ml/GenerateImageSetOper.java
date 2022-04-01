package ml;

import static js.base.Tools.*;

import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics2D;
import java.io.File;
import java.util.List;
import java.util.Random;

import gen.GenerateImagesConfig;
import js.app.AppOper;
import js.file.Files;
import js.geometry.IPoint;
import js.geometry.IRect;
import js.geometry.Matrix;
import js.geometry.MyMath;
import js.graphics.ImgUtil;
import js.graphics.Inspector;
import js.graphics.Paint;
import js.graphics.Plotter;
import js.graphics.RectElement;
import js.graphics.ScriptElement;
import js.graphics.ScriptUtil;
import js.graphics.gen.ElementProperties;
import js.graphics.gen.Script;

public class GenerateImageSetOper extends AppOper {

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
    boolean detector = false;

    switch (config().type()) {
    case DETECTOR:
      detector = true;
      if (config().maxObjects() < 1)
        setError("max_objects bad", config());
      break;
    case CLASSIFIER:
    default:
      throw setError("unsupported type:", config().type());
    }

    File targetDir = files().remakeDirs(config().targetDir());
    File annotationDir = files().mkdirs(ScriptUtil.scriptDirForProject(targetDir));

    String categoriesString = config().categories();

    Inspector insp = Inspector.build(config().inspectionDir());
    insp.minSamples(5);

    for (int i = 0; i < config().imageTotal(); i++) {

      int totalObjects = 1;
      if (detector)
        totalObjects = 1 + random().nextInt(config().maxObjects());

      Plotter p = Plotter.build();
      p.withCanvas(config().imageSize());

      FontInfo fi = randomElement(fonts());
      p.with(PAINT_BGND).fillRect();

      plotNoise(p);

      Script.Builder script = Script.newBuilder();
      List<ScriptElement> scripts = arrayList();

      List<IRect> rectList = arrayList();

      for (int objIndex = 0; objIndex < totalObjects; objIndex++) {
        p.with(randomElement(paints()).toBuilder().font(fi.mFont, 1f));

        int category = random().nextInt(categoriesString.length());
        String text = categoriesString.substring(category, category + 1);

        FontMetrics m = fi.metrics(p.graphics());

        Matrix objectTfm = null;
        Matrix tfmFontOrigin = null;
        IRect tfmRect = null;

        boolean choseValidRectangle = false;

        for (int attempt = 0; attempt < 5; attempt++) {

          int mx = config().imageSize().x / 2;
          int my = config().imageSize().y / 2;

          float rangex = mx * config().translateFactor();
          float rangey = my * config().translateFactor();

          int charWidth = m.charWidth(categoriesString.charAt(0));
          int charHeight = (int) (m.getAscent() * ASCENT_SCALE_FACTOR);

          // This is the offset in the y coordinate to apply when actually rendering the character
          // using Java, so the render location is in terms of the baseline (not our center of the character)j
          IPoint fontRenderOffset = IPoint.with(-charWidth / 2, charHeight / 2);
          tfmFontOrigin = Matrix.getTranslate(fontRenderOffset);
          Matrix tfmImageCenter = Matrix.getTranslate(randGuassian(mx - rangex, mx + rangex),
              randGuassian(my - rangey, my + rangey));
          Matrix tfmRotate = Matrix.getRotate(
              randGuassian(-config().rotFactor() * MyMath.M_DEG, config().rotFactor() * MyMath.M_DEG));
          Matrix tfmScale = Matrix
              .getScale(randGuassian(config().scaleFactorMin(), config().scaleFactorMax()));

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
        scripts.add(rectElement);

        Matrix tfm = Matrix.postMultiply(objectTfm, tfmFontOrigin);
        p.graphics().setTransform(tfm.toAffineTransform());
        p.graphics().drawString(text, 0, 0);
      }
      
      plotNoise(p);

      if (insp.used()) {
        // TODO: inspector is kind of useless if we are writing out script projects anyways
        insp.create();
        insp.image(p.image());
      }

      String imageBaseName = String.format("image_%05d", i);
      {
        String path = Files.setExtension(imageBaseName, ImgUtil.EXT_JPEG);
        File f = new File(targetDir, path);
        if (config().monochrome())
          setError("Monochrome not supported yet");
        ImgUtil.writeImage(files(), p.image(), f);
      }

      script.items(scripts);
      {
        String path = Files.setExtension(imageBaseName, Files.EXT_JSON);
        File f = new File(annotationDir, path);
        ScriptUtil.write(files(), script, f);
      }
    }

    //    byte[] categoryBytes = DataUtil.intsToBytesLittleEndian(categories.array());
    //    files().write(categoryBytes, new File(config().targetDir(), "labels.bin"));
  }

  private IPoint rndPoint() {
    return IPoint.with(random().nextInt(config().imageSize().x), random().nextInt(config().imageSize().y));
  }

  private void plotNoise(Plotter p) {
    int nf = config().noiseFactor();
    if (nf <= 0)
      return;

    IPoint loc = rndPoint();
    int k = random().nextInt(nf);

    for (int i = 0; i < k; i++) {
      IPoint loc2 = rndPoint();
      p.with(randomElement(paints()));
      p.graphics().drawLine(loc.x, loc.y, loc2.x, loc2.y);
      loc = loc2;
    }

  }

  @Override
  public GenerateImagesConfig defaultArgs() {
    return GenerateImagesConfig.DEFAULT_INSTANCE;
  }

  @SuppressWarnings("unchecked")
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
      if (config().monochrome())
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
    float scl = (max - min) * 0.35f;
    float center = (max + min) * 0.5f;
    while (true) {
      float g = (float) (random().nextGaussian() * scl + center);
      if (g >= min && g <= max)
        return g;
    }
  }

  private <T> T randomElement(List<T> elements) {
    return elements.get(random().nextInt(elements.size()));
  }

  private Random mRandom;
  private List<FontInfo> mFonts;
  private List<Paint> mColors;

}
