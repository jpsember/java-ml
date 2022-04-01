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

      Script.Builder script = Script.newBuilder();
      List<ScriptElement> scripts = arrayList();

      for (int objIndex = 0; objIndex < totalObjects; objIndex++) {

        todo("try to choose object locations so the rects don't overlap");
        
        p.with(randomElement(paints()).toBuilder().font(fi.mFont, 1f));

        int category = random().nextInt(categoriesString.length());
        String text = categoriesString.substring(category, category + 1);

        FontMetrics m = fi.metrics(p.graphics());

        int mx = config().imageSize().x / 2;
        int my = config().imageSize().y / 2;

        float rangex = mx * config().translateFactor();
        float rangey = my * config().translateFactor();

        int charWidth = m.charWidth(categoriesString.charAt(0));
        int charHeight = m.getAscent();
        Matrix tfmFontOrigin = Matrix.getTranslate(-charWidth / 2, charHeight / 2);
        Matrix tfmImageCenter = Matrix.getTranslate(randGuassian(mx - rangex, mx + rangex),
            randGuassian(my - rangey, my + rangey));
        Matrix tfmRotate = Matrix.getRotate(
            randGuassian(-config().rotFactor() * MyMath.M_DEG, config().rotFactor() * MyMath.M_DEG));
        Matrix tfmScale = Matrix.getScale(randGuassian(config().scaleFactorMin(), config().scaleFactorMax()));

        Matrix objectTfm = Matrix.postMultiply(tfmImageCenter, tfmScale, tfmRotate);

        IPoint topLeft = IPoint.with(-charWidth / 2, -charHeight / 2);
        IPoint size = IPoint.with(charWidth, charHeight);
        IRect origRect = IRect.withLocAndSize(topLeft, size);

        IRect tfmRect = RectElement.applyTruncatedHeuristicTransform(origRect, objectTfm);

        RectElement rectElement = new RectElement(ElementProperties.newBuilder().category(category), tfmRect);
        scripts.add(rectElement);

        Matrix tfm = Matrix.postMultiply(objectTfm, tfmFontOrigin);
        p.graphics().setTransform(tfm.toAffineTransform());
        p.graphics().drawString(text, 0, 0);

      }

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
        mColors.add(Paint.newBuilder().color(sc[i], sc[i + 1], sc[i + 2]).build());
      }
    }
    return mColors;
  }

  private int[] sColors = { //
      74, 168, 50, //
      50, 107, 168, //
      168, 101, 50, //
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

  private List<FontInfo> fonts() {
    if (mFonts == null) {
      mFonts = arrayList();
      addFont("Dialog");
      addFont("DialogInput");
      addFont("Monospaced");
      addFont("Serif");
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
