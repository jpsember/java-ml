package dev.gen;

import java.io.File;
import js.data.AbstractData;
import js.geometry.IPoint;
import js.json.JSMap;

public class GenerateImagesConfig implements AbstractData {

  public File targetDir() {
    return mTargetDir;
  }

  public int imageTotal() {
    return mImageTotal;
  }

  public IPoint imageSize() {
    return mImageSize;
  }

  public int seed() {
    return mSeed;
  }

  public String categories() {
    return mCategories;
  }

  public boolean writeUncompressed() {
    return mWriteUncompressed;
  }

  public boolean writeFloats() {
    return mWriteFloats;
  }

  public boolean monochrome() {
    return mMonochrome;
  }

  public boolean mergeImages() {
    return mMergeImages;
  }

  public float translateFactor() {
    return mTranslateFactor;
  }

  public float scaleFactorMin() {
    return mScaleFactorMin;
  }

  public float scaleFactorMax() {
    return mScaleFactorMax;
  }

  public int rotFactor() {
    return mRotFactor;
  }

  @Override
  public Builder toBuilder() {
    return new Builder(this);
  }

  public static final String TARGET_DIR = "target_dir";
  public static final String IMAGE_TOTAL = "image_total";
  public static final String IMAGE_SIZE = "image_size";
  public static final String SEED = "seed";
  public static final String CATEGORIES = "categories";
  public static final String WRITE_UNCOMPRESSED = "write_uncompressed";
  public static final String WRITE_FLOATS = "write_floats";
  public static final String MONOCHROME = "monochrome";
  public static final String MERGE_IMAGES = "merge_images";
  public static final String TRANSLATE_FACTOR = "translate_factor";
  public static final String SCALE_FACTOR_MIN = "scale_factor_min";
  public static final String SCALE_FACTOR_MAX = "scale_factor_max";
  public static final String ROT_FACTOR = "rot_factor";

  @Override
  public String toString() {
    return toJson().prettyPrint();
  }

  @Override
  public JSMap toJson() {
    JSMap m = new JSMap();
    m.put(TARGET_DIR, mTargetDir.toString());
    m.put(IMAGE_TOTAL, mImageTotal);
    m.put(IMAGE_SIZE, mImageSize.toJson());
    m.put(SEED, mSeed);
    m.put(CATEGORIES, mCategories);
    m.put(WRITE_UNCOMPRESSED, mWriteUncompressed);
    m.put(WRITE_FLOATS, mWriteFloats);
    m.put(MONOCHROME, mMonochrome);
    m.put(MERGE_IMAGES, mMergeImages);
    m.put(TRANSLATE_FACTOR, mTranslateFactor);
    m.put(SCALE_FACTOR_MIN, mScaleFactorMin);
    m.put(SCALE_FACTOR_MAX, mScaleFactorMax);
    m.put(ROT_FACTOR, mRotFactor);
    return m;
  }

  @Override
  public GenerateImagesConfig build() {
    return this;
  }

  @Override
  public GenerateImagesConfig parse(Object obj) {
    return new GenerateImagesConfig((JSMap) obj);
  }

  private GenerateImagesConfig(JSMap m) {
    {
      mTargetDir = DEF_TARGET_DIR;
      String x = m.opt(TARGET_DIR, (String) null);
      if (x != null) {
        mTargetDir = new File(x);
      }
    }
    mImageTotal = m.opt(IMAGE_TOTAL, 20);
    {
      mImageSize = DEF_IMAGE_SIZE;
      Object x = m.optUnsafe(IMAGE_SIZE);
      if (x != null) {
        mImageSize = IPoint.DEFAULT_INSTANCE.parse(x);
      }
    }
    mSeed = m.opt(SEED, 1965);
    mCategories = m.opt(CATEGORIES, "AB");
    mWriteUncompressed = m.opt(WRITE_UNCOMPRESSED, false);
    mWriteFloats = m.opt(WRITE_FLOATS, false);
    mMonochrome = m.opt(MONOCHROME, false);
    mMergeImages = m.opt(MERGE_IMAGES, false);
    mTranslateFactor = m.opt(TRANSLATE_FACTOR, 0.5f);
    mScaleFactorMin = m.opt(SCALE_FACTOR_MIN, 0.7f);
    mScaleFactorMax = m.opt(SCALE_FACTOR_MAX, 1.3f);
    mRotFactor = m.opt(ROT_FACTOR, 30);
  }

  public static Builder newBuilder() {
    return new Builder(DEFAULT_INSTANCE);
  }

  @Override
  public boolean equals(Object object) {
    if (this == object)
      return true;
    if (object == null || !(object instanceof GenerateImagesConfig))
      return false;
    GenerateImagesConfig other = (GenerateImagesConfig) object;
    if (other.hashCode() != hashCode())
      return false;
    if (!(mTargetDir.equals(other.mTargetDir)))
      return false;
    if (!(mImageTotal == other.mImageTotal))
      return false;
    if (!(mImageSize.equals(other.mImageSize)))
      return false;
    if (!(mSeed == other.mSeed))
      return false;
    if (!(mCategories.equals(other.mCategories)))
      return false;
    if (!(mWriteUncompressed == other.mWriteUncompressed))
      return false;
    if (!(mWriteFloats == other.mWriteFloats))
      return false;
    if (!(mMonochrome == other.mMonochrome))
      return false;
    if (!(mMergeImages == other.mMergeImages))
      return false;
    if (!(mTranslateFactor == other.mTranslateFactor))
      return false;
    if (!(mScaleFactorMin == other.mScaleFactorMin))
      return false;
    if (!(mScaleFactorMax == other.mScaleFactorMax))
      return false;
    if (!(mRotFactor == other.mRotFactor))
      return false;
    return true;
  }

  @Override
  public int hashCode() {
    int r = m__hashcode;
    if (r == 0) {
      r = 1;
      r = r * 37 + mTargetDir.hashCode();
      r = r * 37 + mImageTotal;
      r = r * 37 + mImageSize.hashCode();
      r = r * 37 + mSeed;
      r = r * 37 + mCategories.hashCode();
      r = r * 37 + (mWriteUncompressed ? 1 : 0);
      r = r * 37 + (mWriteFloats ? 1 : 0);
      r = r * 37 + (mMonochrome ? 1 : 0);
      r = r * 37 + (mMergeImages ? 1 : 0);
      r = r * 37 + (int)mTranslateFactor;
      r = r * 37 + (int)mScaleFactorMin;
      r = r * 37 + (int)mScaleFactorMax;
      r = r * 37 + mRotFactor;
      m__hashcode = r;
    }
    return r;
  }

  protected File mTargetDir;
  protected int mImageTotal;
  protected IPoint mImageSize;
  protected int mSeed;
  protected String mCategories;
  protected boolean mWriteUncompressed;
  protected boolean mWriteFloats;
  protected boolean mMonochrome;
  protected boolean mMergeImages;
  protected float mTranslateFactor;
  protected float mScaleFactorMin;
  protected float mScaleFactorMax;
  protected int mRotFactor;
  protected int m__hashcode;

  public static final class Builder extends GenerateImagesConfig {

    private Builder(GenerateImagesConfig m) {
      mTargetDir = m.mTargetDir;
      mImageTotal = m.mImageTotal;
      mImageSize = m.mImageSize;
      mSeed = m.mSeed;
      mCategories = m.mCategories;
      mWriteUncompressed = m.mWriteUncompressed;
      mWriteFloats = m.mWriteFloats;
      mMonochrome = m.mMonochrome;
      mMergeImages = m.mMergeImages;
      mTranslateFactor = m.mTranslateFactor;
      mScaleFactorMin = m.mScaleFactorMin;
      mScaleFactorMax = m.mScaleFactorMax;
      mRotFactor = m.mRotFactor;
    }

    @Override
    public Builder toBuilder() {
      return this;
    }

    @Override
    public int hashCode() {
      m__hashcode = 0;
      return super.hashCode();
    }

    @Override
    public GenerateImagesConfig build() {
      GenerateImagesConfig r = new GenerateImagesConfig();
      r.mTargetDir = mTargetDir;
      r.mImageTotal = mImageTotal;
      r.mImageSize = mImageSize;
      r.mSeed = mSeed;
      r.mCategories = mCategories;
      r.mWriteUncompressed = mWriteUncompressed;
      r.mWriteFloats = mWriteFloats;
      r.mMonochrome = mMonochrome;
      r.mMergeImages = mMergeImages;
      r.mTranslateFactor = mTranslateFactor;
      r.mScaleFactorMin = mScaleFactorMin;
      r.mScaleFactorMax = mScaleFactorMax;
      r.mRotFactor = mRotFactor;
      return r;
    }

    public Builder targetDir(File x) {
      mTargetDir = (x == null) ? DEF_TARGET_DIR : x;
      return this;
    }

    public Builder imageTotal(int x) {
      mImageTotal = x;
      return this;
    }

    public Builder imageSize(IPoint x) {
      mImageSize = (x == null) ? DEF_IMAGE_SIZE : x.build();
      return this;
    }

    public Builder seed(int x) {
      mSeed = x;
      return this;
    }

    public Builder categories(String x) {
      mCategories = (x == null) ? "AB" : x;
      return this;
    }

    public Builder writeUncompressed(boolean x) {
      mWriteUncompressed = x;
      return this;
    }

    public Builder writeFloats(boolean x) {
      mWriteFloats = x;
      return this;
    }

    public Builder monochrome(boolean x) {
      mMonochrome = x;
      return this;
    }

    public Builder mergeImages(boolean x) {
      mMergeImages = x;
      return this;
    }

    public Builder translateFactor(float x) {
      mTranslateFactor = x;
      return this;
    }

    public Builder scaleFactorMin(float x) {
      mScaleFactorMin = x;
      return this;
    }

    public Builder scaleFactorMax(float x) {
      mScaleFactorMax = x;
      return this;
    }

    public Builder rotFactor(int x) {
      mRotFactor = x;
      return this;
    }

  }

  private static final File DEF_TARGET_DIR = new File("source_images");
  private static final IPoint DEF_IMAGE_SIZE  = new IPoint(32, 48);

  public static final GenerateImagesConfig DEFAULT_INSTANCE = new GenerateImagesConfig();

  private GenerateImagesConfig() {
    mTargetDir = DEF_TARGET_DIR;
    mImageTotal = 20;
    mImageSize = DEF_IMAGE_SIZE;
    mSeed = 1965;
    mCategories = "AB";
    mTranslateFactor = 0.5f;
    mScaleFactorMin = 0.7f;
    mScaleFactorMax = 1.3f;
    mRotFactor = 30;
  }

}
