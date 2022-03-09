package dev.gen;

import java.io.File;
import js.data.AbstractData;
import js.file.Files;
import js.geometry.IPoint;
import js.json.JSMap;

public class GenerateTrainingSetConfig implements AbstractData {

  public File sourceImagesDir() {
    return mSourceImagesDir;
  }

  public File sourceImageInfo() {
    return mSourceImageInfo;
  }

  public File targetDir() {
    return mTargetDir;
  }

  public IPoint imageSize() {
    return mImageSize;
  }

  public int maxImages() {
    return mMaxImages;
  }

  @Override
  public Builder toBuilder() {
    return new Builder(this);
  }

  public static final String SOURCE_IMAGES_DIR = "source_images_dir";
  public static final String SOURCE_IMAGE_INFO = "source_image_info";
  public static final String TARGET_DIR = "target_dir";
  public static final String IMAGE_SIZE = "image_size";
  public static final String MAX_IMAGES = "max_images";

  @Override
  public String toString() {
    return toJson().prettyPrint();
  }

  @Override
  public JSMap toJson() {
    JSMap m = new JSMap();
    m.put(SOURCE_IMAGES_DIR, mSourceImagesDir.toString());
    m.put(SOURCE_IMAGE_INFO, mSourceImageInfo.toString());
    m.put(TARGET_DIR, mTargetDir.toString());
    m.put(IMAGE_SIZE, mImageSize.toJson());
    m.put(MAX_IMAGES, mMaxImages);
    return m;
  }

  @Override
  public GenerateTrainingSetConfig build() {
    return this;
  }

  @Override
  public GenerateTrainingSetConfig parse(Object obj) {
    return new GenerateTrainingSetConfig((JSMap) obj);
  }

  private GenerateTrainingSetConfig(JSMap m) {
    {
      mSourceImagesDir = Files.DEFAULT;
      String x = m.opt(SOURCE_IMAGES_DIR, (String) null);
      if (x != null) {
        mSourceImagesDir = new File(x);
      }
    }
    {
      mSourceImageInfo = Files.DEFAULT;
      String x = m.opt(SOURCE_IMAGE_INFO, (String) null);
      if (x != null) {
        mSourceImageInfo = new File(x);
      }
    }
    {
      mTargetDir = DEF_TARGET_DIR;
      String x = m.opt(TARGET_DIR, (String) null);
      if (x != null) {
        mTargetDir = new File(x);
      }
    }
    {
      mImageSize = DEF_IMAGE_SIZE;
      Object x = m.optUnsafe(IMAGE_SIZE);
      if (x != null) {
        mImageSize = IPoint.DEFAULT_INSTANCE.parse(x);
      }
    }
    mMaxImages = m.opt(MAX_IMAGES, 0);
  }

  public static Builder newBuilder() {
    return new Builder(DEFAULT_INSTANCE);
  }

  @Override
  public boolean equals(Object object) {
    if (this == object)
      return true;
    if (object == null || !(object instanceof GenerateTrainingSetConfig))
      return false;
    GenerateTrainingSetConfig other = (GenerateTrainingSetConfig) object;
    if (other.hashCode() != hashCode())
      return false;
    if (!(mSourceImagesDir.equals(other.mSourceImagesDir)))
      return false;
    if (!(mSourceImageInfo.equals(other.mSourceImageInfo)))
      return false;
    if (!(mTargetDir.equals(other.mTargetDir)))
      return false;
    if (!(mImageSize.equals(other.mImageSize)))
      return false;
    if (!(mMaxImages == other.mMaxImages))
      return false;
    return true;
  }

  @Override
  public int hashCode() {
    int r = m__hashcode;
    if (r == 0) {
      r = 1;
      r = r * 37 + mSourceImagesDir.hashCode();
      r = r * 37 + mSourceImageInfo.hashCode();
      r = r * 37 + mTargetDir.hashCode();
      r = r * 37 + mImageSize.hashCode();
      r = r * 37 + mMaxImages;
      m__hashcode = r;
    }
    return r;
  }

  protected File mSourceImagesDir;
  protected File mSourceImageInfo;
  protected File mTargetDir;
  protected IPoint mImageSize;
  protected int mMaxImages;
  protected int m__hashcode;

  public static final class Builder extends GenerateTrainingSetConfig {

    private Builder(GenerateTrainingSetConfig m) {
      mSourceImagesDir = m.mSourceImagesDir;
      mSourceImageInfo = m.mSourceImageInfo;
      mTargetDir = m.mTargetDir;
      mImageSize = m.mImageSize;
      mMaxImages = m.mMaxImages;
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
    public GenerateTrainingSetConfig build() {
      GenerateTrainingSetConfig r = new GenerateTrainingSetConfig();
      r.mSourceImagesDir = mSourceImagesDir;
      r.mSourceImageInfo = mSourceImageInfo;
      r.mTargetDir = mTargetDir;
      r.mImageSize = mImageSize;
      r.mMaxImages = mMaxImages;
      return r;
    }

    public Builder sourceImagesDir(File x) {
      mSourceImagesDir = (x == null) ? Files.DEFAULT : x;
      return this;
    }

    public Builder sourceImageInfo(File x) {
      mSourceImageInfo = (x == null) ? Files.DEFAULT : x;
      return this;
    }

    public Builder targetDir(File x) {
      mTargetDir = (x == null) ? DEF_TARGET_DIR : x;
      return this;
    }

    public Builder imageSize(IPoint x) {
      mImageSize = (x == null) ? DEF_IMAGE_SIZE : x.build();
      return this;
    }

    public Builder maxImages(int x) {
      mMaxImages = x;
      return this;
    }

  }

  private static final File DEF_TARGET_DIR = new File("training_data");
  private static final IPoint DEF_IMAGE_SIZE  = new IPoint(320, 256);

  public static final GenerateTrainingSetConfig DEFAULT_INSTANCE = new GenerateTrainingSetConfig();

  private GenerateTrainingSetConfig() {
    mSourceImagesDir = Files.DEFAULT;
    mSourceImageInfo = Files.DEFAULT;
    mTargetDir = DEF_TARGET_DIR;
    mImageSize = DEF_IMAGE_SIZE;
  }

}
