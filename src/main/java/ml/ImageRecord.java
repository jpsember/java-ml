package ml;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.List;

import js.file.Files;
import js.geometry.IPoint;
import js.geometry.IRect;
import js.graphics.PolygonElement;
import js.graphics.RectElement;
import js.graphics.ScriptElement;
import js.graphics.ScriptUtil;
import js.graphics.gen.Script;
import js.graphics.gen.ScriptElementList;

import static js.base.Tools.*;

import js.graphics.ImageFit;
import js.graphics.ImgUtil;

/**
 * Represents a single image, e.g., for compilation into a training set
 */
public final class ImageRecord {

  public ImageRecord(ImageHandler processor, File sourceFile) {
    mImageProcessor = processor;
    mSourceImageFile = sourceFile;
  }

  public ImageHandler processor() {
    return mImageProcessor;
  }

  /**
   * Discard image resource to free up memory pressure
   */
  public void discardImage() {
    mScriptPath = null;
    mSourceBufferedImage = null;
  }

  /**
   * Convert a BufferedImage to a BufferedImage of our preferred type (or return
   * input image if no conversion is necessary)
   */
  private BufferedImage toIntermediateImageType(BufferedImage sourceImage) {
    int targetType = BufferedImage.TYPE_INT_RGB;
    BufferedImage intermediateImage = ImgUtil.imageAsType(sourceImage, targetType);
    return intermediateImage;
  }

  public File sourceImageFile() {
    return mSourceImageFile;
  }

  public void setAnnotations(ScriptElementList annotation) {
    mAnnotations = annotation;
  }

  private boolean hasPolygon() {
    return ScriptUtil.elementCount(annotations().elements(), PolygonElement.DEFAULT_INSTANCE) != 0;
  }

  private boolean hasBox() {
    return ScriptUtil.elementCount(annotations().elements(), RectElement.DEFAULT_INSTANCE) != 0;
  }

  public ScriptElementList annotations() {
    if (mAnnotations == ScriptElementList.DEFAULT_INSTANCE && !alert("option to omit this check"))
      throw die("probably unexpected: no annotation provided");
    return mAnnotations;
  }

  /**
   * Get the image in its intermediate (i.e. before being converted to a model
   * input) form.
   * 
   * If the source image is a .rax (or .raw) image, the intermediate form is
   * BufferedImage.TYPE_USHORT_GRAY, which is 16-bit unsigned grayscale.
   * 
   * Otherwise, it is BufferedImage.TYPE_RGB, with 8-bit unsigned R,G,B values.
   * Note that in this case, the intermediate form is a color image even if the
   * model calls for a single-channel (monochrome) image.
   */
  public BufferedImage bufferedImage() {
    if (mSourceBufferedImage == null) {
      BufferedImage image = ImgUtil.read(mSourceImageFile);
      if (image.getType() != BufferedImage.TYPE_USHORT_GRAY)
        image = toIntermediateImageType(image);
      mSourceBufferedImage = image;

      // Delete this old code if things seem to be working
      //
      //      String ext = getExtension(mSourceImageFile);
      //      if (ext.equals("raw") || ext.equals("rax")) {
      //        byte[] bytes = Files.toByteArray(mSourceImageFile);
      //        MonoImage monoImage = ImgUtil.decompressRAX(bytes, null);
      //        mSourceBufferedImage = MonoImageUtil.to16BitGrayscaleBufferedImage(monoImage);
      //      } else {
      //        BufferedImage image = ImgUtil.read(mSourceImageFile);
      //        mSourceBufferedImage = toIntermediateImageType(image);
      //      }
    }
    return mSourceBufferedImage;
  }

  public IPoint imageSize() {
    return ImgUtil.size(bufferedImage());
  }

  public void setRejectionReason(String message) {
    if (!mRejectionReason.isEmpty())
      return;
    mRejectionReason = ifNullOrEmpty(message, "(no reason given)");
  }

  public String rejectionReason() {
    return mRejectionReason;
  }

  public File scriptFile() {
    if (mScriptPath == null) {
      mScriptPath = ScriptUtil.scriptPathForImage(sourceImageFile());
    }
    return mScriptPath;
  }

  /**
   * Read script, if one exists; otherwise, return default
   */
  private Script script() {
    if (mScript == null) {
      mScript = Script.DEFAULT_INSTANCE;
      File file = scriptFile();
      if (file.exists()) {
        mScript = ScriptUtil.from(file);
      }
    }
    return mScript;
  }

  public interface Filter {
    void apply(ImageRecord record);
  }

  public static final Filter FILTER_STRANGE_BOX = (rec) -> {
    for (ScriptElement p : rec.shapes()) {
      IRect box = p.bounds();
      if (box.width < 100 || box.height < 100 || box.aspectRatio() < 0.8 || box.aspectRatio() > 1.2) {
        rec.setRejectionReason("...box has small size: " + Math.min(box.width, box.height)
            + " or strange aspect ratio: " + box.aspectRatio());
        break;
      }
    }
  };

  public void readScriptIfExists() {
    Script script = script();
    if (isFalse(script.omit())) {
      try {
        processor().modelHandler().extractShapes(script, shapes());
      } catch (Throwable t) {
        pr("Problem handling script:", scriptFile());
        throw t;
      }
    }
    setAnnotations(Util.compileAnnotation(shapes()));
  }

  /**
   * If no script exists, rejects.
   */
  public static final Filter FILTER_SCRIPT_REQUIRED = (rec) -> {
    if (rec.script().equals(Script.DEFAULT_INSTANCE)) {
      rec.setRejectionReason("FILTER_SCRIPT_REQUIRED");
      return;
    }
  };

  /**
   * Reject records that have no box defined
   */
  public static final Filter FILTER_BOX_REQUIRED = (rec) -> {
    int validBoxCount = 0;
    for (ScriptElement p : rec.shapes())
      if (p.bounds().isValid())
        validBoxCount++;
    if (validBoxCount == 0 || validBoxCount != rec.shapes().size())
      rec.setRejectionReason("BOX_REQUIRED");
  };

  /**
   * Reject records that have no box defined, unless the 'retain' flag is set
   */
  public static final Filter FILTER_BOX_OR_RETAIN_REQUIRED = (rec) -> {
    if (rec.script().retain())
      return;

    for (ScriptElement po : rec.shapes()) {
      if (po.bounds().isDegenerate()) {
        rec.setRejectionReason("BOX_OR_RETAIN_REQUIRED");
        break;
      }
    }

  };

  /**
   * Reject records that have no polygon defined
   */
  public static final Filter FILTER_POLYGON_REQUIRED = (rec) -> {
    if (!rec.hasPolygon())
      rec.setRejectionReason("POLYGON_REQUIRED");
  };

  /**
   * Reject records that have no polygon or box defined, unless the 'retain'
   * flag is set
   */
  public static final Filter FILTER_SHAPE_OR_RETAIN_REQUIRED = (rec) -> {
    if (isFalse(rec.script().retain()) && !rec.hasPolygon() && !rec.hasBox())
      rec.setRejectionReason("SHAPE_OR_RETAIN_REQUIRED");
  };

  /**
   * Reject records that have boxes (i.e. polygons are required instead)
   */
  public static final Filter FILTER_NO_BOXES = (rec) -> {
    if (rec.hasBox())
      rec.setRejectionReason("NO_BOXES");
  };

  /**
   * Reject records that don't have exactly one polygon
   */
  public static final Filter FILTER_SINGLE_POLYGON_REQUIRED = (rec) -> {
    int numPolygons = ScriptUtil.elementCount(rec.annotations().elements(), PolygonElement.DEFAULT_INSTANCE);
    if (numPolygons != 1)
      rec.setRejectionReason("Polygon count= " + numPolygons);
  };

  /**
   * Constructs filter that rejects if source image is not a particular size
   */
  public static final Filter filterEnsureImageSize(IPoint expectedSize) {
    return (rec) -> {
      // Read the image, if not already
      rec.bufferedImage();
      IPoint origSize = rec.processor().model().inputImagePlanarSize();
      if (!origSize.equals(expectedSize)) {
        alert("Encountered unexpected image size:", origSize);
        rec.setRejectionReason("Unexpected image size: " + origSize + "; expected " + expectedSize);
      }
    };
  }

  private static File fileRelativeToGrandparent(File file) {
    File parent = file.getParentFile();
    if (parent != null)
      parent = parent.getParentFile();
    if (parent == null)
      return file;
    return Files.fileWithinOptionalDirectory(file, parent);
  }

  public File getLogDisplayFilename() {
    if (mLogDisplayFilename == null)
      mLogDisplayFilename = fileRelativeToGrandparent(sourceImageFile());
    return mLogDisplayFilename;
  }

  public List<AugmentTransform> transforms() {
    return mTransforms;
  }

  public List<ScriptElement> shapes() {
    return mShapes;
  }

  /**
   * Construct a transformed version of this record, by applying an ImageFit
   * operation
   */
  public ImageRecord constructTransformedVersion(ImageFit imageFit) {
    BufferedImage sourceImage = bufferedImage();
    BufferedImage croppedImage = null;
    IRect destRect = imageFit.transformedSourceRect();
    checkState(destRect.isValid());
    int targetType = BufferedImage.TYPE_3BYTE_BGR;
    // If jpeg, make sure it is compatible with one of the types we know how to write
    switch (sourceImage.getType()) {
    default:
      throw die("unsupported image type:", ImgUtil.toJson(sourceImage), "file:", sourceImageFile());
    case BufferedImage.TYPE_INT_RGB:
    case BufferedImage.TYPE_3BYTE_BGR:
      targetType = sourceImage.getType();
      break;
    }
    croppedImage = imageFit.apply(sourceImage, targetType);
    ImageRecord outputRecord = new ImageRecord(mImageProcessor, sourceImageFile());
    outputRecord.setAnnotations(Util.transform(mAnnotations, imageFit.matrix(), 0));
    outputRecord.mSourceBufferedImage = croppedImage;
    return outputRecord;
  }

  private final File mSourceImageFile;
  private final ImageHandler mImageProcessor;
  private final List<AugmentTransform> mTransforms = arrayList();
  private final List<ScriptElement> mShapes = arrayList();
  private BufferedImage mSourceBufferedImage;
  private String mRejectionReason = "";
  private Script mScript;
  private File mLogDisplayFilename;
  private ScriptElementList mAnnotations = ScriptElementList.DEFAULT_INSTANCE;
  private File mScriptPath;

}
