package ml;

import static js.base.Tools.*;
import static ml.Util.*;

import java.awt.image.BufferedImage;
import java.io.DataOutputStream;
import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.Random;

import gen.AugmentationConfig;
import gen.CompileImagesConfig;
import gen.ImageSetInfo;
import gen.NeuralNetwork;
import gen.TransformWrapper;
//import gen.TransformWrapper;
import js.base.BaseObject;
import js.file.DirWalk;
import js.file.Files;
import js.geometry.IPoint;
import js.geometry.Matrix;
import js.geometry.MyMath;
import js.graphics.ImgEffects;
import js.graphics.ImgUtil;
import js.graphics.ScriptUtil;
import js.graphics.gen.Script;
import js.graphics.gen.ScriptElementList;

/**
 * Used by CompileImagesOper to process images
 */
public class ImageCompiler extends BaseObject {

  public ImageCompiler(CompileImagesConfig config, NeuralNetwork network, Files files) {
    mConfig = nullTo(config, CompileImagesConfig.DEFAULT_INSTANCE).build();
    mFiles = nullTo(files, Files.S);
    int seed = config().seed();
    if (seed <= 0)
      seed = 1965;
    mRandom = new Random(seed);
    mModelHandler = ModelHandler.construct(network);
    mImageTransformer = mModelHandler.buildImageTransformer(config.augmentationConfig(), random());
  }

  public void compileTrainSet(File targetDir) {
    files().remakeDirs(targetDir);
    File imagePath = new File(targetDir, "images.bin");
    File labelsPath = new File(targetDir, "labels.bin");
    File infoPath = new File(targetDir, "image_set_info.json");
    ImageSetInfo.Builder imageSetInfo = ImageSetInfo.newBuilder();
    imageSetInfo.imageCount(imageEntries().size());

    DataOutputStream imagesStream = new DataOutputStream(files().outputStream(imagePath));
    DataOutputStream labelsStream = new DataOutputStream(files().outputStream(labelsPath));

    ModelWrapper model = modelHandler().model();

    ModelServiceProvider provider = buildModelServiceProvider();
    provider.setImageStream(imagesStream);
    provider.setLabelStream(labelsStream);
    provider.storeImageSetInfo(imageSetInfo);
    if (imageSetInfo.imageLengthBytes() <= 0 || imageSetInfo.labelLengthBytes() <= 0)
      throw badState("ImageSetInfo hasn't been completely filled out:", INDENT, imageSetInfo);

    todo("!transform image randomly if training image");

    
    //    for (ImageEntry rec : entries) {
    //      AugmentTransform aug = mProc.buildAugmentTransform();
    //      ImageTransformer<BufferedImage> transformer = mHandler.buildImageTransformer(augmentationConfig(),
    //          random(), mTrainConfig.stats(), rec);
    //      transformer.setInspector(mInspectionManager);
    //      mProc.applyCompileImagePipeline(rec.bufferedImage(), rec.annotations(), aug, transformer,
    //          modelInputReceiver, rec);
    //      if (!cacheImagesInMemory)
    //        rec.discardImage();
    //    }
    //

 //   ImageHandler handler;
    
    for (ImageEntry entry : imageEntries()) {
      
      halt("Use ImageTransformer instead?");
          TransformWrapper transform =buildAugmentTransform();

      BufferedImage img = ImgUtil.read(entry.imageFile);
      
      
//      
//      mImageTransformer.transform(transform.matrix(),transform.inverse(), img, );
//      img = 
//      ImgEffects.applyTransform(img, transform.matrix().toAffineTransform());
//   
//      
      
      
      
      
      
      checkImageSizeAndType(entry.imageFile, img, model.inputImagePlanarSize(), model.inputImageChannels());
      mWorkArray = ImgUtil.floatPixels(img, model.inputImageChannels(), mWorkArray);
      
      
      
      
      
      
      
      
      
      
      
      
      provider.accept(mWorkArray, entry.scriptElements);
    }
    Files.close(imagesStream, labelsStream);

    files().writePretty(infoPath, imageSetInfo.build());
  }

  /**
   * Construct a ModelServiceProvider for the compiler's model type
   */
  public ModelServiceProvider buildModelServiceProvider() {
    ModelServiceProvider provider = modelHandler().buildModelServiceProvider();
    provider.setModel(modelHandler().model());
    return provider;
  }

  private List<ImageEntry> entries() {
    if (mEntries == null) {
      List<ImageEntry> ents = arrayList();
      File imageDir = Files.assertDirectoryExists(config().sourceDir());
      File scriptDir = ScriptUtil.scriptDirForProject(imageDir);
      Files.assertDirectoryExists(scriptDir, "script directory");
      DirWalk w = new DirWalk(imageDir).withRecurse(false).withExtensions(ImgUtil.EXT_JPEG);
      for (File f : w.files()) {
        ImageEntry ent = new ImageEntry();
        ent.imageFile = f;
        File scriptFile = ScriptUtil.scriptPathForImage(f);
        if (scriptFile.exists()) {
          Script script = ScriptUtil.from(scriptFile);
          ent.scriptElements = ScriptUtil.extractScriptElementList(script);
        }
        ents.add(ent);
      }

      int trainCount = ents.size();
      checkArgument(trainCount > 3, "insufficient images:", ents.size());
      MyMath.permute(ents, random());

      mEntries = ents;
    }
    return mEntries;
  }

  private List<ImageEntry> imageEntries() {
    entries();
    return mEntries;
  }

  private CompileImagesConfig config() {
    return mConfig;
  }

  private Files files() {
    return mFiles;
  }

  private Random random() {
    return mRandom;
  }

  private void checkImageSizeAndType(File imageFile, BufferedImage img, IPoint expectedImageSize,
      int expectedImageChannels) {
    IPoint imgSize = ImgUtil.size(img);
    if (mExpectedImageSize == null) {
      mExpectedImageSize = expectedImageSize;
      Integer channels = sImgChannelsMap.get(img.getType());
      if (channels == null)
        throw badArg("Unsupported image type:", INDENT, ImgUtil.toJson(img));
      if (channels != expectedImageChannels) {
        // Special case for using color images to produce monochrome
        if (expectedImageChannels == 1 && img.getType() == BufferedImage.TYPE_3BYTE_BGR)
          ;
        else
          throw badArg("Unsupported image type; wanted channels:", expectedImageChannels, "got:", INDENT,
              ImgUtil.toJson(img));
      }
      mExpectedImageType = img.getType();
    }
    if (img.getType() != mExpectedImageType)

      badArg("Unexpected image type, wanted:", mExpectedImageType, "but got:", INDENT, ImgUtil.toJson(img));
    if (!imgSize.equals(mExpectedImageSize))
      badArg("Unexpected image size, wanted:", mExpectedImageSize, "but got:", INDENT, ImgUtil.toJson(img));
  }

  private static final Map<Integer, Integer> sImgChannelsMap = mapWith(//
      // Let's disable some image types, to make things simpler

      //BufferedImage.TYPE_INT_RGB, 3, //
      //BufferedImage.TYPE_INT_BGR, 3, //
      BufferedImage.TYPE_3BYTE_BGR, 3, //
      BufferedImage.TYPE_BYTE_GRAY, 1, //
      BufferedImage.TYPE_USHORT_GRAY, 1 //
  );

  private static class ImageEntry {
    File imageFile;
    ScriptElementList scriptElements = ScriptElementList.DEFAULT_INSTANCE;
  }

  private ModelHandler modelHandler() {
    return mModelHandler;
  }

  
  
  
  
  
  
  
  
  
  
  
  
  public TransformWrapper buildAugmentTransform() {
    AugmentationConfig ac =config().augmentationConfig();  
    boolean horizFlip = ac.horizontalFlip() && random().nextBoolean();

    IPoint sourceImageSize =  modelHandler().model().inputImagePlanarSize();
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


  
  
  
  
  
  

  private float random(float min, float max) {
    checkArgument(max >= min);
    if (max == min)
      return min;
    return random().nextFloat() * (max - min) + min;
  }

  
  
  
  
  private final CompileImagesConfig mConfig;
  private final Random mRandom;
  private final ModelHandler mModelHandler;
  private final Files mFiles;
  private ImageTransformer mImageTransformer;
  private List<ImageEntry> mEntries;
  private int mExpectedImageType;
  private IPoint mExpectedImageSize = null;
  private float[] mWorkArray;
}
