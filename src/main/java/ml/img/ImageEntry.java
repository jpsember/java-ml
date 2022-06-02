package ml.img;

import static js.base.Tools.*;

import java.io.File;
import java.util.List;

import gen.TransformWrapper;
import js.graphics.ScriptElement;
import js.graphics.ScriptUtil;
import js.graphics.gen.Script;

/**
 * Encapsulates information about a single image for use by ImageCompiler
 */
class ImageEntry {

  ImageEntry(File imageFile) {
    loadTools();
    mImageFile = imageFile;
  }

  public File imageFile() {
    return mImageFile;
  }

  public List<ScriptElement> scriptElements() {
    if (mElements == null) {
      mElements = arrayList();
      File scriptFile = ScriptUtil.scriptPathForImage(imageFile());
      if (scriptFile.exists()) {
        Script script = ScriptUtil.from(scriptFile);
        mElements.addAll(script.items());
      }
    }
    return mElements;
  }

  public void setTransform(TransformWrapper t) {
    mTransform = t;
  }

  public TransformWrapper transform() {
    return mTransform;
  }

  /**
   * Discard any resources that were created while generating an image set
   */
  public void releaseResources() {
    mTransform = TransformWrapper.DEFAULT_INSTANCE;
  }

  private final File mImageFile;
  private TransformWrapper mTransform = TransformWrapper.DEFAULT_INSTANCE;
  private List<ScriptElement> mElements;
}
