package ml.img;

import static js.base.Tools.*;

import java.io.File;

import gen.TransformWrapper;
import js.graphics.ScriptUtil;
import js.graphics.gen.Script;
import js.graphics.gen.ScriptElementList;

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

  public ScriptElementList scriptElementList() {
    if (mScriptElements == null) {
      mScriptElements = ScriptElementList.DEFAULT_INSTANCE;
      File scriptFile = ScriptUtil.scriptPathForImage(imageFile());
      if (scriptFile.exists()) {
        Script script = ScriptUtil.from(scriptFile);
        mScriptElements = ScriptUtil.extractScriptElementList(script);
      }
    }
    return mScriptElements;
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
  private ScriptElementList mScriptElements;
  private TransformWrapper mTransform = TransformWrapper.DEFAULT_INSTANCE;

}
