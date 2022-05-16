package ml.img;

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
  
  private final File mImageFile;
  private ScriptElementList mScriptElements = ScriptElementList.DEFAULT_INSTANCE;
  private TransformWrapper mTransform = TransformWrapper.DEFAULT_INSTANCE;
}