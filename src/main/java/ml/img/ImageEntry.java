package ml.img;

import static js.base.Tools.*;

import java.io.File;
import java.util.List;

import gen.TransformWrapper;
import js.geometry.IRect;
import js.graphics.RectElement;
import js.graphics.ScriptElement;
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

  final static boolean ONEONLY = alert("fixed box loc");

  public ScriptElementList scriptElementList() {
    if (mScriptElements == null) {
      mScriptElements = ScriptElementList.DEFAULT_INSTANCE;
      File scriptFile = ScriptUtil.scriptPathForImage(imageFile());
      if (scriptFile.exists()) {
        Script script = ScriptUtil.from(scriptFile);
        mScriptElements = ScriptUtil.extractScriptElementList(script);

        if (ONEONLY) {
          List<ScriptElement> tmp = arrayList();
          tmp.add(new RectElement(null, new IRect(20, 10, 30, 45)));
          mScriptElements = ScriptElementList.newBuilder().elements(tmp).build();
        }
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