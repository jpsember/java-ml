package ml.img;

import java.io.File;

import js.graphics.ScriptUtil;
import js.graphics.gen.Script;
import js.graphics.gen.ScriptElementList;

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

  private final File mImageFile;
  private ScriptElementList mScriptElements = ScriptElementList.DEFAULT_INSTANCE;
}