package ml;

import static js.base.Tools.*;

import java.util.List;

import js.base.BasePrinter;
import js.graphics.ScriptElement;

/**
 * A wrapper class around an image and its labels to simplify code involving the
 * label and image formats involved
 */
public final class LabelledImage {

  public LabelledImage(ModelWrapper model) {
    mModel = model;
  }

  public void setPixels(float[] imageFloats) {
    setPixelsDefined();
    mPixelsF = imageFloats;
  }

  public void setPixels(byte[] imageBytes) {
    setPixelsDefined();
    mPixelsB = imageBytes;
  }

  public void setAnnotations(List<ScriptElement> annotations) {
    if (mAnnotations != null)
      trouble("already has annotations");
    mAnnotations = annotations;
  }

  public void useOnlySingleElement() {
    if (annotations().size() > 1) {
      alert("using only single element");
      removeAllButFirstN(annotations(), 1);
    }
  }

  public List<ScriptElement> annotations() {
    if (mAnnotations == null)
      trouble("no annotations");
    return mAnnotations;
  }

  private void setPixelsDefined() {
    if (mHasPixels)
      trouble("already has pixels");
    mHasPixels = true;
  }

  public float[] pixelsF() {
    if (mPixelsF == null)
      trouble("no pixelsF");
    return mPixelsF;
  }

  public byte[] pixelsB() {
    if (mPixelsB == null)
      trouble("no pixelsB");
    return mPixelsB;
  }

  private void trouble(Object... messages) {
    throw badState("LabelledImage problem: " + BasePrinter.toString(messages));
  }

  private boolean mHasPixels;
  private float[] mPixelsF;
  private byte[] mPixelsB;
  private List<ScriptElement> mAnnotations;
  private final ModelWrapper mModel;

}
