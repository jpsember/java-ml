package ml;

import static js.base.Tools.*;

import js.base.BaseObject;
import js.geometry.Matrix;
import js.graphics.Inspector;
import js.graphics.gen.ScriptElementList;

/**
 * Converts images of type T to arrays of floats, while applying a linear
 * transformation
 */
public abstract class ImageTransformer<T> extends BaseObject {

  public abstract void transform(Matrix sourceToDestTransform, Matrix destToSourceTransform, T sourceImage,
      float[] destination);

  public final void setInspector(Inspector inspector) {
    loadTools();
    mInspector = Inspector.orNull(inspector);
  }

  public void withPendingAnnotation(ScriptElementList annotation) {
    mPendingAnnotation = annotation;
  }

  public final Inspector inspector() {
    return mInspector;
  }

  /**
   * If there are pending annotations, apply them to the inspector's current
   * image
   */
  protected final void applyPendingAnnotations() {
    if (mPendingAnnotation == null)
      return;
    mInspector.elements(mPendingAnnotation.elements());
    mPendingAnnotation = null;
  }

  private Inspector mInspector = Inspector.NULL_INSPECTOR;
  private ScriptElementList mPendingAnnotation;
}
