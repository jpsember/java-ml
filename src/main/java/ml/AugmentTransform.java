package ml;

import static js.base.Tools.*;
import js.geometry.Matrix;

/**
 * Bookkeeping class for transformation matrix, its inverse, and a rotation_degrees field.  Inverse is lazily calculated;
 * otherwise, we could use pojo (if it supported matrices)
 */
public final class AugmentTransform {

  public AugmentTransform(Matrix transform, int rotationDegrees) {
    todo("Rename class to TransformWrapper");
    mMatrix = transform;
    mRotationDegrees = rotationDegrees;
  }

  public Matrix matrix() {
    return mMatrix;
  }

  /**
   * Get the inverse of the transform
   */
  public Matrix inverse() {
    if (mInverse == null)
      mInverse = matrix().invert();
    return mInverse;
  }

  public int rotationDegrees() {
    return mRotationDegrees;
  }

  private final Matrix mMatrix;
  private Matrix mInverse;
  private final int mRotationDegrees;

}
