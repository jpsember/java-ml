package ml;

import static js.base.Tools.*;

import gen.Vol;
import js.geometry.IPoint;

/**
 * Utilities for working with the Vol data class
 */
public final class VolumeUtil {

  public static IPoint spatialDimension(Vol volume) {
    return new IPoint(volume.width(), volume.height());
  }

  /**
   * Construct a 1x1 'fibre' with a particular depth
   */
  public static Vol fibre(int depth) {
    return build(1, 1, depth);
  }

  /**
   * Get the number of elements contained by this volume
   */
  public static int product(Vol volume) {
    return volume.width() * volume.height() * volume.depth();
  }

  public static String toString(Vol volume) {
    return String.format("(%4d x %4d x %5d)", volume.width(), volume.height(), volume.depth());
  }

  public static Vol reducedForPooling(Vol volume, int strideX, int strideY) {
    if (!(volume.width() % strideX == 0 && volume.height() % strideY == 0))
      throw die("Volume has odd dimension; cannot be reduced spatially;", INDENT, volume);
    return build(volume.width() / strideX, volume.height() / strideY, volume.depth()).build();
  }

  public static Vol withDepth(Vol volume, int depth) {
    return volume.build().toBuilder().depth(depth).build();
  }

  public static Vol build(int width, int height, int depth) {
    Vol volume = Vol.newBuilder().//
        width(width)//
        .height(height)//
        .depth(depth)//
        .build();
    return ensureValid(volume);
  }

  public static Vol ensureValid(Vol v) {
    if (v.width() <= 0 || v.height() <= 0 || v.depth() <= 0)
      throw die("degenerate volume:", INDENT, v);
    return v.build();
  }

}
