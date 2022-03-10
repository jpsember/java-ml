package ml;

import java.util.List;

import js.graphics.gen.ScriptElementList;
import js.graphics.Inspector;

import static js.base.Tools.*;

/**
 * Receives annotated image, suitable for applying to a model (for training or
 * inference)
 */
public interface ModelInputReceiver {

  /**
   * Process an image and its annotations, converting to form suitable for
   * training
   */
  void accept(float[] image, ScriptElementList annotation);

  default void setInspector(Inspector inspector) {
    throw notSupported("setInspector", "implementing class:", getClass().getSimpleName());
  }

  /**
   * Get the (possibly transformed) annotations
   */
  default List<ScriptElementList> annotations() {
    throw new UnsupportedOperationException();
  }

}
