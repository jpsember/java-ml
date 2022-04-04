package ml;

import java.util.List;

import gen.ImageSetInfo;
import js.graphics.gen.ScriptElementList;

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

  /**
   * Fill in information fields.  Some fields may have already been filled in; e.g. the image count, and the 
   */
  void storeImageSetInfo(ImageSetInfo.Builder imageSetInfo);
  
  /**
   * Get the (possibly transformed) annotations
   */
  default List<ScriptElementList> annotations() {
    throw new UnsupportedOperationException();
  }

}
