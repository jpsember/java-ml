package ml;

import java.io.File;
import java.util.List;

import gen.ImageSetInfo;
import js.file.Files;
import js.graphics.gen.Script;
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
   * Fill in information fields. Some fields may have already been filled in;
   * e.g. the image count, and the
   */
  void storeImageSetInfo(ModelWrapper model, ImageSetInfo.Builder imageSetInfo);

  /**
   * Get the (possibly transformed) annotations
   */
  default List<ScriptElementList> annotations() {
    throw new UnsupportedOperationException();
  }

  default void parseInferenceResult(byte[] modelOutput, Script.Builder script) {
    throw new UnsupportedOperationException();
  }

  default void decompileImage(byte[] modelInputImageAsBytes, Files files, File imageFile) {
    throw new UnsupportedOperationException();
  }
  
}
