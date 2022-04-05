package ml;

import java.io.DataOutputStream;
import java.util.List;

import gen.ImageSetInfo;
import js.base.BaseObject;
import js.graphics.gen.Script;
import js.graphics.gen.ScriptElementList;
import static js.base.Tools.*;

/**
 * Model-specific operations
 */
public abstract class ModelServiceProvider extends BaseObject {

  public final void setModel(ModelWrapper model) {
    checkState(mModel == null, "model already defined");
    mModel = model;
    prepareModel();
  }

  public final ModelWrapper model() {
    return mModel;
  }

  /**
   * Perform additional initialization. Default implementation does nothing
   */
  public void prepareModel() {
  }

  public final void setImageStream(DataOutputStream imageStream) {
    checkState(mImageOutputStream == null, "stream already defined");
    mImageOutputStream = imageStream;
  }

  public final void setLabelStream(DataOutputStream labelStream) {
    checkState(mLabelOutputStream == null, "stream already defined");
    mLabelOutputStream = labelStream;
  }

  public final DataOutputStream imageOutputStream() {
    return mImageOutputStream;
  }

  public final DataOutputStream labelOutputStream() {
    return mLabelOutputStream;
  }

  /**
   * Process an image and its annotations, converting to form suitable for
   * training
   */
  public abstract void accept(float[] image, ScriptElementList annotation);

  /**
   * Fill in information fields. Some fields may have already been filled in
   */
  public abstract void storeImageSetInfo(ImageSetInfo.Builder imageSetInfo);

  /**
   * Get the (possibly transformed) annotations
   */
  public List<ScriptElementList> annotations() {
    throw notSupported("subclass:", getClass().getName());
  }

  public void parseInferenceResult(byte[] modelOutput, Script.Builder script) {
    throw notSupported("subclass:", getClass().getName());
  }

  private ModelWrapper mModel;
  private DataOutputStream mImageOutputStream;
  private DataOutputStream mLabelOutputStream;
}
