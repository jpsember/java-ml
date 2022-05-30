package ml.classifier;

import static js.base.Tools.*;

import java.util.List;

import gen.Classifier;
import gen.ImageSetInfo;
import gen.TransformWrapper;
import js.data.DataUtil;
import js.geometry.IRect;
import js.graphics.RectElement;
import js.graphics.ScriptElement;
import js.graphics.ScriptUtil;
import js.graphics.gen.Script;
import ml.LabelledImage;
import ml.ModelWrapper;
import static ml.NetworkUtil.*;

public final class ClassifierModelWrapper extends ModelWrapper<Classifier> {

  @Override
  public void transformAnnotations(List<ScriptElement> in, List<ScriptElement> out,
      TransformWrapper transform) {
    // Annotations hold ony the category, so pass through unchanged
    out.addAll(in);
  }

  @Override
  public void storeImageSetInfo(ImageSetInfo.Builder imageSetInfo) {
    imageSetInfo //
        .labelLengthBytes(1 * bytesPerValue(network().labelDataType())) //
        .imageLengthBytes(inputImageVolumeProduct() * bytesPerValue(network().imageDataType())) //
    ;
  }

  @Override
  public void accept(LabelledImage labelledImage) {
    writeImage(labelledImage);
    transformScreditToModelInput(labelledImage.annotations());
    writeLabels(mCategoryBuffer);
  }

  @Override
  public List<ScriptElement> transformModelInputToScredit(Object input) {
    Classifier cl = modelConfig();
    byte[] categoryBytes = (byte[]) input;
    List<ScriptElement> output = arrayList();
    for (byte catByte : categoryBytes) {
      int category = catByte;
      checkArgument(category >= 0 && category < cl.categoryCount());
      ScriptElement elem = new RectElement(ScriptUtil.setCategory(null, category),
          new IRect(inputImagePlanarSize()));
      output.add(elem);
    }
    return output;
  }

  @Override
  public Object transformScreditToModelInput(List<ScriptElement> scriptElements) {
    checkArgument(scriptElements.size() == 1);
    // The buffer is only designed for a single element, but iterate anyways
    int i = INIT_INDEX;
    for (ScriptElement elem : scriptElements) {
      i++;
      int category = elem.properties().category();
      mCategoryBuffer[i] = (byte) category;
    }
    return mCategoryBuffer;
  }

  @Override
  public void parseInferenceResult(byte[] modelOutput, int confidencePct, Script.Builder script) {
    int[] categories = DataUtil.bytesToIntsLittleEndian(modelOutput);
    int category = categories[0];
    Classifier cl = modelConfig();
    checkArgument(category >= 0 && category < cl.categoryCount());
    ScriptElement elem = new RectElement(ScriptUtil.setCategory(null, category),
        new IRect(inputImagePlanarSize()));
    script.items().add(elem);
  }

  public byte[] getLabelBuffer() {
    return mCategoryBuffer;
  }

  private byte[] mCategoryBuffer = new byte[1];
}
