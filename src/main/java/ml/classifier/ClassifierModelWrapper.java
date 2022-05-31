package ml.classifier;

import static js.base.Tools.*;

import java.util.List;

import gen.Classifier;
import gen.ImageSetInfo;
import js.geometry.IRect;
import js.graphics.RectElement;
import js.graphics.ScriptElement;
import js.graphics.ScriptUtil;
import ml.ModelWrapper;
import static ml.NetworkUtil.*;

public final class ClassifierModelWrapper extends ModelWrapper<Classifier> {

  @Override
  public void storeImageSetInfo(ImageSetInfo.Builder imageSetInfo) {
    imageSetInfo //
        .labelLengthBytes(1 * bytesPerValue(network().labelDataType())) //
        .imageLengthBytes(inputImageVolumeProduct() * bytesPerValue(network().imageDataType())) //
    ;
  }

  @Override
  public List<ScriptElement> transformModelInputToScredit() {
    Classifier cl = modelConfig();
    byte[] categoryBytes = labelBufferBytes();
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
    byte[] categories = labelBufferBytes();
    checkArgument(scriptElements.size() == 1);
    // The buffer is only designed for a single element, but iterate anyways
    int i = INIT_INDEX;
    for (ScriptElement elem : scriptElements) {
      i++;
      int category = elem.properties().category();
      categories[i] = (byte) category;
    }
    return categories;
  }

  public Object constructLabelBuffer() {
    return new byte[1];
  }

}
