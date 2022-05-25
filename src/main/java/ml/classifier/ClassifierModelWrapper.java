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
  public void accept(Object imagePixelArray, List<ScriptElement> scriptElementList) {
    if (scriptElementList.size() != 1)
      throw badArg("expected single element:", INDENT, scriptElementList);
    writeImage(imagePixelArray);
    ScriptElement elem = scriptElementList.get(0);
    int category = elem.properties().category();
    byte[] byteArray = new byte[1];
    byteArray[0] = (byte)category;
    writeLabels(byteArray);
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

}
