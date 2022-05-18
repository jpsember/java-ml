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
        .labelLengthBytes(Float.BYTES * 1) //
        .imageLengthBytes(inputImageVolumeProduct() * Float.BYTES) //
    ;
  }

  @Override
  public void accept(float[] image, List<ScriptElement> scriptElementList) {
    if (scriptElementList.size() != 1)
      throw badArg("expected single element:", INDENT, scriptElementList);
    writeImage(image);
    ScriptElement elem = scriptElementList.get(0);
    int category = elem.properties().category();
    int[] intArray = new int[1];
    intArray[0] = category;
    writeLabels(intArray);
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
