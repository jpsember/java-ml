package ml.classifier;

import static js.base.Tools.*;

import java.util.List;

import gen.Classifier;
import gen.ImageSetInfo;
import gen.TransformWrapper;
import js.data.DataUtil;
import js.geometry.IPoint;
import js.geometry.IRect;
import js.graphics.RectElement;
import js.graphics.ScriptElement;
import js.graphics.TextElement;
import js.graphics.gen.Script;
import js.graphics.gen.ScriptElementList;
import ml.ModelWrapper;

public final class ClassifierModelWrapper extends ModelWrapper {

  @Override
  public void transformAnnotations(List<ScriptElement> in, List<ScriptElement> out,
      TransformWrapper transform) {
    // Classifiers have no annotations
    return;
  }

  @Override
  public void storeImageSetInfo(ImageSetInfo.Builder imageSetInfo) {
    imageSetInfo //
        .labelLengthBytes(Float.BYTES * 1) //
        .imageLengthBytes( inputImageVolumeProduct() * Float.BYTES) //
    ;
  }

  @Override
  public void accept(float[] image, ScriptElementList scriptElementList) {
    if (scriptElementList.elements().size() != 1)
      throw badArg("expected single element:", INDENT, scriptElementList);
    writeImage(image);
    ScriptElement elem = scriptElementList.elements().get(0);
    int category = elem.properties().category();
    int[] intArray = new int[1];
    intArray[0] = category;
    writeLabels(intArray);
  }

  @Override
  public void parseInferenceResult(byte[] modelOutput, Script.Builder script) {
    int[] categories = DataUtil.bytesToIntsLittleEndian(modelOutput);
    int category = categories[0];
    Classifier cl =  modelConfig();
    checkArgument(category >= 0 && category < cl.categoryCount());

    ScriptElement elem;
    if (todo("add support for TextElements to scredit"))
      elem = new RectElement(null,
          IRect.withLocAndSize(IPoint.with(10 + category * 30, 5), IPoint.with(5, 5)));
    else
      elem = new TextElement("" + category, IPoint.with(20, 30));
    script.items().add(elem);
  }

}
