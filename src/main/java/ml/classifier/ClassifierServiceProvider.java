package ml.classifier;

import static js.base.Tools.*;

import gen.Classifier;
import gen.ImageSetInfo;
import js.data.DataUtil;
import js.file.Files;
import js.geometry.IPoint;
import js.geometry.IRect;
import js.graphics.RectElement;
import js.graphics.ScriptElement;
import js.graphics.TextElement;
import js.graphics.gen.Script;
import js.graphics.gen.ScriptElementList;
import ml.ModelServiceProvider;

public class ClassifierServiceProvider extends ModelServiceProvider {

  @Override
  public void storeImageSetInfo(ImageSetInfo.Builder imageSetInfo) {
    imageSetInfo //
        .labelLengthBytes(Float.BYTES * 1) //
        .imageLengthBytes(model().inputImageVolumeProduct() * Float.BYTES) //
    ;
  }

  @Override
  public void accept(float[] image, ScriptElementList scriptElementList) {
    Files.S.writeFloatsLittleEndian(image, imageOutputStream());
    if (scriptElementList.elements().size() != 1)
      throw badArg("expected single element:", INDENT, scriptElementList);
    ScriptElement elem = scriptElementList.elements().get(0);
    int category = elem.properties().category();
    int[] intArray = new int[1];
    intArray[0] = category;
    Files.S.writeIntsLittleEndian(intArray, labelOutputStream());
  }

  @Override
  public void parseInferenceResult(byte[] modelOutput, Script.Builder script) {
    int[] categories = DataUtil.bytesToIntsLittleEndian(modelOutput);
    int category = categories[0];
    Classifier cl = model().modelConfig();
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
