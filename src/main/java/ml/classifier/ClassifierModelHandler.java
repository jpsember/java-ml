package ml.classifier;

import static js.base.Tools.*;

import java.io.DataOutputStream;

import gen.ImageSetInfo;
import gen.PlotInferenceResultsConfig;
import js.data.DataUtil;
import js.file.Files;
import js.geometry.IPoint;
import js.geometry.IRect;
import js.graphics.RectElement;
import js.graphics.ScriptElement;
import js.graphics.TextElement;
import js.graphics.gen.Script;
import js.graphics.gen.ScriptElementList;
import ml.ImageHandler;
import ml.ModelHandler;
import ml.ModelInputReceiver;
import ml.ModelWrapper;

public final class ClassifierModelHandler extends ModelHandler {

  @Override
  public void addImageRecordFilters(ImageHandler p) {
    todo("can we make this method optional?");
  }

  @Override
  public ModelInputReceiver buildModelInputReceiver(DataOutputStream imagesStream,
      DataOutputStream labelsStream) {
    return new OurModelInputReceiver(imagesStream, labelsStream);
  }

  @Override
  public void plotInferenceResults(PlotInferenceResultsConfig config) {
    throw notFinished();
  }

  private class OurModelInputReceiver implements ModelInputReceiver {

    public OurModelInputReceiver(DataOutputStream imagesStream, DataOutputStream labelsStream) {
      mImagesStream = imagesStream;
      mLabelsStream = labelsStream;
    }

    @Override
    public void storeImageSetInfo(ModelWrapper model, ImageSetInfo.Builder imageSetInfo) {
      imageSetInfo //
          .labelLengthBytes(Float.BYTES * 1) //
          .imageLengthBytes(model.inputImagePlanarSize().product() * Float.BYTES) //
      ;
    }

    @Override
    public void accept(float[] image, ScriptElementList annotation) {
      Files.S.writeFloatsLittleEndian(image, mImagesStream);
      if (annotation.elements().size() != 1)
        throw badArg("expected single element:", INDENT, annotation);
      ScriptElement elem = annotation.elements().get(0);
      int category = elem.properties().category();
      int[] intArray = new int[1];
      intArray[0] = category;
      Files.S.writeIntsLittleEndian(intArray, mLabelsStream);
    }

    @Override
    public void parseInferenceResult(byte[] modelOutput, Script.Builder script) {
      int[] categories = DataUtil.bytesToIntsLittleEndian(modelOutput);
      int category = categories[0];
      ScriptElement elem;
      if (todo("add support for TextElements to scredit"))
        elem = new RectElement(null,
            IRect.withLocAndSize(IPoint.with(10 + category * 30, 5), IPoint.with(5, 5)));
      else
        elem = new TextElement("AB".substring(category, category + 1), IPoint.with(20, 30));
      todo("pass in model/model wrapper where appropriate");
      script.items().add(elem);
    }

    private DataOutputStream mImagesStream;
    private DataOutputStream mLabelsStream;

  }

}
