package ml.classifier;

import static js.base.Tools.*;

import java.io.DataOutputStream;

import gen.PlotInferenceResultsConfig;
import js.file.Files;
import js.graphics.ScriptElement;
import js.graphics.gen.ScriptElementList;
import ml.ImageHandler;
import ml.ModelHandler;
import ml.ModelInputReceiver;

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
    public void accept(float[] image, ScriptElementList annotation) {
      Files.S.writeFloatsLittleEndian(image, mImagesStream);
      if (annotation.elements().size() != 1)
        throw badArg("expected single element:", INDENT, annotation);
      ScriptElement elem = annotation.elements().get(0);
      int category = elem.properties().category();
      try {
        mLabelsStream.writeFloat(category);
      } catch (Throwable e) {
        throw Files.asFileException(e);
      }
    }

    private DataOutputStream mImagesStream;
    private DataOutputStream mLabelsStream;
  }

}
