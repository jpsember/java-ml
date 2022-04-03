package ml;

import static js.base.Tools.*;

import java.io.DataOutputStream;

import gen.PlotInferenceResultsConfig;

public final class ClassifierModelHandler extends ModelHandler {

  @Override
  public void addImageRecordFilters(ImageHandler p) {
    todo("can we make this method optional?");
  }

  @Override
  public ModelInputReceiver buildModelInputReceiver(DataOutputStream imagesStream,
      DataOutputStream labelsStream) {
    throw notFinished();
  }

  @Override
  public void plotInferenceResults(PlotInferenceResultsConfig config) {
    throw notFinished();
  }

}
