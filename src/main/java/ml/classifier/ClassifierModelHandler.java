package ml.classifier;

import static js.base.Tools.*;

import gen.PlotInferenceResultsConfig;
import ml.ModelHandler;
import ml.ModelServiceProvider;

public final class ClassifierModelHandler extends ModelHandler {


  @Override
  public ModelServiceProvider buildModelServiceProvider() {
    return new ClassifierServiceProvider();
  }

  @Override
  public void plotInferenceResults(PlotInferenceResultsConfig config) {
    throw notFinished();
  }
}
