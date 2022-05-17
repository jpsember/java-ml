package ml.classifier;

import static js.base.Tools.*;

import java.util.List;

import gen.PlotInferenceResultsConfig;
import gen.TransformWrapper;
import js.graphics.ScriptElement;
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

  @Override
  public void transformAnnotations(List<ScriptElement> in, List<ScriptElement> out, TransformWrapper transform) {
    // Classifiers have no annotations
    return;
  }

}
