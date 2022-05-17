package ml.classifier;

import java.util.List;

import gen.TransformWrapper;
import js.graphics.ScriptElement;
import ml.ModelServiceProvider;
import ml.ModelWrapper;

public final class ClassifierModelWrapper extends ModelWrapper {

  @Override
  public ModelServiceProvider buildModelServiceProvider() {
    return new ClassifierServiceProvider();
  }

  @Override
  public void transformAnnotations(List<ScriptElement> in, List<ScriptElement> out, TransformWrapper transform) {
    // Classifiers have no annotations
    return;
  }

}
