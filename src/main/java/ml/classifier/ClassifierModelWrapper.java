package ml.classifier;

import static js.base.Tools.*;
import static ml.NetworkUtil.*;

import java.io.File;
import java.util.List;

import gen.Classifier;
import gen.ImageSetInfo;
import gen.Layer;
import gen.LayerType;
import gen.Vol;
import js.file.Files;
import js.geometry.IRect;
import js.graphics.RectElement;
import js.graphics.ScriptElement;
import js.graphics.ScriptUtil;
import js.graphics.gen.Script;
import ml.ModelWrapper;
import ml.NetworkAnalyzer;
import ml.NetworkUtil;
import ml.VolumeUtil;

public final class ClassifierModelWrapper extends ModelWrapper<Classifier> {

  @Override
  public boolean processLayer(NetworkAnalyzer analyzer, Layer.Builder layer) {
    if (layer.type() != LayerType.CLASSIFIER)
      return false;
    auxProcessLayer(analyzer, layer);
    return true;
  }

  private void auxProcessLayer(NetworkAnalyzer analyzer, Layer.Builder layer) {
    Classifier cl = modelConfig();

    int inputDepth = layer.inputVolume().depth();

    if (inputDepth != cl.categoryCount()) {
      // Add a fully-connected layer to generate outputs from the inputs.

      Vol inBox = layer.inputVolume();
      Vol outputBox = VolumeUtil.build(1, 1, cl.categoryCount());

      int inputVolume = VolumeUtil.product(inBox);
      int outputVolume = VolumeUtil.product(outputBox);
      layer.outputVolume(outputBox);

      layer.filters(1);
      NetworkUtil.calcWeightsForFC(layer, inputVolume, outputVolume);

    } else {
      // The input volume has the same dimensions as the classifier output layer,
      // so treat the input layer as the output directly
      layer.numWeights(0);
      layer.filters(1);
    }
  }

  @Override
  public void describeLayer(NetworkAnalyzer an, Layer layer, StringBuilder sb) {
    checkArgument(layer.type() == LayerType.CLASSIFIER);
    sb.append(" categories:" + modelConfig().categoryCount());
  }

  @Override
  public void storeImageSetInfo(ImageSetInfo.Builder imageSetInfo) {
    imageSetInfo //
        .labelLengthBytes(1 * bytesPerValue(network().labelDataType())) //
        .imageLengthBytes(inputImageVolumeProduct() * bytesPerValue(network().imageDataType())) //
    ;
  }

  @Override
  public List<ScriptElement> transformModelInputToScredit() {
    Classifier cl = modelConfig();
    byte[] categoryBytes = labelBufferBytes();
    List<ScriptElement> output = arrayList();
    for (byte catByte : categoryBytes) {
      int category = catByte;
      checkArgument(category >= 0 && category < cl.categoryCount());
      ScriptElement elem = new RectElement(ScriptUtil.setCategory(null, category),
          new IRect(inputImagePlanarSize()));
      output.add(elem);
    }
    return output;
  }

  @Override
  public Object transformScreditToModelInput(List<ScriptElement> scriptElements) {
    byte[] categories = labelBufferBytes();
    checkArgument(scriptElements.size() == 1);
    // The buffer is only designed for a single element, but iterate anyways
    int i = INIT_INDEX;
    for (ScriptElement elem : scriptElements) {
      i++;
      int category = elem.properties().category();
      categories[i] = (byte) category;
    }
    return categories;
  }

  public Object constructLabelBuffer() {
    return new byte[1];
  }

  @Override
  public void readInferenceOutputLabels(File file, int imageCount) {
    float[] results = Files.readFloatsLittleEndian(file, "ClassifierModelWrapper.readInferenceOutputLabels");
    int labelCount = imageCount * modelConfig().categoryCount();
    if (labelCount != results.length)
      badArg("label count != labels length", labelCount, results.length, "image count:", imageCount,
          "categoryCount", modelConfig().categoryCount());
    mInferenceOutputLabels = results;
  }

  @Override
  public void transformInferenceOutputToScript(int imageNumber, Script.Builder script) {
    List<ScriptElement> output = arrayList();
    int cc = modelConfig().categoryCount();
    float maxLogProb = 0;
    int maxCat = -1;
    int j = imageNumber * cc;
    for (int i = 0; i < cc; i++) {
      var logProb = mInferenceOutputLabels[j + i];
      if (maxCat < 0 || logProb > maxLogProb) {
        maxCat = i;
        maxLogProb = logProb;
      }
    }

    ScriptElement elem = new RectElement(ScriptUtil.setCategory(null, maxCat),
        new IRect(inputImagePlanarSize()));
    output.add(elem);

    script.items(output);
  }

  private float[] mInferenceOutputLabels;

}
