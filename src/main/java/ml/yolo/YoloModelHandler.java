package ml.yolo;

import java.awt.image.BufferedImage;
import java.io.DataInputStream;
import java.io.File;
import java.util.List;

import static js.base.Tools.*;
import js.file.Files;
import js.geometry.FPoint;
import js.geometry.FRect;
import js.geometry.IPoint;
import js.graphics.ImgUtil;
import js.graphics.RectElement;
import js.graphics.ScriptElement;
import js.graphics.ScriptUtil;
import js.graphics.gen.ElementProperties;
import js.graphics.gen.Script;
import ml.Util;
import ml.ImageHandler;
import ml.ImageRecord;
import ml.ModelHandler;
import ml.ModelInputReceiver;
import ml.NetworkAnalyzer;
import gen.Layer;
import gen.LayerType;
import gen.PlotInferenceResultsConfig;
import gen.Vol;
import gen.Yolo;
import ml.NetworkUtil;
import ml.VolumeUtil;
import ml.yolo.YoloResultParser;
import ml.yolo.YoloUtil;

public final class YoloModelHandler extends ModelHandler {

  @Override
  public void addImageRecordFilters(ImageHandler p) {
    todo("move yolo-related classes to own subpackage");
    loadTools();
    p.withFilter(ImageRecord.FILTER_SCRIPT_REQUIRED);
    p.withFilter(ImageRecord.FILTER_SHAPE_OR_RETAIN_REQUIRED);
    // Read the image last, in case already filtered
    p.withFilter(ImageRecord.filterEnsureImageSize(model().inputImagePlanarSize()));
  }

  @Override
  public ModelInputReceiver buildModelInputReceiver() {
    YoloImageReceiver r = new YoloImageReceiver();
    todo("do we need to call storeImageSetInfo?");
    return r;
  }

  @Override
  public void plotInferenceResults(PlotInferenceResultsConfig config) {

    Yolo yolo = model().modelConfig();

    File imagesFile = Files.assertExists(new File(config.inferenceInputDir(), Util.EVAL_IMAGES_FILENAME));
    File resultsFile = Files.assertExists(new File(config.inferenceResultsDir(), Util.EVAL_RESULTS_FILENAME));

    int imageLengthInBytes = model().inputImageVolumeProduct() * Float.BYTES;
    int imageCount = (int) (imagesFile.length() / imageLengthInBytes);

    Files.assertFileLength(imagesFile, imageCount * (long) imageLengthInBytes, "images");
    Files.assertFileLength(resultsFile, imageCount * (long) YoloUtil.imageLabelFloatCount(yolo) * Float.BYTES,
        "results");

    DataInputStream imagesInput = Files.dataInputStream(imagesFile);
    DataInputStream resultsInput = Files.dataInputStream(resultsFile);

    Files.S.backupAndRemake(config.outputDir());

    File annotationDir = ScriptUtil.scriptDirForProject(config.outputDir());
    Files.S.mkdirs(annotationDir);
    YoloResultParser yr = new YoloResultParser(yolo);
    yr.withConfidenceFilter(config.confidencePct() / 100f);

    for (int imageNumber = 0; imageNumber < imageCount; imageNumber++) {
      yr.setVerbose(config.inspectionFrameNumber() == imageNumber);
      yr.log("Plot Inference Results, image", imageNumber);
      float[] pixels = Files.readFloatsLittleEndian(imagesInput, model().inputImageVolumeProduct());
      BufferedImage bi = constructBufferedImage(pixels);

      String imageFilenameRoot = String.format("%03d", imageNumber);
      File imgDest = new File(config.outputDir(), imageFilenameRoot + ".jpg");
      ImgUtil.writeImage(Files.S, bi, imgDest);

      float[] imageLabelData = Files.readFloatsLittleEndian(resultsInput,
          YoloUtil.imageLabelFloatCount(yolo));
      List<ScriptElement> boxList = yr.readImageResult(imageLabelData);
      if (config.maxIOverU() > 0) {
        boxList = YoloUtil.performNonMaximumSuppression(boxList, config.maxIOverU());
      }

      Script.Builder script = Script.newBuilder();

      if (config.plotAnchorBoxes() && imageNumber == 0) {
        generateAnchorBoxSummary(boxList);
      }
      script.items(boxList);

      File annotPath = new File(annotationDir, imageFilenameRoot + ".json");
      ScriptUtil.write(Files.S, script, annotPath);
    }

    Files.close(imagesInput, resultsInput);
  }

  private void generateAnchorBoxSummary(List<ScriptElement> elements) {
    Yolo yolo = model().modelConfig();
    FPoint blockSize = yolo.blockSize().toFPoint();
    FPoint halfBlock = blockSize.scaledBy(0.5f);
    int numBox = YoloUtil.anchorBoxCount(yolo);

    for (int i = 0; i < numBox; i++) {
      IPoint cpt;
      {
        int row = i / 3;
        int col = (i % 3);
        cpt = new IPoint(col * blockSize.x * 4 + halfBlock.x, row * blockSize.y * 4 + halfBlock.y);
      }

      FPoint box = yolo.anchorBoxesPixels().get(i).toFPoint();
      float x = cpt.x - box.x / 2;
      float y = cpt.y - box.y / 2;
      elements.add(new RectElement(ElementProperties.newBuilder().category(0),
          new FRect(x, y, box.x, box.y).toIRect()));
      elements.add(new RectElement(ElementProperties.newBuilder().category(1),
          new FRect(cpt.x - halfBlock.x, cpt.y - halfBlock.x, blockSize.x, blockSize.y).toIRect()));
    }
  }

  @Override
  public void extractShapes(Script script, List<ScriptElement> target) {
    assertNoMixing(script);
    target.addAll(ScriptUtil.polygonElements(script));
    for (RectElement b : ScriptUtil.rectElements(script)) {
      target.add(b);
    }
  }

  @Override
  public boolean processLayer(NetworkAnalyzer analyzer, int layerIndex) {
    Layer.Builder builder = analyzer.layerBuilder(layerIndex);
    if (builder.type() == LayerType.YOLO) {
      auxProcessLayer(analyzer, builder);
      return true;
    }
    return false;
  }

  private void auxProcessLayer(NetworkAnalyzer analyzer, Layer.Builder layer) {
    Yolo yol = model().modelConfig();
    yol = YoloUtil.validate(yol);

    IPoint inputImageSize = VolumeUtil.spatialDimension(layer.inputVolume());
    IPoint gridSize = YoloUtil.gridSize(yol);

    if (!inputImageSize.equals(gridSize)) {
      todo("Perhaps this shouldn't be a problem if fullyConnected is in effect?");
      analyzer.addProblem("Spatial dimension of input volume:", inputImageSize, "!= Yolo grid size",
          gridSize);
      return;
    }
    int valuesPerBlock = YoloUtil.valuesPerBlock(yol);
    IPoint grid = YoloUtil.gridSize(yol);

    // I think what happens here is we apply 1x1 spatial filters to the input volume
    // to produce an output volume that has the same spatial dimension as the input,
    // but with the number of filters chosen to equal valuesPerBlock
    int numFilters = valuesPerBlock;

    if (layer.filters() != 0 && layer.filters() != numFilters) {
      analyzer.addProblem("Unexpected Yolo filters:", layer.filters(), "!=", numFilters);
      return;
    }
    layer.filters(numFilters);

    Vol inBox = layer.inputVolume();
    Vol outputBox = VolumeUtil.build(grid.x, grid.y, valuesPerBlock);

    NetworkUtil.calcWeightsForConv(layer, VolumeUtil.fibre(inBox.depth()), valuesPerBlock, outputBox);
    layer.outputVolume(outputBox);
  }

  @Override
  public void describeLayer(NetworkAnalyzer an, Layer layer, StringBuilder sb) {
    Yolo yol = model().modelConfig();
    sb.append("anchors:" + yol.anchorBoxesPixels().size());
    sb.append(" categories:" + yol.categoryCount());
  }

}
