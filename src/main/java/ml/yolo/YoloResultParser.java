package ml.yolo;

import static js.base.Tools.*;
import static ml.yolo.YoloUtil.*;

import java.util.Collections;
import java.util.List;

import js.base.BaseObject;
import js.geometry.FPoint;
import js.geometry.FRect;
import js.geometry.IPoint;
import js.geometry.IRect;
import js.geometry.MyMath;
import js.graphics.RectElement;
import js.graphics.ScriptElement;
import js.graphics.gen.CategoryConfidence;
import js.graphics.gen.ElementProperties;
import js.json.JSMap;
import ml.NetworkUtil;
import gen.Yolo;

/**
 * Parses output from a Yolo model to produce annotations
 */
public final class YoloResultParser extends BaseObject {

  public YoloResultParser(Yolo yolo) {
    mYolo = yolo;
    mGridToImageScale = yolo.blockSize().toFPoint();
    mGridSize = YoloUtil.gridSize(yolo);
  }

  public void withConfidenceFilter(float confidence) {
    mConfidenceThreshold = confidence;
  }


  public List<ScriptElement> readImageResult(float[] imageData) {
 
    log("Constructing YOLO result for image");
    log("...confidence threshold %", pct(mConfidenceThreshold));

    int expectedDataLength = mGridSize.product() * YoloUtil.valuesPerBlock(mYolo);
    if (imageData.length != expectedDataLength) {
      throw die("length of image data:", imageData.length, "!= expected value:", expectedDataLength, INDENT,
          mYolo);
    }

    List<ScriptElement> boxList = arrayList();

    float[] f = imageData;

    int anchorBoxCount = YoloUtil.anchorBoxCount(mYolo);
    int fieldsPerBox = YoloUtil.valuesPerAnchorBox(mYolo);

    final float logitMinForResult = NetworkUtil.logit(mConfidenceThreshold);

    int categoryCount = mYolo.categoryCount();

    List<CategoryConfidence> categoryConfidences = arrayList();
    StringBuilder sbLine = null;
    StringBuilder sbGrid = null;
    JSMap gridMap = null;
    if (verbose()) {
      sbLine = new StringBuilder();
      sbGrid = new StringBuilder();
      gridMap = map();
    }

    float highestObjectnessLogitSeen = 0f;
    int fieldSetIndex = 0;
    for (int cellY = 0; cellY < mGridSize.y; cellY++) {
      if (verbose()) {
        sbGrid.setLength(0);
        sbGrid.append("[");
      }

      for (int cellX = 0; cellX < mGridSize.x; cellX++) {

        char cellString = 0;
        for (int anchorBox = 0; anchorBox < anchorBoxCount; anchorBox++, fieldSetIndex += fieldsPerBox) {

          float objectnessLogit = f[fieldSetIndex + F_CONFIDENCE];
          // Note, this check will cause us to skip a lot of computation, which
          // suggests we probably don't want to embed the sigmoid/exp postprocessing steps into the model
          if (objectnessLogit < logitMinForResult)
            continue;
          highestObjectnessLogitSeen = Math.max(highestObjectnessLogitSeen, objectnessLogit);
          todo("do we always want to perform sigmoid here, i.e., is it in input or an output?");
          float objectnessConfidence = NetworkUtil.sigmoid(objectnessLogit);

          if (verbose()) {
            sbLine.setLength(0);
            sbLine.append(String.format("y%2d  x%2d  ", cellY, cellX));
            if (anchorBoxCount > 1)
              sbLine.append(String.format("a%d  ", anchorBox));
            sbLine.append(String.format("%5.1f  ", pct(objectnessConfidence)));
          }

          IPoint anchorBoxPixels = mYolo.anchorBoxesPixels().get(anchorBox);
          float anchorBoxWidth = anchorBoxPixels.x;
          float anchorBoxHeight = anchorBoxPixels.y;

          CategoryConfidence bestCategory = CategoryConfidence.DEFAULT_INSTANCE;
          if (categoryCount > 1) {
            int k = fieldSetIndex + F_CLASS_PROBABILITIES;
            categoryConfidences.clear();
            for (int i = 0; i < categoryCount; i++)
              categoryConfidences
                  .add(CategoryConfidence.newBuilder().category(i).confidenceLogit(f[k + i]).build());
            Collections.sort(categoryConfidences,
                (a, b) -> -Float.compare(a.confidenceLogit(), b.confidenceLogit()));
            bestCategory = categoryConfidences.get(0);
            if (verbose()) {
              CategoryConfidence second = categoryConfidences.get(1);
              sbLine.append(toString(bestCategory));
              sbLine.append(toString(second));
              sbLine.append(" ");
            }
            if (cellString == 0)
              cellString = (char) ('0' + bestCategory.category());
          }

          int k = fieldSetIndex + F_BOX_XYWH;
          if (verbose()) {
            sbLine
                .append(String.format("logt[%5.1f %5.1f %5.1f %5.1f]  ", f[k], f[k + 1], f[k + 2], f[k + 3]));
          }

          // Note that the x,y (centerpoints) are relative to the cell,
          // while the width and height are relative to the anchor box size

          float bx = NetworkUtil.sigmoid(f[k + 0]);
          float by = NetworkUtil.sigmoid(f[k + 1]);
          float ws = NetworkUtil.exp(f[k + 2]);
          float hs = NetworkUtil.exp(f[k + 3]);

          todo("but can we precalculate the training labels to save some calc?");

          if (verbose()) {
            sbLine.append(String.format("sg/ex[%4.2f %4.2f %4.2f %4.2f]  ", bx, by, ws, hs));
          }

          float bw = anchorBoxWidth * ws;
          float bh = anchorBoxHeight * hs;
          float midpointX = (bx + cellX) * mGridToImageScale.x;
          float midpointY = (by + cellY) * mGridToImageScale.y;

          IRect boxRect = new FRect(midpointX - bw / 2, midpointY - bh / 2, bw, bh).toIRect();

          // I think we want to use the 'objectness' confidence, without incorporating the best category's confidence in any way
          ElementProperties.Builder prop = ElementProperties.newBuilder();
          prop.category(bestCategory.category());
          prop.confidence(MyMath.parameterToPercentage(objectnessConfidence));
          // todo: add support anchor box property? 
          // prop.anchor(anchorBox)
          RectElement boxObj = new RectElement(prop, boxRect);

          if (verbose()) {
            sbLine.append(boxObj.bounds());
            log(sbLine.toString());
          }
          boxList.add(boxObj);
        }

        if (verbose()) {
          if (cellString == 0)
            cellString = ':';
          sbGrid.append(' ');
          sbGrid.append(cellString);
        }
      }
      if (verbose()) {
        sbGrid.append("]");
        gridMap.put(String.format("y %2d", cellY), sbGrid.toString());
      }
    }

    if (verbose()) {
      pr();
      pr(gridMap);
      pr();
      pr("Valid anchor boxes:", boxList.size());
      todo("do we always want to perform sigmoid here, i.e., is it in input or an output?");
      pr("Highest conf:", pct(NetworkUtil.sigmoid(highestObjectnessLogitSeen)));
      if (boxList.isEmpty())
        pr("*** No boxes detected");
    }

    return boxList;
  }

  private static float pct(float confidence) {
    return confidence * 100;
  }

  private static String toString(CategoryConfidence b) {
    return String.format("'%d'(%4.1f) ", b.category(), b.confidenceLogit());
  }

  private final Yolo mYolo;
  private final IPoint mGridSize;
  private float mConfidenceThreshold = 0.8f;
  private FPoint mGridToImageScale;
}
