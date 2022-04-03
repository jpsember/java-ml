package ml;

import static js.base.Tools.*;

import java.io.File;
import java.util.List;

import gen.Layer;
import gen.LayerType;
import gen.DescribeNetworkConfig;
import gen.NeuralNetwork;
import js.file.Files;
import js.app.AppOper;
import js.base.BasePrinter;
import js.data.DataUtil;
import ml.VolumeUtil;

public class DescribeNetworkOper extends AppOper {

  @Override
  public String userCommand() {
    return "network";
  }

  @Override
  public String getHelpDescription() {
    return "describe convolutional neural network architecture";
  }

  @Override
  public DescribeNetworkConfig defaultArgs() {
    return DescribeNetworkConfig.DEFAULT_INSTANCE;
  }

  private static final int MAX_COL = 4;

  @Override
  public void perform() {
    mConfig = config();

    NetworkAnalyzer an = mNetworkAnalyzer = NetworkAnalyzer.build(architecture());

    generateHeader();

    col(1);
    add("Input image");
    col(3);
    add(VolumeUtil.toString(an.layer(0).inputVolume()));

    auxPerform();

    if (an.problemsFound()) {
      generateDashes();
      for (String problem : an.problemList())
        addAlert(problem);
    }

    dump();
  }

  private void auxPerform() {
    NetworkAnalyzer an = mNetworkAnalyzer;

    for (Layer layer : an.result().layers()) {
      col(0);
      add(VolumeUtil.toString(layer.inputVolume()));

      StringBuilder sb = new StringBuilder();
      sb.append(layer.type().toString());
      sb.append(' ');

      switch (layer.type()) {

      default:
        an.handler().describeLayer(an, layer, sb);
        break;

      case CONV:
        sb.append("(" + layer.kernelWidth() + " x " + layer.kernelWidth() + ")");
        sb.append(" filters:" + layer.filters());
        appendDropout(sb, layer.dropout());
        if (layer.pool())
          addPoolStride(sb, layer);
        break;

      case LEAKY_RELU:
        sb.append("alpha:" + layer.alpha());
        break;

      case MAXPOOL:
        addPoolStride(sb, layer);
        break;

      case FC:
        sb.append(" filters:" + layer.filters());
        appendDropout(sb, layer.dropout());
        break;

      case OUTPUT:
        break;
      }

      add(sb);

      if (layer.numWeights() != 0)
        displayVarSize(layer.numWeights());

      col(3);
      if (layer.type() != LayerType.OUTPUT)
        add(VolumeUtil.toString(layer.outputVolume()));
    }

    generateDashes();

    long varSizeBytes = displayVarSize(an.weightCount(), true);

    if (networkArgs().maxSizeMb() != 0) {
      if (varSizeBytes > networkArgs().maxSizeMb() * DataUtil.ONE_MB)
        an.addProblem("Model size", varSizeBytes, "exceeds maximum", networkArgs().maxSizeMb(), "Mb");
    }
  }

  private void addPoolStride(StringBuilder sb, Layer layer) {
    sb.append(" stride:");
    sb.append(layer.strideX());
    if (layer.strideY() != layer.strideX()) {
      sb.append(",");
      sb.append(layer.strideY());
    }
  }

  private static void appendDropout(StringBuilder sb, float dropout) {
    if (dropout > 0)
      sb.append(" dropout:" + String.format("%4.2f", dropout));
  }

  private long displayVarSize(long vars) {
    return displayVarSize(vars, false);
  }

  private long displayVarSize(long vars, boolean convertToBytes) {
    long size = vars;
    if (convertToBytes)
      size *= Float.BYTES;
    col(2);
    String x = String.format("%,d", size);
    if (convertToBytes)
      x = "Bytes: " + x;
    add(x);
    return size;
  }

  private static String[] sDashes = { "------------------------", //
      "---------------------------------", //
      "--------------------", //
      "------------------------",//
  };

  private void dash() {
    add(sDashes[mColumn]);
  }

  private void generateHeader() {
    add("Input vol");
    add("Layer");
    add("Weights");
    add("Output vol");
    generateDashes();
  }

  private void generateDashes() {
    col(0);
    for (int i = 0; i < MAX_COL; i++)
      dash();
  }

  private DescribeNetworkConfig networkArgs() {
    return mConfig;
  }

  private void col(int column) {
    while (mColumn != column) {
      add("");
    }
  }

  private void add(Object message) {
    mFields.add(message.toString());
    mColumn++;
    if (mColumn == MAX_COL) {
      mColumn = 0;
      mRow++;
    }
  }

  /**
   * Display the table of information
   */
  private void dump() {
    // Finish off last row if necessary
    col(0);

    int numRows = mRow;
    // Determine maximum width of each column
    int[] colWidth = new int[MAX_COL];
    {
      int i = INIT_INDEX;
      for (String x : mFields) {
        i++;
        int col = i % MAX_COL;
        colWidth[col] = Math.max(colWidth[col], x.length());
      }
    }

    StringBuilder sb = new StringBuilder();

    for (int row = 0; row < numRows; row++) {
      for (int col = 0; col < MAX_COL; col++) {

        String text = mFields.get(row * MAX_COL + col);

        if (col > 0) {
          final int gapBetweenColumns = 2;
          sb.append(spaces(gapBetweenColumns));
        }

        boolean rightJustified = (col != 1);
        String justifySpacing = spaces(colWidth[col] - text.length());
        if (rightJustified) {
          sb.append(justifySpacing);
          sb.append(text);
        } else {
          sb.append(text);
          sb.append(justifySpacing);
        }

      }
      sb.append('\n');
    }
    pr(sb.toString());
    if (!mMessages.isEmpty()) {
      pr();
      for (String message : mMessages)
        pr(message);
    }
  }

  private void addAlert(Object... messages) {
    mMessages.add("*** " + BasePrinter.toString(messages));
  }

  private NeuralNetwork architecture() {
    if (mArchitecture == null) {
      File path = mConfig.path();
      if (Files.empty(path))
        throw setError("No network_path defined");
      mArchitecture = NetworkUtil.resolveNetwork(null, null, path);
    }
    return mArchitecture;
  }

  private NeuralNetwork mArchitecture;
  private DescribeNetworkConfig mConfig;
  private List<String> mFields = arrayList();
  private int mRow, mColumn;
  private List<String> mMessages = arrayList();
  private NetworkAnalyzer mNetworkAnalyzer;
}
