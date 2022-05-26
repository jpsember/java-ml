package ml;

import static js.base.Tools.*;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.InputStream;

import gen.EvalModelConfig;
import gen.ImageSetInfo;
import js.app.AppOper;
import js.file.Files;
import js.graphics.ImgUtil;
import js.graphics.ScriptUtil;
import js.graphics.gen.Script;

public class EvalModelOper extends AppOper {

  @Override
  public String userCommand() {
    return "evalmodel";
  }

  @Override
  public String getHelpDescription() {
    return "Evaluate training results by plotting training snapshot";
  }

  private int calcRecordsInFile(File file, int recordLength) {
    Files.assertExists(file);
    long fileLen = file.length();
    int numRecords = (int) (fileLen / recordLength);
    if (numRecords * recordLength != fileLen)
      setError("Unexpected file length:", fileLen, "for record length", recordLength, INDENT, file);
    return numRecords;
  }

  @Override
  public void perform() {
    mModel = ModelWrapper.constructFor(config().network(), config().networkPath());
    ModelWrapper model = mModel;

    // Read and parse label information

    ImageSetInfo.Builder infoBuilder = ImageSetInfo.newBuilder();
    model.storeImageSetInfo(infoBuilder);
    ImageSetInfo imageSetInfo = infoBuilder.build();
    File imagesPath = Files.join(config().trainTestDir(), "images.bin");
    File labelsPath = Files.join(config().trainTestDir(), "results.bin");

    int numImages = calcRecordsInFile(imagesPath, imageSetInfo.imageLengthBytes());
    int numLabels = calcRecordsInFile(labelsPath, imageSetInfo.labelLengthBytes());
    checkArgument(numImages == numLabels, "number of images != number of labels");

    InputStream imagesStream = Files.openInputStream(imagesPath);
    InputStream labelsStream = Files.openInputStream(labelsPath);

    File targetDir = files().remakeDirs(config().evalDir());
    files().mkdirs(ScriptUtil.scriptDirForProject(targetDir));

    for (int imageNumber = 0; imageNumber < numImages; imageNumber++) {
      float[] imageFloats = Files.readFloatsLittleEndian(imagesStream,
          imageSetInfo.imageLengthBytes() / Float.BYTES);
      byte[] labelBytes = Files.readBytes(labelsStream, imageSetInfo.labelLengthBytes());
      Script.Builder script = Script.newBuilder();
      model.parseInferenceResult(labelBytes, 80, script);
      Script s = script.build();

      BufferedImage bufferedImage = ImgUtil.floatsToBufferedImage(imageFloats, model.inputImagePlanarSize(),
          model.inputImageChannels());
      File outputImage = new File(targetDir, String.format("%03d.jpg", imageNumber));
      ImgUtil.writeImage(files(), bufferedImage, outputImage);
      if (ScriptUtil.isUseful(s)) {
        ScriptUtil.writeIfUseful(files(), s.build(), ScriptUtil.scriptPathForImage(outputImage));
      }
    }
  }

  @Override
  public EvalModelConfig defaultArgs() {
    return EvalModelConfig.DEFAULT_INSTANCE;
  }

  @Override
  public EvalModelConfig config() {
    return super.config();
  }

  private ModelWrapper mModel;

}
