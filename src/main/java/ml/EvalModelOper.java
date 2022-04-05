package ml;

import static js.base.Tools.*;

import java.io.File;
import java.io.InputStream;

import gen.EvalModelConfig;
import gen.ImageSetInfo;
import js.app.AppOper;
import js.file.Files;
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
    mModelHandler = NetworkUtil.constructModelHandler(null, config().network(), config().networkPath());

    ModelWrapper model = modelHandler().model();

    // Read and parse label information

    ImageSetInfo.Builder infoBuilder = ImageSetInfo.newBuilder();
    ModelInputReceiver modelService = modelHandler().buildModelInputReceiver(null, null);
    modelService.storeImageSetInfo(model, infoBuilder);
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
      byte[] imageBytes = Files.readBytes(imagesStream, imageSetInfo.imageLengthBytes());
      byte[] labelBytes = Files.readBytes(labelsStream, imageSetInfo.labelLengthBytes());
      Script.Builder script = Script.newBuilder();
      modelService.parseInferenceResult(labelBytes, script);
      Script s = script.build();
      
      File outputImage = new File(targetDir, String.format("%03d.jpg",imageNumber));
      modelService.decompileImage(imageBytes, files(), outputImage);
      if (ScriptUtil.isUseful(s)) {
        ScriptUtil.writeIfUseful(files(), s.build(), ScriptUtil.scriptPathForImage(outputImage));
      }
    }

   }

  @Override
  public EvalModelConfig defaultArgs() {
    return EvalModelConfig.DEFAULT_INSTANCE;
  }

  @SuppressWarnings("unchecked")
  @Override
  public EvalModelConfig config() {
    return super.config();
  }

  private ModelHandler modelHandler() {
    return mModelHandler;
  }

  private ModelHandler mModelHandler;

}
