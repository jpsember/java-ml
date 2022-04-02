package ml;

import static js.base.Tools.*;

import gen.CompileImagesConfig;
import js.app.AppOper;

public class CompileImagesOper extends AppOper {

  @Override
  public String userCommand() {
    return "compileimages";
  }

  @Override
  public String getHelpDescription() {
    return "Compile sets of training or testing images";
  }

  @Override
  public void perform() {

    ImageCompiler c = new ImageCompiler(config());
    c.setFiles(files());
    c.compileTrainSet(config().targetDirTrain());
    c.compileTestSet(config().targetDirTest());
    todo("Option to support streaming service");
  }

  @Override
  public CompileImagesConfig defaultArgs() {
    return CompileImagesConfig.DEFAULT_INSTANCE;
  }

  @SuppressWarnings("unchecked")
  @Override
  public CompileImagesConfig config() {
    return super.config();
  }

}
