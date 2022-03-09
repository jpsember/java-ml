package ml;

import js.app.App;

import static js.base.Tools.*;

public class Main extends App {

  public static final String VERSION = "1.0";

  public static void main(String[] args) {
    loadTools();
    App app = new Main();
    app.startApplication(args);
    app.exitWithReturnCode();
  }

  @Override
  public String getVersion() {
    return VERSION;
  }

  @Override
  protected void registerOperations() {
    registerOper(new GenerateImageSetOper());
    registerOper(new GenerateTrainingSetOper());
  }

}
