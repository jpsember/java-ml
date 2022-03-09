/**
 * MIT License
 * 
 * Copyright (c) 2021 Jeff Sember
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * 
 **/
package ml;

import static js.base.Tools.*;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.OutputStream;

import gen.AnnotationFile;
import gen.GenerateTrainingSetConfig;
import js.app.AppOper;
import js.file.Files;
import js.geometry.IPoint;
import js.graphics.ImgUtil;

public class GenerateTrainingSetOper extends AppOper {

  @Override
  public String userCommand() {
    return "gentrainset";
  }

  @Override
  public String getHelpDescription() {
    return "Generate set of training images";
  }

  @Override
  public void perform() {
    OutputStream imageStream = files().outputStream(new File(targetDir(), "images.bin"));

    AnnotationFile af = Files.parseAbstractData(AnnotationFile.DEFAULT_INSTANCE,
        Files.assertExists(config().sourceImageInfo(), "source_image_info"));
    File srcImages = Files.assertDirectoryExists(config().sourceImagesDir(), "source_images_dir");

    log("Source image count:", af.filenames().size());
    int index = INIT_INDEX;
    for (String filename : af.filenames()) {
      index++;
      if (config().maxImages() != 0 && index == config().maxImages()) {
        log("...max images reached:", index);
        break;
      }
      if (verbose() && index != 0 && index % 100 == 0)
        log("...processing image #", index);
      File srcImageFile = new File(srcImages, filename);
      BufferedImage srcImage = ImgUtil.read(srcImageFile);
      IPoint size = ImgUtil.size(srcImage);
      checkArgument(size.equals(config().imageSize()), "unexpected image size");
      srcImage = ImgUtil.imageAsType(srcImage, BufferedImage.TYPE_INT_RGB);
      todo("add annotations to output");

      float[] pixels = ImgUtil.floatPixels(srcImage, 3, null);
      files().writeFloatsLittleEndian(pixels, imageStream);
    }
    Files.close(imageStream);

    //
    //    byte[] categoryBytes = DataUtil.intsToBytesLittleEndian(categories.array());
    //    files().write(categoryBytes, new File(config().targetDir(), "labels.bin"));
  }

  @Override
  public GenerateTrainingSetConfig defaultArgs() {
    return GenerateTrainingSetConfig.DEFAULT_INSTANCE;
  }

  @SuppressWarnings("unchecked")
  @Override
  public GenerateTrainingSetConfig config() {
    return super.config();
  }

  private File targetDir() {
    if (mTargetDir == null) {
      File t = Files.assertNonEmpty(config().targetDir(), "target_dir");
      mTargetDir = files().mkdirs(t);
    }
    return mTargetDir;
  }

  private File mTargetDir;
}
