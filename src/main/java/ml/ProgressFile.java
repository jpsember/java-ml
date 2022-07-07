package ml;

import js.file.Files;
import js.json.JSMap;

import static js.base.Tools.*;

import java.io.BufferedWriter;
import java.io.Closeable;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;

import gen.CompileImagesConfig;

public final class ProgressFile implements Closeable {

  public ProgressFile(CompileImagesConfig config) {
    mConfig = config;
  }

  /**
   * Write a string to the progress file
   */
  public void write(String content) {
    writer().println(content);
  }

  public void flush() {
    writer().flush();
  }

  @Override
  public void close() {
    if (mProgressFile == null)
      return;
    try {
      Files.close(writer());
    } finally {
      mProgressFile = null;
      mPrintWriter = null;
    }
  }

  public void displayMessage(JSMap m) {
    String content = m.get("text");
    write(content);
    pr(content);
    flush();
  }

  private void append(Object msg) {
    sb().append(msg);
  }

  /* private */ void append(String msg, int minWidth) {
    int padding = minWidth - msg.length();
    if (padding > 0)
      append(spaces(padding));
    append(msg);
  }

  private File file() {
    if (mProgressFile == null)
      mProgressFile = Files.assertNonEmpty(mConfig.progressFile(), "progress_file");
    return mProgressFile;
  }

  private PrintWriter writer() {
    if (mPrintWriter == null) {
      try {
        mPrintWriter = new PrintWriter(new BufferedWriter(new FileWriter(file(), true)));
      } catch (Throwable t) {
        throw Files.asFileException(t);
      }
    }
    return mPrintWriter;
  }

  private StringBuilder sb() {
    return mStringBuilder;
  }

  /* private */ final CompileImagesConfig mConfig;
  private File mProgressFile;
  private PrintWriter mPrintWriter;
  private final StringBuilder mStringBuilder = new StringBuilder();

}
