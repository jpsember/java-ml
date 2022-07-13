package ml;

import js.file.Files;

import static js.base.Tools.*;

import java.io.BufferedWriter;
import java.io.Closeable;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;

import gen.CompileImagesConfig;

public final class ProgressFile implements Closeable {

  public ProgressFile(CompileImagesConfig config) {
    loadTools();
    mConfig = config;
  }

  /**
   * Write a string to the progress file. Thread safe
   */
  public synchronized void write(String content) {
    writer().println(content);
  }

  public synchronized void flush() {
    writer().flush();
  }

  @Override
  public synchronized void close() {
    if (mProgressFile == null)
      return;
    try {
      Files.close(writer());
    } finally {
      mProgressFile = null;
      mPrintWriter = null;
    }
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

  private final CompileImagesConfig mConfig;
  private File mProgressFile;
  private PrintWriter mPrintWriter;

}
