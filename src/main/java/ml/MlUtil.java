package ml;

import java.util.Random;

public final class MlUtil {

  /**
   * Construct a Random instance
   * 
   * @param seed
   *          if zero, derives from current epoch time
   */
  public static Random buildRandom(int seed) {
    if (seed == 0)
      seed = (int) System.currentTimeMillis();
    return new Random(seed);
  }

}
