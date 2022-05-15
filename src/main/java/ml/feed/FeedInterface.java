package ml.feed;

import static js.base.Tools.*;

import js.geometry.MyMath;

public interface FeedInterface {

  void performConsumerLogic();

  /**
   * Set cursor within consumer object list
   */
  void setCursor(int value);

  Obj objAtCursor();

  /**
   * Attempt to find an object in the active object list that does not appear in
   * the consumer list. Returns Obj.NULL if none found
   */
  Obj findUnclaimedObj();


  /**
   * Determine if an object is within the consumer list
   */
   boolean inConsumerObjList(Obj obj) ;
   
  Obj claimObj(Obj obj);

  void stalled(Object... msgs);

  /**
   * Determine if an object was the last one consumed
   */
  boolean isLastConsumed(Obj obj);

  /**
   * Have consumer 'use' an object; increment its usage counter
   */
  void consume(Obj obj, Object... msgs);
}
