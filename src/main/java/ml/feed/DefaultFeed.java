package ml.feed;

import static js.base.Tools.*;

import js.geometry.MyMath;

public   class DefaultFeed implements FeedInterface {

  
  public void setFeed(FeedOper oper) {
  mOper = oper;
  }
  private FeedOper mOper;

  @Override
  public void performConsumerLogic() {

    Obj cursorObject = objAtCursor();

    if (cursorObject.isNull()) {
      Obj obj = findUnclaimedObj();
      if (obj.defined())
        cursorObject = claimObj(obj);
    }

    if (cursorObject.isNull()) {
      stalled("no obj avail");
      return;
    }

    if (isLastConsumed(cursorObject)) {
      stalled("same as last");
      return;
    }

    consume(cursorObject);
    setCursor(mOper.mCursor + 1);
  }

  @Override
  public boolean isLastConsumed(Obj obj) {
    checkArgument(obj.defined());
    return obj.id == mOper.mLastConsumerIdProcessed;
  }

  @Override
  public Obj claimObj(Obj obj) {
    checkArgument(obj.defined());
    mOper. mConsumerObjList.set(mOper.mCursor, obj);
     out("claiming " + obj.id);
    return obj;
  }

  @Override
  public Obj objAtCursor() {
    return mOper.mConsumerObjList.get(mOper.mCursor);
  }

  @Override
  public void setCursor(int value) {
    mOper.mCursor = MyMath.myMod(value,mOper. config().consumeSetSize());
  }

  @Override
  public void consume(Obj obj, Object... msgs) {
    checkArgument(obj.defined());
    obj.used++;
    checkState(obj.used <= mOper.config().objMaxUse());
   mOper. setConsumerStatus(mOper.STATUS_CONSUMED, msgs);
    mOper. mLastConsumerIdProcessed = obj.id;
    mOper. out("updating " + obj.id);
    mOper.outConsumerObj();
  }

  @Override
  public void stalled(Object... msgs) {
   mOper. setConsumerStatus(mOper.STATUS_STALLED, msgs);
    mOper. mStalledCount++;
  }

  /**
   * Attempt to find an object in the active object list that does not appear in
   * the consumer list. Returns Obj.NULL if none found
   */
  @Override
  public Obj findUnclaimedObj() {
    for (Obj obj : mOper.mActiveObjList) {
      if (!inConsumerObjList(obj)) {
        return obj;
      }
    }
    return Obj.NULL;
  }

  @Override
  public boolean inConsumerObjList(Obj obj) {
     for (Obj ent : mOper.mConsumerObjList)
       if (ent.id == obj.id)
         return true;
     return false;
   }
  protected void out(Object... msgs) {
    mOper.out(msgs);
  }
  
}
