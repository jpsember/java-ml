package ml.feed;

public class Alg1 extends FeedAlg {

  public void updateConsumerLogic() {

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
    setCursor(cursor() + 1);
  }
}
