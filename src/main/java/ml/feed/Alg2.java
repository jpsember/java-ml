package ml.feed;

public class Alg2 extends FeedAlg {

  public void updateConsumerLogic() {

    // Find obj with furthest distance from its last usage, 
    // filling in empty slots where possible
    int slots = config().consumeSetSize();
    int maxDist = -1;
    int bestCursor = -1;
    for (int i = 0; i < slots; i++) {
      setCursor(cursor() + 1);
      Obj cursorObject = objAtCursor();
      if (cursorObject.isNull()) {
        Obj obj = findUnclaimedObj();
        if (obj.defined())
          cursorObject = claimObj(obj);
      }
      if (cursorObject.isNull()) continue;
      // There must be at least one object between this one and the last one consumed
      if (isLastConsumed(cursorObject)) continue;
      int dist = distanceFromPreviousUse(cursorObject);
      if (dist > maxDist) {
        maxDist = dist;
        bestCursor = cursor();
      }
    }
    
    if (bestCursor < 0) {
      stalled("no different object avail");
      return;
    }
    setCursor(bestCursor);
    consume(objAtCursor());
  }
}
