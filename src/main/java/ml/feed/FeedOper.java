package ml.feed;

import static js.base.Tools.*;

import java.util.List;
import java.util.Random;
import java.util.SortedMap;

import gen.FeedConfig;
import js.app.AppOper;
import js.base.BasePrinter;
import js.geometry.MyMath;

public class FeedOper extends AppOper implements FeedInterface {

  @Override
  public String userCommand() {
    loadTools();
    return "feed";
  }

  @Override
  public String getHelpDescription() {
    return "Investigate strategies for feeding training data to Python code";
  }

  @Override
  public void perform() {

    mFeedInterface = new DefaultFeed();
    mFeedInterface.setFeed(this);
    
    // Ensure consumer has objects, even if they are null ones
    //
    while (mConsumerObjList.size() < config().consumeSetSize())
      mConsumerObjList.add(Obj.NULL);

    pushEvent(EVT_PRODUCER, 0);
    pushEvent(EVT_CONSUMER, 0);

    for (int evtCount = 0; mConsumerLogicCount - mStalledCount < config().objConsumedTotal(); evtCount++) {
      checkState(evtCount < 20000);

      long time = mEventQueue.firstKey();
      int code = mEventQueue.get(time);
      mEventQueue.remove(time);
      mCurrentTimestamp = time;
      if (code == EVT_PRODUCER) {
        mActor = "producer";
        updateProducer();
      } else {
        mActor = "consumer";
        updateConsumer();
      }
    }

    out("Consumer:", mConsumerEventLog);
    out("Efficiency %:", (100 * (mConsumerLogicCount - mStalledCount)) / mConsumerLogicCount);

    pr(mLog);
  }

  @Override
  public FeedConfig defaultArgs() {
    return FeedConfig.DEFAULT_INSTANCE;
  }

  @Override
  public FeedConfig config() {
    return super.config();
  }

  private void updateProducer() {
    long delay = 0;
    if (mActiveObjList.size() < config().produceSetSize()) {
      Obj ent = new Obj(mNextIdProduced++);
      mActiveObjList.add(ent);
      out("creating " + ent.id);
      delay = config().produceTimeMs();
    }
    pushEvent(EVT_PRODUCER, delay);
  }

  /**
   * Discard objects that have been used the max number of times
   */
  private void discardStaleConsumerObjects() {
    List<Obj> filtered = arrayList();
    for (Obj obj : mActiveObjList) {
      if (obj.used < config().objMaxUse())
        filtered.add(obj);
      else {
        out("discarding " + obj.id);
        discardConsumerObj(obj.id);
      }
    }
    mActiveObjList.clear();
    mActiveObjList.addAll(filtered);
  }

  /**
   * Output summary of consumer's objects
   */
    void outConsumerObj() {
    StringBuilder sb = new StringBuilder();
    int i = INIT_INDEX;
    for (Obj obj : mConsumerObjList) {
      i++;
      sb.append(String.format("%c%s ", i == mCursor ? '*' : ' ', (summary(obj))));
    }
    out(sb.toString());
  }

  /**
   * Update the consumer's logic
   */
  private void updateConsumer() {
    prepareConsumerVars();
    outConsumerObj();
    discardStaleConsumerObjects();
   mFeedInterface. performConsumerLogic();
    checkState(mStatus != STATUS_NONE);
    scheduleNextConsumerEvent();
  }

  private void scheduleNextConsumerEvent() {
    pushEvent(EVT_CONSUMER, mStatus == STATUS_CONSUMED ? config().consumeTimeMs() : 100);
  }

  private void prepareConsumerVars() {
    mStatus = 0;
  }

  private void discardConsumerObj(int id) {
    for (int i = 0; i < mConsumerObjList.size(); i++)
      if (mConsumerObjList.get(i).id == id)
        mConsumerObjList.set(i, Obj.NULL);
  }

  private String summary(Obj ent) {
    StringBuilder sb = new StringBuilder();
    if (ent.id == 0) {
      sb.append("----");
      sb.append(spaces(1 + config().objMaxUse()));
    } else {
      sb.append(String.format("%3d", ent.id));

      sb.append(':');
      for (int i = 1; i <= config().objMaxUse(); i++) {
        sb.append(ent.used >= i ? '▆' : '▁');
      }
    }
    return sb.toString();
  }

    void out(Object... msgs) {
    if (mCurrentTimestamp != mPrevLogTime) {
      mLog.append("_\n");
    }
    mPrevLogTime = mCurrentTimestamp;

    String s = BasePrinter.toString(msgs);
    mLog.append(String.format("%6d: (%s)  %s\n", mCurrentTimestamp, mActor, s));
  }

  // ------------------------------------------------------------------
  // Event queue, sorted by timestamp
  // ------------------------------------------------------------------

  /**
   * Schedule another producer or consumer logic update event
   */
  private void pushEvent(int code, long delay) {
    delay += rand().nextInt(50);
    long targTime = mCurrentTimestamp + delay;
    int mod = (int) (targTime & 1);
    int targMod = code & 1;
    if (mod != targMod)
      targTime++;
    mEventQueue.put(targTime, code);
  }

  private Random rand() {
    if (mRandom == null)
      mRandom = new Random(config().seed());
    return mRandom;
  }

  private Random mRandom;

  // Current time being processed
  //
  private long mCurrentTimestamp;

  /**
   * Event codes
   */
  private static final int EVT_PRODUCER = 1;
  private static final int EVT_CONSUMER = 2;

  private SortedMap<Long, Integer> mEventQueue = treeMap();

  // Items generated by producer that haven't yet been deleted
  //
    List<Obj> mActiveObjList = arrayList();

  // ------------------------------------------------------------------

  // Items being tracked by consumer; if slot is vacant, it will contain the NULL obj
  //
    List<Obj> mConsumerObjList = arrayList();

    int mCursor;
  private StringBuilder mLog = new StringBuilder();
  private int mNextIdProduced = 500;

  // Name of entity being updated (PRODUCER, CONSUMER)
  //
  private String mActor = "!!unknown!!";

  private long mPrevLogTime;

  // ------------------------------------------------------------------
  // Consumer logic 
  // ------------------------------------------------------------------

  /**
   * Attempt to find an object in the active object list that does not appear in
   * the consumer list. Returns Obj.NULL if none found
   */
  @Override
  public Obj findUnclaimedObj() {
    for (Obj obj : mActiveObjList) {
      if (!inConsumerObjList(obj)) {
        return obj;
      }
    }
    return Obj.NULL;
  }

 @Override
 public boolean inConsumerObjList(Obj obj) {
    for (Obj ent : mConsumerObjList)
      if (ent.id == obj.id)
        return true;
    return false;
  }

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
    setCursor(mCursor + 1);
  }

  @Override
  public boolean isLastConsumed(Obj obj) {
    checkArgument(obj.defined());
    return obj.id == mLastConsumerIdProcessed;
  }

  @Override
  public Obj claimObj(Obj obj) {
    checkArgument(obj.defined());
    mConsumerObjList.set(mCursor, obj);
    out("claiming " + obj.id);
    return obj;
  }

  @Override
  public Obj objAtCursor() {
    return mConsumerObjList.get(mCursor);
  }

  @Override
  public void setCursor(int value) {
    mCursor = MyMath.myMod(value, config().consumeSetSize());
  }

  @Override
  public void consume(Obj obj, Object... msgs) {
    checkArgument(obj.defined());
    obj.used++;
    checkState(obj.used <= config().objMaxUse());
    setConsumerStatus(STATUS_CONSUMED, msgs);
    mLastConsumerIdProcessed = obj.id;
    out("updating " + obj.id);
    outConsumerObj();
  }

  @Override
  public void stalled(Object... msgs) {
    setConsumerStatus(STATUS_STALLED, msgs);
    mStalledCount++;
  }

    void setConsumerStatus(int status, Object... msgs) {
    checkState(mStatus == STATUS_NONE);
    if (msgs.length != 0)
      out(msgs);
    mStatus = status;
    mConsumerLogicCount++;
    mConsumerEventLog.append(status == STATUS_STALLED ? 'X' : '█');
  }

  // Id of most recent consumer object processed
  //
    int mLastConsumerIdProcessed;

 private   DefaultFeed mFeedInterface;
  private StringBuilder mConsumerEventLog = new StringBuilder();
  private int mConsumerLogicCount;
    int mStalledCount;
  private int mStatus;
  private static final int STATUS_NONE = 0;
    static final int STATUS_STALLED = 1;
    static final int STATUS_CONSUMED = 2;
}
