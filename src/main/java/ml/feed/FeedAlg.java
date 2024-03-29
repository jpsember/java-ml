package ml.feed;

import static js.base.Tools.*;

import java.util.List;
import java.util.Random;
import java.util.SortedMap;

import gen.FeedConfig;
import js.base.BaseObject;
import js.base.BasePrinter;
import js.geometry.MyMath;
import ml.MlUtil;

public abstract class FeedAlg extends BaseObject {

  public final void perform() {

    // Ensure consumer has objects, even if they are null ones
    //
    while (mConsumerObjList.size() < config().consumeSetSize())
      mConsumerObjList.add(Obj.NULL);

    pushEvent(EVT_PRODUCER, 0);
    pushEvent(EVT_CONSUMER, 0);

    for (int evtCount = 0; consumedObjCount() < config().objConsumedTotal(); evtCount++) {
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

    out("Producer count:", mProducerTasks);
    out("Consumer:", mConsumerEventLog);
    out("Efficiency %:", (100 * consumedObjCount()) / mConsumerLogicCount);
    out("Avg dist:", mDistSum / (float) (consumedObjCount() - 1));
    pr(mLog);
  }

  public final void setConfig(FeedConfig config) {
    mConfig = config;
  }

  private FeedConfig mConfig;

  private int consumedObjCount() {
    return mConsumerLogicCount - mStalledCount;
  }

  public final FeedConfig config() {
    return mConfig;
  }

  private void updateProducer() {
    long delay = 0;
    if (mActiveObjList.size() < config().produceSetSize()) {
      Obj ent = new Obj(mNextIdProduced++);
      mActiveObjList.add(ent);
      out2("creating " + ent.id);
      delay = config().produceTimeMs();
      mProducerTasks++;
    }
    pushEvent(EVT_PRODUCER, delay);
  }

  private int mProducerTasks;

  /**
   * Discard objects that have been used the max number of times
   */
  private void discardStaleConsumerObjects() {
    List<Obj> filtered = arrayList();
    for (Obj obj : mActiveObjList) {
      if (obj.used < config().recycle())
        filtered.add(obj);
      else {
        out2("discarding " + obj.id);
        discardConsumerObj(obj.id);
      }
    }
    mActiveObjList.clear();
    mActiveObjList.addAll(filtered);
  }

  /**
   * Output summary of consumer's objects
   */
  public final void outConsumerObj() {
    StringBuilder sb = new StringBuilder();
    int i = INIT_INDEX;
    for (Obj obj : mConsumerObjList) {
      i++;
      sb.append(String.format("%c%s ", i == cursor() ? '*' : ' ', (summary(obj))));
    }
    out2(sb.toString());
  }

  /**
   * Update the consumer's logic
   */
  private void updateConsumer() {
    prepareConsumerVars();
    outConsumerObj();
    discardStaleConsumerObjects();
    updateConsumerLogic();
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
      sb.append(spaces(1 + config().recycle()));
    } else {
      sb.append(String.format("%3d", ent.id));

      sb.append(':');
      for (int i = 1; i <= config().recycle(); i++) {
        sb.append(ent.used >= i ? '▆' : '▁');
      }
    }
    return sb.toString();
  }

  public final void out2(Object... msgs) {
    if (verbose())
      out(msgs);
  }

  public final void out(Object... msgs) {
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

  public final Random rand() {
    if (mRandom == null)
      mRandom = MlUtil.buildRandom(config().seed());
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
  private List<Obj> mConsumerObjList = arrayList();

  private int mCursor;
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
  public final Obj findUnclaimedObj() {
    for (Obj obj : mActiveObjList) {
      if (!inConsumerObjList(obj)) {
        return obj;
      }
    }
    return Obj.NULL;
  }

  public final boolean inConsumerObjList(Obj obj) {
    for (Obj ent : mConsumerObjList)
      if (ent.id == obj.id)
        return true;
    return false;
  }

  public final boolean isLastConsumed(Obj obj) {
    checkArgument(obj.defined());
    return obj.id == mLastConsumerIdProcessed;
  }

  public final Obj claimObj(Obj obj) {
    checkArgument(obj.defined());
    mConsumerObjList.set(cursor(), obj);
    out2("claiming " + obj.id);
    return obj;
  }

  public final Obj objAtCursor() {
    return mConsumerObjList.get(cursor());
  }

  public final int cursor() {
    return mCursor;
  }

  public final void setCursor(int value) {
    mCursor = MyMath.myMod(value, config().consumeSetSize());
  }

  public final int distanceFromPreviousUse(Obj obj) {
    int dist = config().consumeSetSize();
    List<Integer> list = mConsumedIdsList;
    int i = list.size() - 1;
    while (i >= 0) {
      if (list.get(i) == obj.id) {
        dist = Math.min(dist, list.size() - i);
        break;
      }
      i--;
    }
    return dist;
  }

  public final void consume(Obj obj, Object... msgs) {
    checkArgument(obj.defined());
    obj.used++;
    checkState(obj.used <= config().recycle());
    mLastConsumerIdProcessed = obj.id;
    setConsumerStatus(STATUS_CONSUMED, msgs);
    out2("updating " + obj.id);
    outConsumerObj();

    // record distance from last use of this object
    {
      List<Integer> list = mConsumedIdsList;
      if (!list.isEmpty())
        mDistSum += distanceFromPreviousUse(obj);
      list.add(obj.id);
    }
  }

  private List<Integer> mConsumedIdsList = arrayList();
  private int mDistSum;

  public final void stalled(Object... msgs) {
    setConsumerStatus(STATUS_STALLED, msgs);
    mStalledCount++;
  }

  private static String sNames = "abcdefghikmjlnorpstquwxyuz";

  private void setConsumerStatus(int status, Object... msgs) {
    checkState(mStatus == STATUS_NONE);
    if (msgs.length != 0)
      out2(msgs);
    mStatus = status;
    mConsumerLogicCount++;
    mConsumerEventLog
        .append(status == STATUS_STALLED ? '█' : sNames.charAt(mLastConsumerIdProcessed % sNames.length()));
  }

  // Id of most recent consumer object processed
  //
  private int mLastConsumerIdProcessed;

  private StringBuilder mConsumerEventLog = new StringBuilder();
  private int mConsumerLogicCount;
  private int mStalledCount;
  private int mStatus;
  private static final int STATUS_NONE = 0;
  private static final int STATUS_STALLED = 1;
  private static final int STATUS_CONSUMED = 2;

  public abstract void updateConsumerLogic();

}
