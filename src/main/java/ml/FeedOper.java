package ml;

import static js.base.Tools.*;

import java.util.List;
import java.util.Random;
import java.util.SortedMap;

import gen.FeedConfig;
import js.app.AppOper;
import js.base.BasePrinter;

public class FeedOper extends AppOper {

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
    int maxEvents = 500;

    pushEvent(EVT_PRODUCER, 0);
    pushEvent(EVT_CONSUMER, 0);

    for (int evtCount = 0; evtCount < maxEvents; evtCount++) {
      long time = mQueue.firstKey();
      int code = mQueue.get(time);
      mQueue.remove(time);
      mCurrentTime = time;
      if (code == EVT_PRODUCER)
        updateProducer();
      else
        updateConsumer();

      if (false && mLog.length() > 5000)
        break;
    }

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

  private void pushEvent(int code, long delay) {
    delay += rand(50);
    long targTime = mCurrentTime + delay;
    int mod = (int) (targTime & 1);
    int targMod = code & 1;
    if (mod != targMod)
      targTime++;
    mQueue.put(targTime, code);
  }

  private int rand(int range) {
    if (mRandom == null)
      mRandom = new Random(config().seed());
    return mRandom.nextInt(range);
  }

  private void updateProducer() {
    mActor = "producer";
    long delay = 0;
    if (mItems.size() < config().produceSetSize()) {
      Obj ent = new Obj(mNextIdProduced++);
      mItems.add(ent);
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
    for (Obj obj : mItems) {
      if (obj.used < config().objMaxUse())
        filtered.add(obj);
      else {
        out("discarding " + obj.id);
        discardConsumerObj(obj.id);
      }
    }
    mItems.clear();
    mItems.addAll(filtered);
  }

  private void showConsumerList() {

    StringBuilder sb = new StringBuilder();
    int i = INIT_INDEX;
    for (Obj obj : mConsumerList) {
      i++;
      sb.append(String.format("%c%s ", i == mCursor ? '*' : ' ', (summary(obj))));
    }
    out(sb.toString());
  }

  private void updateConsumer() {
    mActor = "consumer";

    // Ensure consumer list has objects, even if they are null ones
    while (mConsumerList.size() < config().consumeSetSize())
      mConsumerList.add(NULL_ENTRY);

    showConsumerList();
    discardStaleConsumerObjects();

    Obj currEnt = mConsumerList.get(mCursor);

    if (undefined(currEnt)) {
      for (Obj ent : mItems) {
        if (!feedEntryActive(ent.id)) {
          mConsumerList.set(mCursor, ent);
          out("activating " + ent.id);
          currEnt = ent;
          break;
        }
      }
    }

    long nextDelay = 100;
    if (undefined(currEnt)) {
      out("stalled");
    } else {
      currEnt.used++;
      mCursor = (1 + mCursor) % config().consumeSetSize();
      nextDelay = config().consumeTimeMs();
      out("updating " + currEnt.id);
      showConsumerList();
    }
    pushEvent(EVT_CONSUMER, nextDelay);
  }

  private void discardConsumerObj(int id) {
    for (int i = 0; i < mConsumerList.size(); i++)
      if (mConsumerList.get(i).id == id)
        mConsumerList.set(i, NULL_ENTRY);
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

  private void out(Object... msgs) {
    if (mCurrentTime != mPrevTime) {
      mLog.append("_\n");
    }
    mPrevTime = mCurrentTime;

    String s = BasePrinter.toString(msgs);
    mLog.append(String.format("%6d: (%s)  %s\n", mCurrentTime, mActor, s));
  }

  private boolean undefined(Obj ent) {
    return !isDefined(ent);
  }

  private boolean isDefined(Obj ent) {
    return ent != null && ent.id != 0;
  }

  private boolean feedEntryActive(int id) {
    for (Obj ent : mConsumerList)
      if (ent.id == id)
        return true;
    return false;
  }

  private static final int EVT_PRODUCER = 1;
  private static final int EVT_CONSUMER = 2;

  private static class Obj {
    public Obj(int id) {
      this.id = id;
    }

    int id;
    int used;
  }

  private static final Obj NULL_ENTRY = new Obj(0);

  private SortedMap<Long, Integer> mQueue = treeMap();

  // Items currently in existence
  //
  private List<Obj> mItems = arrayList();

  //  private Map<Integer, Obj> mItems = hashMap();

  private long mCurrentTime;
  private Random mRandom;

  // Items being tracked by consumer; id is zero if slot is vacant
  //
  private List<Obj> mConsumerList = arrayList();

  private int mCursor = 0;
  private StringBuilder mLog = new StringBuilder();
  private int mNextIdProduced = 500;
  private String mActor = "!!unknown!!";
  private long mPrevTime;
}
