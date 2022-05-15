package ml;

import static js.base.Tools.*;

import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.SortedMap;

import gen.FeedConfig;
import gen.FeedEntry;
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
    int maxEvents = 200;

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

      if (mLog.length() > 1000)
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
    if (mItems.size() == config().produceSetSize()) {
      pushEvent(EVT_PRODUCER, 0);
      return;
    }

    FeedEntry.Builder ent = FeedEntry.newBuilder().id(mNextIdProduced++);
    mItems.put(ent.id(), ent);
    addEvent("producer creating", summary(ent));

    pushEvent(EVT_PRODUCER, config().produceTimeMs());
  }

  private void updateConsumer() {
    List<FeedEntry.Builder> ents = getEntries();

    if (mWorkList.isEmpty()) {
      while (mWorkList.size() < config().consumeSetSize()) {
        mWorkList.add(FeedEntry.newBuilder());
      }
    }

    FeedEntry.Builder currEnt = mWorkList.get(mCursor);
    addEvent("cursor:", mCursor, "entry:", summary(currEnt));
    if (undefined(currEnt)) {
      addEvent("looking for an inactive entry to fill slot, from set of size", mItems.size(), "ents size:",
          ents.size());
      for (FeedEntry.Builder ent : ents) {
        if (!feedEntryActive(ent.id())) {
          // ent.active(true);
          mWorkList.add(ent);
          addEvent("making active:", summary(ent));
          currEnt = ent;
          break;
        }
      }
    }

    long nextDelay = 100;
    if (undefined(currEnt)) {
      addEvent("consumer stalled");
    } else {
      currEnt.used(currEnt.used() + 1);
      mCursor = (1 + mCursor) % config().consumeSetSize();
      nextDelay = config().consumeTimeMs();
      addEvent("consumer updating", summary(currEnt));
    }
    pushEvent(EVT_CONSUMER, nextDelay);
  }

  private String summary(FeedEntry ent) {
    if (ent.id() == 0)
      return "---";
    return String.format("%4d (%d)", ent.id(), ent.used());
  }

  private void addEvent(Object... msgs) {
    String s = BasePrinter.toString(msgs);
    mLog.append(String.format("%6d: %s\n", mCurrentTime, s));
  }

  private boolean undefined(FeedEntry ent) {
    return !isDefined(ent);
  }

  private boolean isDefined(FeedEntry ent) {
    return ent != null && ent.id() != 0;
  }

  private boolean feedEntryActive(int id) {
    for (FeedEntry ent : mWorkList)
      if (ent.id() == id)
        return true;
    return false;
  }

  private List<FeedEntry.Builder> getEntries() {
    List<FeedEntry.Builder> ents = arrayList();
    ents.addAll(mItems.values());
    return ents;
  }

  private static final int EVT_PRODUCER = 1;
  private static final int EVT_CONSUMER = 2;

  private SortedMap<Long, Integer> mQueue = treeMap();

  // Items currently in existence
  //
  private Map<Integer, FeedEntry.Builder> mItems = hashMap();

  private long mCurrentTime;
  private Random mRandom;

  // Items being tracked by consumer; id is zero if slot is vacant
  //
  private List<FeedEntry.Builder> mWorkList = arrayList();
  
  private int mCursor = 0;
  private StringBuilder mLog = new StringBuilder();
  private int mNextIdProduced = 500;
}
