package ml;

import static js.base.Tools.*;

import java.util.List;
import java.util.Map;
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

      if (mLog.length() > 5000)
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
    actor = "producer";
    long delay = 0;
    if (mItems.size() < config().produceSetSize()) {
      FeedEntry ent = new FeedEntry(mNextIdProduced++);
      mItems.put(ent.id, ent);
      addEvent("creating", summary(ent));
      delay = config().produceTimeMs();
    } else
      addEvent("set full");
    pushEvent(EVT_PRODUCER, delay);
  }

  private void updateConsumer() {
    actor = "consumer";

    List<FeedEntry> ents = getEntries();

    if (mWorkList.isEmpty()) {
      while (mWorkList.size() < config().consumeSetSize()) {
        mWorkList.add(NULL_ENTRY);
      }
    }

    FeedEntry currEnt = mWorkList.get(mCursor);
    addEvent("cursor:", mCursor, "entry:", summary(currEnt));
    if (undefined(currEnt)) {
      if (false)
        addEvent("looking for an inactive entry to fill slot, from set of size", mItems.size(), "ents size:",
            ents.size());
      for (FeedEntry ent : ents) {
        if (!feedEntryActive(ent.id)) {
          mWorkList.add(ent);
          addEvent("making active:", summary(ent));
          currEnt = ent;
          break;
        }
      }
    }

    long nextDelay = 100;
    if (undefined(currEnt)) {
      addEvent("stalled");
    } else {
      currEnt.used++;
      mCursor = (1 + mCursor) % config().consumeSetSize();
      nextDelay = config().consumeTimeMs();
      addEvent("updating", summary(currEnt));
    }
    pushEvent(EVT_CONSUMER, nextDelay);
  }

  private String summary(FeedEntry ent) {
    if (ent.id == 0)
      return "---";
    return String.format("%4d (%d)", ent.id, ent.used);
  }

  private void addEvent(Object... msgs) {
    String s = BasePrinter.toString(msgs);
    mLog.append(String.format("%6d: (%s)  %s\n", mCurrentTime, actor, s));
  }

  private boolean undefined(FeedEntry ent) {
    return !isDefined(ent);
  }

  private boolean isDefined(FeedEntry ent) {
    return ent != null && ent.id != 0;
  }

  private boolean feedEntryActive(int id) {
    for (FeedEntry ent : mWorkList)
      if (ent.id == id)
        return true;
    return false;
  }

  private List<FeedEntry> getEntries() {
    List<FeedEntry> ents = arrayList();
    ents.addAll(mItems.values());
    return ents;
  }

  private static final int EVT_PRODUCER = 1;
  private static final int EVT_CONSUMER = 2;

  private SortedMap<Long, Integer> mQueue = treeMap();

  // Items currently in existence
  //
  private Map<Integer, FeedEntry> mItems = hashMap();

  private long mCurrentTime;
  private Random mRandom;

  // Items being tracked by consumer; id is zero if slot is vacant
  //
  private List<FeedEntry> mWorkList = arrayList();

  private int mCursor = 0;
  private StringBuilder mLog = new StringBuilder();
  private int mNextIdProduced = 500;
  private String actor = "???";

  private static class FeedEntry {
    public FeedEntry(int id) {
      this.id = id;
    }

    int id;
    int used;
  }

  private static final FeedEntry NULL_ENTRY = new FeedEntry(0);

}
