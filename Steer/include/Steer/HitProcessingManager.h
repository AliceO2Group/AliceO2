// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_HITPROCESSINGMANAGER_H
#define O2_HITPROCESSINGMANAGER_H

#include <Steer/InteractionSampler.h>
#include <TGeoManager.h>
#include <string>
#include <vector>
#include <map>
#include <functional>

#include "TChain.h"

namespace o2
{
namespace steer
{

// a structure describing EventPart
// (an elementary constituent of a collision)
struct EventPart {
  EventPart() = default;
  EventPart(int s, int e) : sourceID(s), entryID(e) {}
  int sourceID = 0; // the ID of the source (0->backGround; > 1 signal source)
  // the sourceID should correspond to the chain ID
  int entryID = 0; // the event/entry ID inside the chain corresponding to sourceID

  static bool isSignal(EventPart e) { return e.sourceID > 1; }
  static bool isBackGround(EventPart e) { return !isSignal(e); }
  ClassDefNV(EventPart, 1);
};

// class fully describing the Collision contexts
class RunContext
{
 public:
  TBranch* getBranch(std::string_view name, int sourceid = 0) const
  {
    if (mChains[sourceid]) {
      return mChains[sourceid]->GetBranch(name.data());
    }
    return nullptr;
  }

  int getNCollisions() const { return mNofEntries; }

  void setMaxNumberParts(int maxp) { mMaxPartNumber = maxp; }
  int getMaxNumberParts() const { return mMaxPartNumber; }

  const std::vector<o2::MCInteractionRecord>& getEventRecords() const { return mEventRecords; }
  const std::vector<std::vector<EventPart>>& getEventParts() const { return mEventParts; }
  const std::vector<TChain*>& getChains() const { return mChains; }

  void printCollisionSummary() const
  {
    for (int i = 0; i < mEventRecords.size(); ++i) {
      std::cout << "Collision " << i << " TIME " << mEventRecords[i].timeNS;
      for (auto& e : mEventParts[i]) {
        std::cout << " (" << e.sourceID << " , " << e.entryID << ")";
      }
      std::cout << "\n";
    }
  }

 private:
  int mNofEntries;
  int mMaxPartNumber; // max number of parts in any given collision
  std::vector<o2::MCInteractionRecord> mEventRecords;
  // for each collision we record the constituents (which shall not exceed mMaxPartNumber)
  std::vector<std::vector<EventPart>> mEventParts;
  std::vector<TChain*> mChains; //! pointers to input chains

  // it would also be appropriate to record the filenames
  // that went into the chain

  friend class HitProcessingManager;
  ClassDefNV(RunContext, 1);
};

using RunFunct_t = std::function<void(const RunContext&)>;

/// O2 specific run class; steering hit processing
class HitProcessingManager
{
 public:
  /// get access to singleton instance
  static HitProcessingManager& instance()
  {
    static HitProcessingManager mgr;
    return mgr;
  }
  ~HitProcessingManager() = default;

  // add background file to chain
  void addInputFile(std::string_view simfilename);

  // add a signal file to chain corresponding to signal index "signalindex"
  void addInputSignalFile(std::string_view signalfilename, int signalindex = 1);

  void setGeometryFile(std::string const& geomfile) { mGeometryFile = geomfile; }

  void setInteractionSampler();
  void sampleCollisionTimes();
  void sampleCollisionConstituents();

  void run();

  void registerRunFunction(RunFunct_t&& f);

  // setup the run with ncollisions to treat
  // if -1 and only background chain will do number of entries in chain
  void setupRun(int ncollisions = -1);

  const RunContext& getRunContext() { return mRunContext; }

 private:
  HitProcessingManager() : mSimChains() {}
  bool setupChain();

  std::vector<RunFunct_t> mRegisteredRunFunctions;
  RunContext mRunContext;

  // this should go into the RunContext --> the manager only fills it
  std::vector<std::string> mBackgroundFileNames;
  std::map<int, std::vector<std::string>> mSignalFileNames;
  std::string mGeometryFile; // geometry file if any

  o2::steer::InteractionSampler mInteractionSampler;

  int mNumberOfCollisions; // how many collisions we want to generate and process

  std::vector<TChain*> mSimChains;
  // ClassDefOverride(HitProcessingManager, 0)
};

inline void HitProcessingManager::sampleCollisionTimes()
{
  mRunContext.mEventRecords.resize(mRunContext.mNofEntries);
  mInteractionSampler.generateCollisionTimes(mRunContext.mEventRecords);
}

inline void HitProcessingManager::sampleCollisionConstituents()
{
  auto getBackgroundRoundRobin = [this]() {
    static int bgcounter = 0;
    int numbg = mSimChains[0]->GetEntries();
    if (bgcounter == numbg) {
      bgcounter = 0;
    }
    return EventPart(0, bgcounter++);
  };

  const int nsignalids = mSimChains.size() - 1;
  auto getSignalRoundRobin = [this, nsignalids]() {
    static int bgcounter = 0;
    static int signalid = 0;
    static std::vector<int> counter(nsignalids, 0);
    if (signalid == nsignalids) {
      signalid = 0;
    }
    const auto realsourceid = signalid + 1;
    int numentries = mSimChains[realsourceid]->GetEntries();
    if (counter[signalid] == numentries) {
      counter[signalid] = 0;
    }
    EventPart e(realsourceid, counter[signalid]);
    counter[signalid]++;
    signalid++;
    return e;
  };

  // we fill mRunContext.mEventParts
  auto& eventparts = mRunContext.mEventParts;
  eventparts.clear();
  eventparts.resize(mRunContext.mEventRecords.size());
  for (int i = 0; i < mRunContext.mEventRecords.size(); ++i) {
    eventparts[i].clear();
    // push any number of constituents?
    // for the moment just 2 : one background and one signal
    eventparts[i].emplace_back(getBackgroundRoundRobin());
    if (mSimChains.size() > 1) {
      eventparts[i].emplace_back(getSignalRoundRobin());
    }
  }

  // push any number of constituents?
  // for the moment just max 2 : one background and one signal
  mRunContext.setMaxNumberParts(1);
  if (mSimChains.size() > 1) {
    mRunContext.setMaxNumberParts(2);
  }

  mRunContext.printCollisionSummary();
}

inline void HitProcessingManager::run()
{
  setupRun();
  // sample other stuff
  for (auto& f : mRegisteredRunFunctions) {
    f(mRunContext);
  }
}

template <typename HitType, typename Task_t>
std::function<void(const o2::steer::RunContext&)> defaultRunFunction(Task_t& task, std::string_view brname)
{
  //  using HitType = Task_t::InputType;
  return [&task, brname](const o2::steer::RunContext& c) {
    HitType* hittype = nullptr;
    auto br = c.getBranch(brname.data());
    assert(br);
    br->SetAddress(&hittype);
    for (auto entry = 0; entry < c.getNCollisions(); ++entry) {
      br->GetEntry(entry);
      task.setData(hittype, &c);
      task.Exec("");
    }
    task.FinishTask();
    // delete hittype
  };
}

inline void HitProcessingManager::registerRunFunction(RunFunct_t&& f) { mRegisteredRunFunctions.emplace_back(f); }

inline void HitProcessingManager::addInputFile(std::string_view simfilename)
{
  mBackgroundFileNames.emplace_back(simfilename);
}

inline void HitProcessingManager::addInputSignalFile(std::string_view simfilename, int signal)
{
  if (mSignalFileNames.find(signal) == mSignalFileNames.end()) {
    // insert empty vector for id signal
    mSignalFileNames.insert(std::pair<int, std::vector<std::string>>(signal, std::vector<std::string>()));
  }
  mSignalFileNames[signal].emplace_back(simfilename);
}
}
}

#endif
