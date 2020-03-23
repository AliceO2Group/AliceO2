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

#include "CommonDataFormat/InteractionRecord.h"
#include "SimulationDataFormat/DigitizationContext.h"
#include "Steer/InteractionSampler.h"
#include "DetectorsCommonDataFormats/NameConf.h"
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

using RunFunct_t = std::function<void(const o2::steer::DigitizationContext&)>;

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

  o2::steer::InteractionSampler& getInteractionSampler() { return mInteractionSampler; }

  void sampleCollisionTimes();
  void sampleCollisionConstituents();

  void run();

  void registerRunFunction(RunFunct_t&& f);

  // setup the run with ncollisions to treat
  // if -1 and only background chain will do number of entries in chain
  void setupRun(int ncollisions = -1);

  const o2::steer::DigitizationContext& getDigitizationContext() { return mDigitizationContext; }

  // serializes the runcontext to file
  void writeDigitizationContext(const char* filename) const;
  // setup run from serialized context; returns true if ok
  bool setupRunFromExistingContext(const char* filename);

 private:
  HitProcessingManager() : mSimChains() {}
  bool setupChain();

  std::vector<RunFunct_t> mRegisteredRunFunctions;
  o2::steer::DigitizationContext mDigitizationContext;

  // this should go into the DigitizationContext --> the manager only fills it
  std::vector<std::string> mBackgroundFileNames;
  std::map<int, std::vector<std::string>> mSignalFileNames;
  std::string mGeometryFile; // geometry file if any

  o2::steer::InteractionSampler mInteractionSampler;

  int mNumberOfCollisions; // how many collisions we want to generate and process

  std::vector<TChain*> mSimChains;
  // ClassDefOverride(HitProcessingManager, 0);
};

inline void HitProcessingManager::sampleCollisionTimes()
{
  mDigitizationContext.getEventRecords().resize(mDigitizationContext.getNCollisions());
  mInteractionSampler.generateCollisionTimes(mDigitizationContext.getEventRecords());
  mDigitizationContext.getBunchFilling() = mInteractionSampler.getBunchFilling();
  mDigitizationContext.setMuPerBC(mInteractionSampler.getMuPerBC());
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

  // we fill mDigitizationContext.mEventParts
  auto& eventparts = mDigitizationContext.getEventParts();
  eventparts.clear();
  eventparts.resize(mDigitizationContext.getEventRecords().size());
  for (int i = 0; i < mDigitizationContext.getEventRecords().size(); ++i) {
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
  mDigitizationContext.setMaxNumberParts(1);
  if (mSimChains.size() > 1) {
    mDigitizationContext.setMaxNumberParts(2);
  }

  mDigitizationContext.printCollisionSummary();
}

inline void HitProcessingManager::run()
{
  setupRun();
  // sample other stuff
  for (auto& f : mRegisteredRunFunctions) {
    f(mDigitizationContext);
  }
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
} // namespace steer
} // namespace o2

#endif
