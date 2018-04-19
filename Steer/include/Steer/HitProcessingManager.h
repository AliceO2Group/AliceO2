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
#include <functional>

#include "TChain.h"

namespace o2
{
namespace steer
{
// class fully describing the setup
class RunContext
{
 public:
  TBranch* getBranch(std::string_view name) const
  {
    if (mChain) {
      return mChain->GetBranch(name.data());
    }
    return nullptr;
  }

  int getNEntries() const { return mNofEntries; }
  const std::vector<o2::MCInteractionRecord>& getEventRecords() const { return mEventRecords; }
 private:
  int mNofEntries;
  std::vector<o2::MCInteractionRecord> mEventRecords;
  // std::vector<EventIndices> mEvents; // EventIndices (sourceID, chainID, entry ID)
  TChain* mChain; // pointer to input chain

  friend class HitProcessingManager;
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

  // void addInputFiles(const std::vector<std::string>& simfilenames);
  // void addInputSignalFiles(const std::vector<std::string>& signalfilenames);
  void addInputFile(std::string_view simfilename);
  // void addInputSignalFile(std::string_view signalfilenames);

  void setInteractionSampler();
  void sampleEventTimes();
  void sampleSignalEvents();

  void run();

  void registerRunFunction(RunFunct_t&& f);

 private:
  HitProcessingManager() : mSimChain("o2sim") {}
  void setupRun();
  void setupChain();

  std::vector<RunFunct_t> mRegisteredRunFunctions;
  RunContext mRunContext;
  std::vector<std::string> mSimFileNames;
  // std::vector<std::string> mSignalFileNames;
  o2::steer::InteractionSampler mInteractionSampler;

  TChain mSimChain; // ("o2sim");

  // ClassDefOverride(HitProcessingManager, 0)
};

inline void HitProcessingManager::sampleEventTimes()
{
  mRunContext.mEventRecords.resize(mRunContext.mNofEntries);
  mInteractionSampler.generateCollisionTimes(mRunContext.mEventRecords);
}

inline void HitProcessingManager::setupChain()
{
  mSimChain.Reset();
  for (auto& filename : mSimFileNames) {
    mSimChain.AddFile(filename.data());
  }
  mRunContext.mChain = &mSimChain;
  mRunContext.mNofEntries = mSimChain.GetEntries();
}

inline void HitProcessingManager::setupRun()
{
  setupChain();
  // load geometry
  TGeoManager::Import("O2geometry.root");
  sampleEventTimes();
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
    for (auto entry = 0; entry < c.getNEntries(); ++entry) {
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
  mSimFileNames.emplace_back(simfilename);
}
}
}

#endif
