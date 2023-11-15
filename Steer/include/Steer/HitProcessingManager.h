// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_HITPROCESSINGMANAGER_H
#define O2_HITPROCESSINGMANAGER_H

#include "CommonDataFormat/InteractionRecord.h"
#include "SimulationDataFormat/DigitizationContext.h"
#include "SimulationDataFormat/InteractionSampler.h"
#include "CommonUtils/NameConf.h"
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

  // add background file (simprefix) to chain
  void addInputFile(std::string_view simfilename);

  // add a signal file (simprefix) to chain corresponding to signal index "signalindex"
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

  const o2::steer::DigitizationContext& getDigitizationContext() const { return mDigitizationContext; }
  o2::steer::DigitizationContext& getDigitizationContext() { return mDigitizationContext; }

  // serializes the runcontext to file
  void writeDigitizationContext(const char* filename) const;
  // setup run from serialized context; returns true if ok
  bool setupRunFromExistingContext(const char* filename);

  void setRandomEventSequence(bool b) { mSampleCollisionsRandomly = b; }

 private:
  HitProcessingManager() : mSimChains() {}
  bool setupChain();

  bool checkConsistency() const;

  std::vector<RunFunct_t> mRegisteredRunFunctions;
  o2::steer::DigitizationContext mDigitizationContext;

  // this should go into the DigitizationContext --> the manager only fills it
  std::vector<std::string> mBackgroundFileNames;
  std::map<int, std::vector<std::string>> mSignalFileNames;
  std::string mGeometryFile; // geometry file if any

  o2::steer::InteractionSampler mInteractionSampler;

  int mNumberOfCollisions; // how many collisions we want to generate and process
  bool mSampleCollisionsRandomly = false; // if we sample the sequence of event ids randomly (with possible repetition)

  std::vector<TChain*> mSimChains;
  // ClassDefOverride(HitProcessingManager, 0);
};

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
