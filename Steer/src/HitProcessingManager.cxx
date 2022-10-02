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

#include "Steer/HitProcessingManager.h"
#include <fairlogger/Logger.h>
#include <vector>
#include <map>
#include <iostream>
#include <TFile.h>
#include <TClass.h>
#include <TRandom3.h>

ClassImp(o2::steer::HitProcessingManager);

namespace o2
{
namespace steer
{

bool HitProcessingManager::setupChain()
{
  if (mBackgroundFileNames.size() == 0 && mSignalFileNames.size() == 0) {
    // there are no files to be analysed/processed;
    LOG(warning) << "No files to be analysed";
    return false;
  }

  // count max signal id
  int sourcecounter = 0;
  int maxsourceid = 0;
  for (auto& pair : mSignalFileNames) {
    sourcecounter++;
    maxsourceid = std::max(pair.first, maxsourceid);
  }
  if (maxsourceid != sourcecounter) {
    LOG(warning) << "max source id " << maxsourceid << " vs " << sourcecounter;
  }
  LOG(info) << "setting up " << maxsourceid + 1 << " chains";
  mSimChains.resize(maxsourceid + 1);

  // allocate chains
  for (int i = 0; i < mSimChains.size(); ++i) {
    // make o2sim a parameter
    mSimChains[i] = new TChain("o2sim");
  }

  // background chain
  auto& c = *mSimChains[0];
  c.Reset();
  for (auto& filename : mBackgroundFileNames) {
    c.AddFile(o2::base::NameConf::getMCHeadersFileName(filename.data()).c_str());
  }

  for (auto& pair : mSignalFileNames) {
    const auto& signalid = pair.first;
    const auto& filenamevector = pair.second;
    auto& chain = *mSimChains[signalid];
    for (auto& filename : filenamevector) {
      chain.AddFile(o2::base::NameConf::getMCHeadersFileName(filename.data()).c_str());
    }
  }

  return true;
}

void HitProcessingManager::setupRun(int ncollisions)
{
  if (!setupChain()) {
    return;
  }
  if (mGeometryFile.size() > 0) {
    // load geometry
    TGeoManager::Import(mGeometryFile.c_str());
  }

  //
  if (ncollisions != -1) {
    mNumberOfCollisions = ncollisions;
  } else {
    mNumberOfCollisions = mSimChains[0]->GetEntries();
    LOG(info) << "Automatic deduction of number of collisions ... will just take number of background entries "
              << mNumberOfCollisions;
  }
  mDigitizationContext.setNCollisions(mNumberOfCollisions);
  sampleCollisionTimes();

  // sample collision (background-signal) constituents
  sampleCollisionConstituents();

  // store prefixes as part of Context
  std::vector<std::string> prefixes;
  prefixes.emplace_back(mBackgroundFileNames[0]);
  for (auto k : mSignalFileNames) {
    prefixes.emplace_back(k.second[0]);
  }
  mDigitizationContext.setSimPrefixes(prefixes);
}

void HitProcessingManager::writeDigitizationContext(const char* filename) const
{
  mDigitizationContext.saveToFile(filename);
}

bool HitProcessingManager::setupRunFromExistingContext(const char* filename)
{
  auto context = DigitizationContext::loadFromFile(filename);
  if (context) {
    context->printCollisionSummary();
    mDigitizationContext = *context;
    return true;
  }
  LOG(warn) << "NO DIGITIZATIONCONTEXT FOUND";
  return false;
}

void HitProcessingManager::sampleCollisionTimes()
{
  mDigitizationContext.getEventRecords().resize(mDigitizationContext.getNCollisions());
  mInteractionSampler.generateCollisionTimes(mDigitizationContext.getEventRecords());
  mDigitizationContext.setBunchFilling(mInteractionSampler.getBunchFilling());
  mDigitizationContext.setMuPerBC(mInteractionSampler.getMuPerBC());
}

void HitProcessingManager::sampleCollisionConstituents()
{
  TRandom3 rnd(0); // we don't use the global to be in isolation
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

  auto getRandomBackground = [this, &rnd]() {
    int numbg = mSimChains[0]->GetEntries();
    const auto eventID = (int)numbg * rnd.Rndm();
    return EventPart(0, eventID);
  };

  auto getRandomSignal = [this, nsignalids, &rnd]() {
    const auto sourceID = 1 + (int)(rnd.Rndm() * nsignalids);
    const auto signalID = (int)(rnd.Rndm() * mSimChains[sourceID]->GetEntries());
    return EventPart(sourceID, signalID);
  };

  // we fill mDigitizationContext.mEventParts
  auto& eventparts = mDigitizationContext.getEventParts();
  eventparts.clear();
  eventparts.resize(mDigitizationContext.getEventRecords().size());
  for (int i = 0; i < mDigitizationContext.getEventRecords().size(); ++i) {
    eventparts[i].clear();
    // NOTE: THIS PART WOULD BENEFIT FROM A MAJOR REDESIGN
    // WISHFUL ITEMS WOULD BE:
    // - ALLOW COLLISION ENGINEERING FROM OUTSIDE (give wanted sequence as file)
    //    * the outside person can decide what kind of sampling and sequence to use
    // - CHECK IF VERTEX IS CONSISTENT
    if (mSampleCollisionsRandomly) {
      eventparts[i].emplace_back(getRandomBackground());
      if (mSimChains.size() > 1) {
        eventparts[i].emplace_back(getRandomSignal());
      }
    } else {
      // push any number of constituents?
      // for the moment just 2 : one background and one signal
      eventparts[i].emplace_back(getBackgroundRoundRobin());
      if (mSimChains.size() > 1) {
        eventparts[i].emplace_back(getSignalRoundRobin());
      }
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

void HitProcessingManager::run()
{
  setupRun();
  // sample other stuff
  for (auto& f : mRegisteredRunFunctions) {
    f(mDigitizationContext);
  }
}

} // end namespace steer
} // end namespace o2
