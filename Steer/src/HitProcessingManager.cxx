// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Steer/HitProcessingManager.h"
#include "FairLogger.h"
#include <vector>
#include <map>
#include <iostream>
#include <TFile.h>
#include <TClass.h>

ClassImp(o2::steer::HitProcessingManager);

namespace o2
{
namespace steer
{

bool HitProcessingManager::setupChain()
{
  if (mBackgroundFileNames.size() == 0 && mSignalFileNames.size() == 0) {
    // there are no files to be analysed/processed;
    LOG(WARNING) << "No files to be analysed";
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
    LOG(WARNING) << "max source id " << maxsourceid << " vs " << sourcecounter;
  }
  LOG(INFO) << "setting up " << maxsourceid + 1 << " chains";
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
    c.AddFile(filename.data());
  }

  for (auto& pair : mSignalFileNames) {
    const auto& signalid = pair.first;
    const auto& filenamevector = pair.second;
    auto& chain = *mSimChains[signalid];
    for (auto& filename : filenamevector) {
      chain.AddFile(filename.data());
    }
  }

  // print out entries in each chain
  for (int i = 0; i < mSimChains.size(); ++i) {
    mRunContext.mChains.emplace_back(mSimChains[i]);
    LOG(INFO) << "CHAIN " << i << " has " << mSimChains[i]->GetEntries() << " entries ";
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
    LOG(INFO) << "Automatic deduction of number of collisions ... will just take number of background entries "
              << mNumberOfCollisions;
  }
  mRunContext.mNofEntries = mNumberOfCollisions;
  sampleCollisionTimes();

  // sample collision (background-signal) constituents
  sampleCollisionConstituents();
}

void RunContext::printCollisionSummary() const
{
  std::cout << "Summary of RunContext --\n";
  std::cout << "Number of Collisions " << mEventRecords.size() << "\n";
  for (int i = 0; i < mEventRecords.size(); ++i) {
    std::cout << "Collision " << i << " TIME " << mEventRecords[i].timeNS;
    for (auto& e : mEventParts[i]) {
      std::cout << " (" << e.sourceID << " , " << e.entryID << ")";
    }
    std::cout << "\n";
  }
}

void HitProcessingManager::writeRunContext(const char* filename) const
{
  TFile file(filename, "RECREATE");
  auto cl = TClass::GetClass(typeid(mRunContext));
  file.WriteObjectAny(&mRunContext, cl, "RunContext");
  file.Close();
}

bool HitProcessingManager::setupRunFromExistingContext(const char* filename)
{
  RunContext* incontext = nullptr;
  TFile file(filename, "OPEN");
  file.GetObject("RunContext", incontext);

  if (incontext) {
    incontext->printCollisionSummary();
    mRunContext = *incontext;
    return true;
  }
  LOG(INFO) << "NO COLLISIONOBJECT FOUND";
  return false;
}
} // end namespace steer
} // end namespace o2
