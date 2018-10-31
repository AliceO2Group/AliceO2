// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//  @file   StepInfo.cxx
//  @author Sandro Wenzel
//  @since  2017-06-29
//  @brief  structures encapsulating information about MC stepping

#include "MCStepLogger/StepInfo.h"
#include <TArrayI.h>
#include <TParticle.h>
#include <TVirtualMC.h>
#include <chrono>

#include <TDatabasePDG.h>
#include <TGeoManager.h>
#include <TGeoMedium.h>
#include <TGeoVolume.h>
#include <cassert>
#include <iostream>

ClassImp(o2::StepInfo);
ClassImp(o2::MagCallInfo);

namespace o2
{
// construct directly using virtual mc
StepInfo::StepInfo(TVirtualMC* mc)
{
  assert(mc);

  // init base time point
  if (stepcounter == -1) {
    starttime = std::chrono::high_resolution_clock::now();
  }
  stepcounter++;
  stepid = stepcounter;

  auto stack = mc->GetStack();

  trackID = stack->GetCurrentTrackNumber();
  lookupstructures.insertPDG(trackID, mc->TrackPid());

  auto id = mc->CurrentVolID(copyNo);
  volId = id;

  auto curtrack = stack->GetCurrentTrack();
  auto parentID = curtrack->IsPrimary() ? -1 : stack->GetCurrentParentTrackNumber();
  lookupstructures.insertParent(trackID, parentID);

  // try to resolve the module via external map
  // keep information in faster vector once looked up
  auto volname = mc->CurrentVolName();
  lookupstructures.insertVolName(volId, volname);

  if (volnametomodulemap && volnametomodulemap->size() > 0 && volId >= 0) {
    if (lookupstructures.getModuleAt(volId) == nullptr) {
      // lookup in map
      auto iter = volnametomodulemap->find(volname);
      if (iter != volnametomodulemap->end()) {
        lookupstructures.insertModuleName(volId, iter->second);
      }
    }
  }

  double xd, yd, zd;
  mc->TrackPosition(xd, yd, zd);
  x = xd;
  y = yd;
  z = zd;
  step = mc->TrackStep();
  maxstep = mc->MaxStep();
  E = curtrack->Energy();
  auto now = std::chrono::high_resolution_clock::now();
  // cputimestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(now - starttime).count();
  nsecondaries = mc->NSecondaries();

  if (nsecondaries > 0) {
    secondaryprocesses = new int[nsecondaries];
    // for the processes
    for (int i = 0; i < nsecondaries; ++i) {
      secondaryprocesses[i] = mc->ProdProcess(i);
    }
  }

  TArrayI procs;
  mc->StepProcesses(procs);
  nprocessesactive = procs.GetSize();

  // was track stopped due to energy limit ??
  stopped = mc->IsTrackStop();
}

std::chrono::time_point<std::chrono::high_resolution_clock> StepInfo::starttime;
int StepInfo::stepcounter = -1;
std::map<std::string, std::string>* StepInfo::volnametomodulemap = nullptr;
std::vector<std::string*> StepInfo::volidtomodulevector;
StepLookups StepInfo::lookupstructures;

MagCallInfo::MagCallInfo(TVirtualMC* mc, float ax, float ay, float az, float aBx, float aBy, float aBz)
  : x{ ax }, y{ ay }, z{ az }, B{ std::sqrt(aBx * aBx + aBy * aBy + aBz * aBz) }
{
  stepcounter++;
  id = stepcounter;
  stepid = StepInfo::stepcounter;
}

int MagCallInfo::stepcounter = -1;
}
