// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//****************************************************************************
//* This file is free software: you can redistribute it and/or modify        *
//* it under the terms of the GNU General Public License as published by     *
//* the Free Software Foundation, either version 3 of the License, or        *
//* (at your option) any later version.                                      *
//*                                                                          *
//* Primary Authors: Sandro Wenzel <sandro.wenzel@cern.ch>                   *
//*                                                                          *
//* The authors make no claims about the suitability of this software for    *
//* any purpose. It is provided "as is" without express or implied warranty. *
//****************************************************************************

//  @file   StepInfo.cxx
//  @author Sandro Wenzel
//  @since  2017-06-29
//  @brief  structures encapsulating information about MC stepping


#include <StepInfo.h>
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
  currentinstance = this;

  eventid = mc->CurrentEvent();
  auto stack = mc->GetStack();

  trackID = stack->GetCurrentTrackNumber();
  pdg = mc->TrackPid();
  auto particle = TDatabasePDG::Instance()->GetParticle(pdg);
  pname = particle ? particle->GetName() : "NULL";
  auto id = mc->CurrentVolID(copyNo);
  volId = id;

  auto curtrack = stack->GetCurrentTrack();
  primary = curtrack ? curtrack->IsPrimary() : false;

  auto geovolume = gGeoManager->GetCurrentVolume();
  volname = geovolume ? geovolume->GetName() : "NULL";
  auto medium = geovolume ? geovolume->GetMedium() : nullptr;
  mediumname = medium ? medium->GetName() : "NULL";

  // try to resolve the module via external map
  // keep information in faster vector once looked up
  modulename = "UNKNOWN";
  if (volnametomodulemap && volnametomodulemap->size() > 0 && volId >= 0) {
    // lookup in vector first
    if (volId >= volidtomodulevector.size()) {
      volidtomodulevector.resize(volId + 1, nullptr);
    }
    if (volidtomodulevector[volId] == nullptr) {
      // lookup in map
      auto iter = volnametomodulemap->find(volname);
      if (iter != volnametomodulemap->end()) {
        volidtomodulevector[volId] = &iter->second;
      }
    }
    if (volidtomodulevector[volId]) {
      modulename = *volidtomodulevector[volId];
    }
  }

  // auto v2 = gGeoManager->GetCurrentNavigator()->GetCurrentVolume();
  // if (strcmp(mc->CurrentVolName(), v2->GetName())!=0){
  //  std::cerr << "inconsistent state\n";
  //}

  double xd, yd, zd;
  mc->TrackPosition(xd, yd, zd);
  x = xd;
  y = yd;
  z = zd;
  E = curtrack->Energy();
  auto now = std::chrono::high_resolution_clock::now();
  cputimestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(now - starttime).count();
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

  // is track entering volume

  // is track exiting volume

  // was track stoped due to energy limit ??
  stopped = mc->IsTrackStop();
}

std::chrono::time_point<std::chrono::high_resolution_clock> StepInfo::starttime;
int StepInfo::stepcounter = -1;
StepInfo* StepInfo::currentinstance = nullptr;
std::map<std::string, std::string>* StepInfo::volnametomodulemap = nullptr;
std::vector<std::string*> StepInfo::volidtomodulevector;

MagCallInfo::MagCallInfo(TVirtualMC* mc, float ax, float ay, float az, float aBx, float aBy, float aBz)
  : x{ ax }, y{ ay }, z{ az }, Bx{ aBx }, By{ aBy }, Bz{ aBz }
{
  stepcounter++;
  id = stepcounter;
  stepid = StepInfo::stepcounter;
  // copy the stepinfo
  if (StepInfo::currentinstance) {
    // stepinfo = *StepInfo::currentinstance;
  }
}

int MagCallInfo::stepcounter = -1;
}
