// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file FemtoDreamEventCuts.cxx
/// \brief Implementation of the FemtoDreamEventCuts
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "o2femtodream/FemtoDreamCollisionSelection.h"
#include <iostream>

using namespace o2::analysis::femtoDream;

FemtoDreamCollisionSelection::FemtoDreamCollisionSelection()
  : mZvtxMax(999.f),
    mCheckTrigger(false),
    mTrigger(kINT7),
    mCheckOffline(false),
    mHistogramRegistry(nullptr),
    mDoQA(false)
{
}

FemtoDreamCollisionSelection::FemtoDreamCollisionSelection(float zvtxMax, bool checkTrigger, triggerAliases trig, bool checkOffline)
  : mZvtxMax(zvtxMax),
    mCheckTrigger(checkTrigger),
    mTrigger(trig),
    mCheckOffline(checkOffline),
    mHistogramRegistry(nullptr),
    mDoQA(false)
{
}

void FemtoDreamCollisionSelection::init(HistogramRegistry* registry)
{
  mHistogramRegistry = registry;
  mDoQA = true;

  mHistogramRegistry->add("Event/zvtxhist", "; vtx_{z} (cm); Entries", kTH1F, {{1000, -15, 15}});
  mHistogramRegistry->add("Event/MultV0M", "; vMultV0M; Entries", kTH1F, {{1000, 0, 1000}});
}

std::string FemtoDreamCollisionSelection::getCutHelp()
{
  return "Max. z-vertex (cm); "
         "Check trigger; "
         "Trigger; "
         "Check offline";
}

void FemtoDreamCollisionSelection::printCuts()
{
  std::cout << "Debug information for FemtoDreamCollisionSelection \n Max. z-vertex: " << mZvtxMax << "\n Check trigger: " << mCheckTrigger << "\n Trigger: " << mTrigger << "\n Check offline: " << mCheckOffline << "\n";
}
