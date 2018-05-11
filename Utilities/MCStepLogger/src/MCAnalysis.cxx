// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCStepLogger/MCAnalysis.h"
#include "MCStepLogger/MCAnalysisManager.h"

ClassImp(o2::mcstepanalysis::MCAnalysis);

using namespace o2::mcstepanalysis;

MCAnalysis::MCAnalysis(const std::string& name)
  : mName(name), mIsInitialized(false), mAnalysisFile(nullptr)
{
  // Automatically register to manager
  auto& anamgr = MCAnalysisManager::Instance();
  anamgr.registerAnalysis(this);
  mAnalysisManager = &anamgr;
}