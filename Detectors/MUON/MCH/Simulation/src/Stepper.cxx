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

#include "Stepper.h"

#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsBase/Stack.h"
#include "SimulationDataFormat/TrackReference.h"
#include "TGeoManager.h"
#include "TVirtualMC.h"
#include "TVirtualMCStack.h"
#include "TMCProcess.h"
#include <iomanip>
#include "TArrayI.h"

#include "FairRootManager.h"

namespace o2
{
namespace mch
{

Stepper::Stepper() : mHits{o2::utils::createSimVector<o2::mch::Hit>()} {}
Stepper::~Stepper()
{
  o2::utils::freeSimVector(mHits);
}

void Stepper::process(const TVirtualMC& vmc)
{

  if (!(vmc.TrackCharge())) {
    // Only charged particles
    return;
  }

  o2::SimTrackStatus t{vmc};

  int detElemId;
  vmc.CurrentVolOffID(2, detElemId); // go up 2 levels in the hierarchy to get the DE

  auto stack = static_cast<o2::data::Stack*>(vmc.GetStack());

  if (t.isEntering() || t.isExiting()) {
    // generate a track referenced
    o2::TrackReference tr{vmc, o2::detectors::DetID::MCH};
    tr.setUserId(detElemId);
    stack->addTrackReference(tr);
  }

  if (t.isEntering()) {
    float x, y, z;
    vmc.TrackPosition(x, y, z);
    mEntrancePoint.SetXYZ(x, y, z);
    resetStep();
  }

  mTrackEloss += vmc.Edep();
  mTrackLength += vmc.TrackStep();

  if (t.isExiting() || t.isStopped()) {
    float x, y, z;
    vmc.TrackPosition(x, y, z);
    mHits->emplace_back(stack->GetCurrentTrackNumber(), detElemId, mEntrancePoint,
                        math_utils::Point3D<float>{x, y, z}, mTrackEloss, mTrackLength);
    resetStep();
  }
}

void Stepper::registerHits(const char* branchName)
{
  if (FairRootManager::Instance()) {
    FairRootManager::Instance()->RegisterAny(branchName, mHits, kTRUE);
  }
}

void Stepper::resetStep()
{
  mTrackEloss = 0.0;
  mTrackLength = 0.0;
}

void Stepper::resetHits()
{
  if (!o2::utils::ShmManager::Instance().isOperational()) {
    mHits->clear();
  }
}

} // namespace mch
} // namespace o2
