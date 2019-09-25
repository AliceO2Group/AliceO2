// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MIDSimulation/Stepper.h"

#include "CommonUtils/ShmAllocator.h"
#include "SimulationDataFormat/Stack.h"
#include "SimulationDataFormat/TrackReference.h"
#include "TVirtualMC.h"
#include "FairRootManager.h"

namespace o2
{
namespace mid
{

Stepper::Stepper() : mHits{o2::utils::createSimVector<o2::mid::Hit>()} {}
Stepper::~Stepper()
{
  o2::utils::freeSimVector(mHits);
}

bool Stepper::process(const TVirtualMC& vmc)
{
  o2::SimTrackStatus ts{vmc};

  int detElemId;
  vmc.CurrentVolOffID(1, detElemId); // go up 1 level in the hierarchy to get the DE

  auto stack = static_cast<o2::data::Stack*>(vmc.GetStack());

  if (ts.isEntering() || ts.isExiting()) {
    // generate a track referenced
    o2::TrackReference tr{vmc, detElemId};
    stack->addTrackReference(tr);
  }

  if (ts.isEntering()) {
    float x, y, z;
    vmc.TrackPosition(x, y, z);
    mEntrancePoint.SetXYZ(x, y, z);
    resetStep();
  }

  mTrackEloss += vmc.Edep();
  mTrackLength += vmc.TrackStep();

  if (ts.isExiting() || ts.isStopped()) {
    float x, y, z;
    vmc.TrackPosition(x, y, z);
    mHits->emplace_back(stack->GetCurrentTrackNumber(), detElemId, mEntrancePoint,
                        Point3D<float>{x, y, z}, mTrackEloss, mTrackLength);
    resetStep();
  }

  return true;
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

} // namespace mid
} // namespace o2
