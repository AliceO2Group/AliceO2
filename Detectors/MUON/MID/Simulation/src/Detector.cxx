// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MIDSimulation/Detector.h"
#include "MIDSimulation/Geometry.h"
#include <TGeoManager.h>
#include <TGeoVolume.h>
#include "FairVolume.h"

ClassImp(o2::mid::Detector);

namespace o2
{
namespace mid
{

Detector::Detector(bool active) : o2::Base::DetImpl<Detector>("MID", active),
                                  mStepper()
{
}

void Detector::defineSensitiveVolumes()
{
  for (auto* vol : getSensitiveVolumes()) {
    AddSensitiveVolume(vol);
  }
}

void Detector::InitializeO2Detector()
{
  defineSensitiveVolumes();
}

bool Detector::ProcessHits(FairVolume* vol)
{
  return mStepper.process(*fMC);
}

std::vector<Hit>* Detector::getHits(int iColl)
{
  if (iColl == 0) {
    return mStepper.getHits();
  }
  return nullptr;
}

bool Detector::setHits(int iColl, std::vector<Hit>* ptr)
{
  if (iColl == 0) {
    mStepper.setHits(ptr);
  }
  return false;
}

void Detector::Register()
{
  /// Registers hits

  mStepper.registerHits(addNameTo("Hit").c_str());
}

void Detector::EndOfEvent() { mStepper.resetHits(); }

void Detector::ConstructGeometry()
{
  TGeoVolume* top = gGeoManager->GetTopVolume();
  if (!top) {
    throw std::runtime_error("Cannot create MID geometry without a top volume");
  }
  createGeometry(*top);
}

} // namespace mid
} // namespace o2
