// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHSimulation/Detector.h"
#include "MCHSimulation/Geometry.h"
#include "Stepper.h"
#include "TGeoManager.h"
#include <sstream>
#include <iostream>
#include <array>
#include "TGeoManager.h"

ClassImp(o2::mch::Detector);

namespace o2
{
namespace mch
{

Detector::Detector(bool active)
  : o2::Base::DetImpl<Detector>("MCH", active), mStepper{ new o2::mch::Stepper }
{
}

Detector::~Detector()
{
  delete mStepper;
}

void Detector::defineSensitiveVolumes()
{
  for (auto* vol : getSensitiveVolumes()) {
    AddSensitiveVolume(vol);
  }
}

void Detector::Initialize()
{
  defineSensitiveVolumes();
  o2::Base::Detector::Initialize();
}

void Detector::ConstructGeometry()
{
  TGeoVolume* top = gGeoManager->GetTopVolume();
  if (!top) {
    throw std::runtime_error("Cannot create MCH geometry without a top volume");
  }
  createGeometry(*top);
}

Bool_t Detector::ProcessHits(FairVolume* v)
{
  mStepper->process(*fMC);
  return kTRUE;
}

std::vector<o2::mch::Hit>* Detector::getHits(int) { return nullptr; /*return mStepper->getHits();*/ }

void Detector::Register()
{
  // TODO : get another way to do I/O (i.e. separate concerns)

  mStepper->registerHits(addNameTo("Hit").c_str());
}

void Detector::EndOfEvent() { mStepper->resetHits(); }

} // namespace mch
} // namespace o2
