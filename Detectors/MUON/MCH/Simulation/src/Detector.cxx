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

#include "MCHSimulation/Detector.h"
#include "MCHGeometryCreator/Geometry.h"
#include "DetectorsBase/Stack.h"
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
  : o2::base::DetImpl<Detector>("MCH", active), mStepper{new o2::mch::Stepper}
{
}

Detector::Detector(const Detector& rhs)
  : o2::base::DetImpl<Detector>(rhs), mStepper{new o2::mch::Stepper}
{
}

Detector::~Detector()
{
  delete mStepper;
}

void Detector::defineSensitiveVolumes()
{
  for (auto* vol : geo::getSensitiveVolumes()) {
    AddSensitiveVolume(vol);
  }
}

void Detector::InitializeO2Detector()
{
  defineSensitiveVolumes();
}

void Detector::ConstructGeometry()
{
  TGeoVolume* top = gGeoManager->GetTopVolume();
  if (!top) {
    throw std::runtime_error("Cannot create MCH geometry without a top volume");
  }
  geo::createGeometry(*gGeoManager, *top);
}

void Detector::addAlignableVolumes() const
{
  if (!gGeoManager) {
    throw std::runtime_error("Cannot add alignable volumes without TGeoManager");
  }
  geo::addAlignableVolumes(*gGeoManager);
}

Bool_t Detector::ProcessHits(FairVolume* v)
{
  mStepper->process(*fMC);
  (static_cast<o2::data::Stack*>(fMC->GetStack()))->addHit(GetDetId());
  return kTRUE;
}

std::vector<o2::mch::Hit>* Detector::getHits(int i)
{
  if (i == 0) {
    return mStepper->getHits();
  }
  return nullptr;
}

bool Detector::setHits(int i, std::vector<o2::mch::Hit>* ptr)
{
  if (i == 0) {
    mStepper->setHits(ptr);
  }
  return false;
}

void Detector::Register()
{
  // TODO : get another way to do I/O (i.e. separate concerns)

  mStepper->registerHits(addNameTo("Hit").c_str());
}

void Detector::EndOfEvent() { mStepper->resetHits(); }

} // namespace mch
} // namespace o2
