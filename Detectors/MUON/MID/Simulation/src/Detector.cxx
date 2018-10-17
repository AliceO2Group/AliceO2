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
#include "Geometry.h"
#include <TGeoManager.h>
#include <TGeoVolume.h>
#include <vector>
#include "FairVolume.h"
#include <stdexcept>

using namespace o2::mid;

Detector::Detector(bool active) : o2::Base::DetImpl<Detector>("MID", active),
                                  mHits(new std::vector<HitType>)
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

bool Detector::ProcessHits(FairVolume* v)
{
  return false;
}

void Detector::ConstructGeometry()
{
  TGeoVolume* top = gGeoManager->GetTopVolume();
  if (!top) {
    throw std::runtime_error("Cannot create MID geometry without a top volume");
  }
  createGeometry(*top);
}

ClassImp(Detector);
