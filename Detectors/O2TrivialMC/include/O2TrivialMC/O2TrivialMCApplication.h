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

// An absolut minimal implementation of a TVirtualMCApplication for studies and code that e.g. only depends on the presence of a VMC.

#ifndef ALICEO2_MC_TRIVIALMCAPPLICATION_H_
#define ALICEO2_MC_TRIVIALMCAPPLICATION_H_

#include "TGeoManager.h"
#include "TGeoBBox.h"
#include "TGeoMaterial.h"
#include "TGeoMedium.h"

#include "TVirtualMCApplication.h"

namespace o2
{

namespace mc
{

class O2TrivialMCApplication : public TVirtualMCApplication
{
 public:
  O2TrivialMCApplication() : TVirtualMCApplication("O2TrivialMCApplication", "O2TrivialMCApplication") {}
  ~O2TrivialMCApplication() override = default;
  O2TrivialMCApplication(O2TrivialMCApplication const& app) {}
  void ConstructGeometry() override
  {
    auto geoMgr = gGeoManager;
    // we need some dummies, any material and medium will do
    auto mat = new TGeoMaterial("vac", 0, 0, 0);
    auto med = new TGeoMedium("vac", 1, mat);
    auto vol = geoMgr->MakeBox("cave", med, 1, 1, 1);
    geoMgr->SetTopVolume(vol);
    geoMgr->CloseGeometry();
  }
  void InitGeometry() override {}
  void GeneratePrimaries() override {}
  void BeginEvent() override {}
  void BeginPrimary() override {}
  void PreTrack() override {}
  void Stepping() override {}
  void PostTrack() override {}
  void FinishPrimary() override {}
  void FinishEvent() override {}
  TVirtualMCApplication* CloneForWorker() const override
  {
    return new O2TrivialMCApplication(*this);
  }
};

} // namespace mc

} // namespace o2

#endif
