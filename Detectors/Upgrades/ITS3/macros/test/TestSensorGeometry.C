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

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "Framework/Logger.h"
#include "ITS3Base/SpecsV2.h"
#include "ITS3Simulation/ITS3Layer.h"

#include "TFile.h"
#include "TGeoManager.h"
#include "TGeoMaterial.h"
#include "TGeoMedium.h"
#include "TList.h"
#endif

void TestSensorGeometry(bool checkFull = false)
{
  gGeoManager = new TGeoManager("simple", "Simple geometry");
  TGeoMaterial* matVacuum = new TGeoMaterial("Vacuum", 0, 0, 0);
  TGeoMedium* Vacuum = new TGeoMedium("Vacuum", 1, matVacuum);

  auto top = gGeoManager->MakeBox("TOP", Vacuum, 270., 270., 120.);
  gGeoManager->SetTopVolume(top);

  o2::its3::ITS3Layer layer0{0, top, nullptr,
                             o2::its3::ITS3Layer::BuildLevel::kLayer, true};

  // Print available medias
  TIter next{gGeoManager->GetListOfMedia()};
  TObject* obj;
  while ((obj = (TObject*)next())) {
    LOGP(info, "Media {}", obj->GetName());
  }

  gGeoManager->CloseGeometry();
  gGeoManager->SetVisLevel(99);
  if (checkFull) {
    gGeoManager->CheckGeometryFull();
  }
  gGeoManager->CheckOverlaps(0.0001);
  TIter nextOverlap{gGeoManager->GetListOfOverlaps()};
  while ((obj = (TObject*)nextOverlap())) {
    LOGP(info, "Overlap in {}", obj->GetName());
  }

  std::unique_ptr<TFile> f{TFile::Open("geo.root", "RECREATE")};
  f->WriteTObject(gGeoManager, "geometry");
}
