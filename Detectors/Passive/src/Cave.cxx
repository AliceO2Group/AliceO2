// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.


/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/

// -------------------------------------------------------------------------
// -----                    Cave  file                               -----
// -----                Created 26/03/14  by M. Al-Turany              -----
// -------------------------------------------------------------------------
#include "DetectorsBase/MaterialManager.h"
#include "DetectorsBase/Detector.h"
#include "DetectorsPassive/Cave.h"
#include "FairLogger.h"
#include "TGeoManager.h"
#include "TGeoVolume.h"

using namespace o2::Passive;

void Cave::createMaterials()
{
  auto& matmgr = o2::Base::MaterialManager::Instance();
  // Create materials and media
  Int_t isxfld;
  Float_t sxmgmx;
  o2::Base::Detector::initFieldTrackingParams(isxfld, sxmgmx);

  // AIR
  Float_t aAir[4] = { 12.0107, 14.0067, 15.9994, 39.948 };
  Float_t zAir[4] = { 6., 7., 8., 18. };
  Float_t wAir[4] = { 0.000124, 0.755267, 0.231781, 0.012827 };
  Float_t dAir = 1.20479E-3 * 960. / 1014.;

  //
  matmgr.Mixture("CAVE", 2, "Air", aAir, zAir, dAir, 4, wAir);
  //
  matmgr.Medium("CAVE", 2, "Air", 2, 0, isxfld, sxmgmx, 10, -1, -0.1, 0.1, -10);
}

void Cave::ConstructGeometry()
{
  createMaterials();
  auto& matmgr = o2::Base::MaterialManager::Instance();
  Float_t dALIC[3];

  // TODO: add various options depending on whether ZDC is present or not
  dALIC[0] = 2000;
  dALIC[1] = 2000;
  dALIC[2] = 3000;
  auto cavevol = gGeoManager->MakeBox("cave", gGeoManager->GetMedium("CAVE_Air"), dALIC[0], dALIC[1], dALIC[2]);
  gGeoManager->SetTopVolume(cavevol);
}

Cave::Cave() : FairDetector() {}
Cave::~Cave() = default;
Cave::Cave(const char* name, const char* Title) : FairDetector(name, Title) {}
Cave::Cave(const Cave& rhs) : FairDetector(rhs) {}
Cave& Cave::operator=(const Cave& rhs)
{
  // self assignment
  if (this == &rhs)
    return *this;

  // base class assignment
  FairModule::operator=(rhs);
  return *this;
}

FairModule* Cave::CloneModule() const { return new Cave(*this); }
void Cave::FinishPrimary()
{
  LOG(DEBUG) << "CAVE: Primary finished" << FairLogger::endl;
  for (auto& f : mFinishPrimaryHooks) {
    f();
  }
}

bool Cave::ProcessHits(FairVolume*)
{
  LOG(FATAL) << "CAVE ProcessHits called; should never happen" << FairLogger::endl;
  return false;
}

ClassImp(o2::Passive::Cave)
