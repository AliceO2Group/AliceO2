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

/// \file HalfDetector.cxx
/// \brief Class Building the geometry of one half of the ALICE Muon Forward Tracker
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>

#include "TGeoMatrix.h"

#include <fairlogger/Logger.h>

#include "MFTBase/HalfDiskSegmentation.h"
#include "MFTBase/HalfSegmentation.h"
#include "MFTBase/HalfDisk.h"
#include "MFTBase/Geometry.h"
#include "MFTBase/HalfDetector.h"
//#include "MFTBase/PowerSupplyUnit.h"
#include "MFTBase/MFTBaseParam.h"

using namespace o2::mft;

ClassImp(o2::mft::HalfDetector);

/// \brief Default constructor

//_____________________________________________________________________________
HalfDetector::HalfDetector() : TNamed(), mHalfVolume(nullptr), mSegmentation(nullptr) {}

/// \brief Constructor

//_____________________________________________________________________________
HalfDetector::HalfDetector(HalfSegmentation* seg) : TNamed(), mHalfVolume(nullptr), mSegmentation(seg)
{

  Geometry* mftGeom = Geometry::instance();

  SetUniqueID(mSegmentation->GetUniqueID());

  SetName(Form("MFT_H_%d", mftGeom->getHalfID(GetUniqueID())));

  LOG(debug) << Form("Creating : %s ", GetName());

  mHalfVolume = new TGeoVolumeAssembly(GetName());

  //mPSU = new PowerSupplyUnit();

  createHalfDisks();
}

//_____________________________________________________________________________
HalfDetector::~HalfDetector() = default;

/// \brief Creates the Half-disks composing the Half-MFT

//_____________________________________________________________________________
void HalfDetector::createHalfDisks()
{
  LOG(debug) << "MFT: " << Form("Creating  %d Half-Disk ", mSegmentation->getNHalfDisks());
  auto& mftBaseParam = MFTBaseParam::Instance();

  for (Int_t iDisk = 0; iDisk < mSegmentation->getNHalfDisks(); iDisk++) {
    HalfDiskSegmentation* halfDiskSeg = mSegmentation->getHalfDisk(iDisk);
    auto* halfDisk = new HalfDisk(halfDiskSeg);
    Int_t halfDiskId = Geometry::instance()->getDiskID(halfDiskSeg->GetUniqueID());
    mHalfVolume->AddNode(halfDisk->getVolume(), halfDiskId, halfDiskSeg->getTransformation());
    delete halfDisk;
  }
}
