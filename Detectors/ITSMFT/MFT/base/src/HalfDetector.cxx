// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HalfDetector.cxx
/// \brief Class Building the geometry of one half of the ALICE Muon Forward Tracker
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>

#include "TGeoMatrix.h"

#include "FairLogger.h"

#include "MFTBase/HalfDiskSegmentation.h"
#include "MFTBase/HalfSegmentation.h"
#include "MFTBase/HalfDisk.h"
#include "MFTBase/Geometry.h"
#include "MFTBase/HalfDetector.h"
#include "MFTBase/PowerSupplyUnit.h"

using namespace o2::mft;

ClassImp(o2::mft::HalfDetector);

/// \brief Default constructor

//_____________________________________________________________________________
HalfDetector::HalfDetector() : TNamed(), mHalfVolume(nullptr), mSegmentation(nullptr), mPSU(nullptr) {}

/// \brief Constructor

//_____________________________________________________________________________
HalfDetector::HalfDetector(HalfSegmentation* seg) : TNamed(), mHalfVolume(nullptr), mSegmentation(seg), mPSU(nullptr)
{

  Geometry* mftGeom = Geometry::instance();

  SetUniqueID(mSegmentation->GetUniqueID());

  SetName(Form("MFT_H_%d", mftGeom->getHalfID(GetUniqueID())));

  LOG(DEBUG) << Form("Creating : %s ", GetName());

  mHalfVolume = new TGeoVolumeAssembly(GetName());

  mPSU = new PowerSupplyUnit();

  createHalfDisks();
}

//_____________________________________________________________________________
HalfDetector::~HalfDetector() = default;

/// \brief Creates the Half-disks composing the Half-MFT

//_____________________________________________________________________________
void HalfDetector::createHalfDisks()
{
  LOG(DEBUG) << "MFT: " << Form("Creating  %d Half-Disk ", mSegmentation->getNHalfDisks());

  for (Int_t iDisk = 0; iDisk < mSegmentation->getNHalfDisks(); iDisk++) {
    HalfDiskSegmentation* halfDiskSeg = mSegmentation->getHalfDisk(iDisk);
    auto* halfDisk = new HalfDisk(halfDiskSeg);
    Int_t halfDiskId = Geometry::instance()->getDiskID(halfDiskSeg->GetUniqueID());
    mHalfVolume->AddNode(halfDisk->getVolume(), halfDiskId, halfDiskSeg->getTransformation());
    delete halfDisk;
  }
  TGeoVolumeAssembly* mHalfPSU = mPSU->create();
  TGeoTranslation* tHalfPSU = new TGeoTranslation("tHalfPSU", 0, 0.4, -72.6 + 46.0);
  tHalfPSU->RegisterYourself();
  mHalfVolume->AddNode(mHalfPSU, 0, tHalfPSU);
}
