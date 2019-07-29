// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Segmentation.cxx
/// \brief Class for the virtual segmentation of the ALICE Muon Forward Tracker
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>

#include "FairLogger.h"

#include "MFTBase/LadderSegmentation.h"
#include "MFTBase/HalfDiskSegmentation.h"
#include "MFTBase/HalfSegmentation.h"
#include "MFTBase/Segmentation.h"

using namespace o2::mft;

ClassImp(o2::mft::Segmentation);

//_____________________________________________________________________________
Segmentation::Segmentation() : mHalves(nullptr) {}

//_____________________________________________________________________________
Segmentation::Segmentation(const Char_t* nameGeomFile) : mHalves(nullptr)
{

  // constructor

  mHalves = new TClonesArray("o2::mft::HalfSegmentation", NumberOfHalves);
  mHalves->SetOwner(kTRUE);

  auto* halfBottom = new HalfSegmentation(nameGeomFile, Bottom);
  auto* halfTop = new HalfSegmentation(nameGeomFile, Top);

  new ((*mHalves)[Bottom]) HalfSegmentation(*halfBottom);
  new ((*mHalves)[Top]) HalfSegmentation(*halfTop);

  delete halfBottom;
  delete halfTop;

  LOG(DEBUG1) << "MFT segmentation set!" << FairLogger::endl;
}

//_____________________________________________________________________________
Segmentation::~Segmentation()
{

  if (mHalves)
    mHalves->Delete();
  delete mHalves;
}

/// \brief Returns pointer to the segmentation of the half-MFT
/// \param iHalf Integer : 0 = Bottom; 1 = Top
/// \return Pointer to a HalfSegmentation

//_____________________________________________________________________________
HalfSegmentation* Segmentation::getHalf(Int_t iHalf) const
{
  LOG(DEBUG) << Form("Ask for half %d (of %d and %d)", iHalf, Bottom, Top);

  return ((iHalf == Top || iHalf == Bottom) ? ((HalfSegmentation*)mHalves->At(iHalf)) : nullptr);
}

/// Clear the TClonesArray holding the HalfSegmentation objects

//_____________________________________________________________________________
void Segmentation::Clear(const Option_t* /*opt*/)
{
  if (mHalves) {
    mHalves->Delete();
    delete mHalves;
    mHalves = nullptr;
  }
}

/// Returns the local ID of the sensor on the entire disk specified
///
/// \param sensor Int_t : Sensor ID
/// \param ladder Int_t : Ladder ID holding the Sensor
/// \param disk Int_t : Half-Disk ID holding the Sensor
/// \param half Int_t : Half-MFT  ID holding the Sensor
///
/// \return A fixed number that represents the ID of the sensor on the disk. It goes from 0 to the max number of sensor
/// on the disk
//_____________________________________________________________________________
Int_t Segmentation::getDetElemLocalID(Int_t half, Int_t disk, Int_t ladder, Int_t sensor) const
{

  Int_t localId = 0;

  if (half == 1)
    localId += getHalf(0)->getHalfDisk(disk)->getNChips();

  for (Int_t iLad = 0; iLad < getHalf(half)->getHalfDisk(disk)->getNLadders(); iLad++) {
    if (iLad < ladder) {
      localId += getHalf(half)->getHalfDisk(disk)->getLadder(iLad)->getNSensors();
    } else {
      for (Int_t iSens = 0; iSens < getHalf(half)->getHalfDisk(disk)->getLadder(iLad)->getNSensors(); iSens++) {
        if (iSens == sensor)
          return localId;
        localId++;
      }
    }
  }

  return -1;
}
