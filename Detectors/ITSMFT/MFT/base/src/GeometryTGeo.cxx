// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GeometryTGeo.cxx
/// \brief Implementation of the GeometryTGeo class
/// \author bogdan.vulpescu@cern.ch 
/// \date 01/08/2016

#include <boost/bimap.hpp>

#include "TGeoManager.h"
#include "TSystem.h"

#include "FairLogger.h"

#include "ITSMFTBase/Segmentation.h"
#include "ITSMFTBase/SegmentationPixel.h"

#include "MFTBase/Constants.h"
#include "MFTBase/Geometry.h"
#include "MFTBase/GeometryTGeo.h"
#include "MFTBase/HalfSegmentation.h"
#include "MFTBase/HalfDiskSegmentation.h"
#include "MFTBase/LadderSegmentation.h"

using o2::ITSMFT::Segmentation;
using o2::ITSMFT::SegmentationPixel;

using namespace o2::MFT;

ClassImp(o2::MFT::GeometryTGeo)

TString GeometryTGeo::sVolumeName   = "MFT";
TString GeometryTGeo::sHalfName     = "MFT_H";
TString GeometryTGeo::sDiskName     = "MFT_D";
TString GeometryTGeo::sLadderName   = "MFT_L";
TString GeometryTGeo::sSensorName   = "MFT_S";

TString GeometryTGeo::sSegmentationFileName = "mftSegmentations.root";

//_____________________________________________________________________________
GeometryTGeo::GeometryTGeo(Bool_t forceBuild, Bool_t loadSegmentations) 
  : TObject(),
    mSensorMatrices(nullptr),
    mSensorMatricesToITS(nullptr),
    mRotateMFTtoITS(nullptr),
    mTrackingToLocalMatrices(nullptr),
    mSegmentations(nullptr),
    mNumberOfDisks(0),
    mNumberOfChips(0),
    mChipIDmap(),
    mChipHalfID(nullptr),
    mChipDiskID(nullptr),
    mChipPlaneID(nullptr),
    mChipLadderID(nullptr),
    mChipSensorID(nullptr)
{
  // default constructor

  if (forceBuild) {
    build(loadSegmentations);
  }

}

//_____________________________________________________________________________
GeometryTGeo::~GeometryTGeo()
{
  // destructor

  delete mSensorMatrices;
  delete mSensorMatricesToITS;
  delete mSegmentations;

  delete [] mChipHalfID;
  delete [] mChipDiskID;
  delete [] mChipPlaneID;
  delete [] mChipLadderID;
  delete [] mChipSensorID;

}

//_____________________________________________________________________________
GeometryTGeo::GeometryTGeo(const GeometryTGeo& src)
  : TObject(src),
    mSensorMatrices(nullptr),
    mSensorMatricesToITS(nullptr),
    mRotateMFTtoITS(nullptr),
    mTrackingToLocalMatrices(nullptr),
    mSegmentations(nullptr),
    mNumberOfDisks(src.mNumberOfDisks),
    mNumberOfChips(src.mNumberOfChips),
    mChipIDmap(),
    mChipHalfID(nullptr),
    mChipDiskID(nullptr),
    mChipPlaneID(nullptr),
    mChipLadderID(nullptr),
    mChipSensorID(nullptr)
{
  // copy constructor

  mRotateMFTtoITS = new TGeoHMatrix(*src.mRotateMFTtoITS);

  if (src.mSensorMatrices) {
    mSensorMatrices = new TObjArray(mNumberOfChips);
    mSensorMatrices->SetOwner(kTRUE);
    for (int i = 0; i < mNumberOfChips; i++) {
      const TGeoHMatrix* mat = (TGeoHMatrix*)src.mSensorMatrices->At(i);
      mSensorMatrices->AddAt(new TGeoHMatrix(*mat), i);
    }
  }
  if (src.mSensorMatricesToITS) {
    mSensorMatricesToITS = new TObjArray(mNumberOfChips);
    mSensorMatricesToITS->SetOwner(kTRUE);
    for (int i = 0; i < mNumberOfChips; i++) {
      const TGeoHMatrix* mat = (TGeoHMatrix*)src.mSensorMatricesToITS->At(i);
      mSensorMatricesToITS->AddAt(new TGeoHMatrix(*mat), i);
    }
  }
  if (src.mSegmentations) {
    int sz = src.mSegmentations->GetEntriesFast();
    mSegmentations = new TObjArray(sz);
    mSegmentations->SetOwner(kTRUE);
    for (int i = 0; i < sz; i++) {
      Segmentation* sg = (Segmentation*)src.mSegmentations->UncheckedAt(i);
      if (!sg) {
	continue;
      }
      mSegmentations->AddAt(sg->Clone(), i);
    }
  }

  for (BimapType::const_iterator iter = src.mChipIDmap.begin(), iend = src.mChipIDmap.end(); iter != iend; ++iter) {
    mChipIDmap.insert(BimapType::value_type(iter->left,iter->right));
  }

  if (mNumberOfChips) {
    mChipHalfID = new Int_t[mNumberOfChips];
    mChipDiskID = new Int_t[mNumberOfChips];
    mChipPlaneID = new Int_t[mNumberOfChips];
    mChipLadderID = new Int_t[mNumberOfChips];
    mChipSensorID = new Int_t[mNumberOfChips];
    for (Int_t i = 0; i < mNumberOfChips; i++) {
      mChipHalfID[i] = src.mChipHalfID[i];
      mChipDiskID[i] = src.mChipDiskID[i];
      mChipPlaneID[i] = src.mChipPlaneID[i];
      mChipLadderID[i] = src.mChipLadderID[i];
      mChipSensorID[i] = src.mChipSensorID[i];
    }
  }

}

//_____________________________________________________________________________
GeometryTGeo &GeometryTGeo::operator=(const GeometryTGeo &src)
{

  if (this == &src) {
    return *this;
  }

  TObject::operator=(src);
  mNumberOfDisks = src.mNumberOfDisks;
  mNumberOfChips = src.mNumberOfChips;

  mRotateMFTtoITS = new TGeoHMatrix(*src.mRotateMFTtoITS);

  if (src.mSensorMatrices) {
    delete mSensorMatrices;
    mSensorMatrices = new TObjArray(mNumberOfChips);
    mSensorMatrices->SetOwner(kTRUE);
    for (int i = 0; i < mNumberOfChips; i++) {
      const TGeoHMatrix* mat = (TGeoHMatrix*)src.mSensorMatrices->At(i);
      mSensorMatrices->AddAt(new TGeoHMatrix(*mat), i);
    }
  }
  if (src.mSensorMatricesToITS) {
    delete mSensorMatricesToITS;
    mSensorMatricesToITS = new TObjArray(mNumberOfChips);
    mSensorMatricesToITS->SetOwner(kTRUE);
    for (int i = 0; i < mNumberOfChips; i++) {
      const TGeoHMatrix* mat = (TGeoHMatrix*)src.mSensorMatricesToITS->At(i);
      mSensorMatricesToITS->AddAt(new TGeoHMatrix(*mat), i);
    }
  }
  if (src.mSegmentations) {
    int sz = src.mSegmentations->GetEntriesFast();
    mSegmentations = new TObjArray(sz);
    mSegmentations->SetOwner(kTRUE);
    for (int i = 0; i < sz; i++) {
      Segmentation* sg = (Segmentation*)src.mSegmentations->UncheckedAt(i);
      if (!sg) {
	continue;
      }
      mSegmentations->AddAt(sg->Clone(), i);
    }
  }
  for (BimapType::const_iterator iter = src.mChipIDmap.begin(), iend = src.mChipIDmap.end(); iter != iend; ++iter) {
    mChipIDmap.insert(BimapType::value_type(iter->left,iter->right));
  }
  
  if (mNumberOfChips) {
    mChipHalfID = new Int_t[mNumberOfChips];
    mChipDiskID = new Int_t[mNumberOfChips];
    mChipPlaneID = new Int_t[mNumberOfChips];
    mChipLadderID = new Int_t[mNumberOfChips];
    mChipSensorID = new Int_t[mNumberOfChips];
    for (Int_t i = 0; i < mNumberOfChips; i++) {
      mChipHalfID[i] = src.mChipHalfID[i];
      mChipDiskID[i] = src.mChipDiskID[i];
      mChipPlaneID[i] = src.mChipPlaneID[i];
      mChipLadderID[i] = src.mChipLadderID[i];
      mChipSensorID[i] = src.mChipSensorID[i];
    }
  }
  return *this;
}

//_____________________________________________________________________________
Int_t GeometryTGeo::getChipIndex(Int_t half, Int_t disk, Int_t ladder, Int_t sensor) const
{

  Geometry *mftGeo = Geometry::instance();
  Int_t sensorUniqueID, chipID;
  for (Int_t plane = 0; plane < 2; plane++) {
    sensorUniqueID = mftGeo->getObjectID(Geometry::SensorType,half,disk,plane,ladder,sensor);
    for (BimapType::left_map::const_iterator left_iter = mChipIDmap.left.begin(), iend = mChipIDmap.left.end(); left_iter != iend; ++left_iter) {
      if (left_iter->first == sensorUniqueID) {
	chipID = left_iter->second; 
      }
    }
  }

  return chipID;

}

//_____________________________________________________________________________
void GeometryTGeo::fetchMatrices()
{

  if (!gGeoManager) {
    LOG(FATAL) << "Geometry is not loaded" << FairLogger::endl;
  }

  mSensorMatrices = new TObjArray(mNumberOfChips);
  mSensorMatrices->SetOwner(kTRUE);
  for (int i = 0; i < mNumberOfChips; i++) {
    mSensorMatrices->AddAt(new TGeoHMatrix(*extractMatrixSensor(i)), i);
  }

  mSensorMatricesToITS = new TObjArray(mNumberOfChips);
  mSensorMatricesToITS->SetOwner(kTRUE);

  TGeoHMatrix *mSensorMatrixToITS[mNumberOfChips];
  //
  // xITS    0  +1   0   xMFT
  // yITS =  0   0  +1 * yMFT
  // zITS   +1   0   0   zMFT
  //
  mRotateMFTtoITS = new TGeoHMatrix();
  mRotateMFTtoITS->RotateY(-90.);
  mRotateMFTtoITS->RotateZ(-90.);
  
  Double_t *rotMFTtoITSmatrix;
  Double_t rotMatrix_2[9] = {0.,0.,0.,0.,0.,0.,0.,0.,0.};
  for (int iChip = 0; iChip < mNumberOfChips; iChip++) {
    rotMFTtoITSmatrix = mRotateMFTtoITS->GetRotationMatrix();
    mSensorMatrixToITS[iChip] = new TGeoHMatrix(*(TGeoHMatrix*)mSensorMatrices->At(iChip));
    Double_t *rotMatrix_1 = mSensorMatrixToITS[iChip]->GetRotationMatrix();
    for (Int_t i = 0; i < 3; i++) {
      for (Int_t j = 0; j < 3; j++) {
	rotMatrix_2[i*3+j] = 0.;
	for (Int_t k = 0; k < 3; k++) {
	  rotMatrix_2[i*3+j] += rotMFTtoITSmatrix[i*3+k]*rotMatrix_1[k*3+j];
	}
      }
    }
    mSensorMatrixToITS[iChip]->SetRotation(rotMatrix_2);
    mSensorMatricesToITS->AddAt(mSensorMatrixToITS[iChip],iChip);
  }
  
  createT2LMatrices();

}

//_____________________________________________________________________________
void GeometryTGeo::createT2LMatrices()
{
  // create tracking to local (Sensor!) matrices

}

//_____________________________________________________________________________
TGeoHMatrix* GeometryTGeo::extractMatrixSensor(Int_t index) const
{

  Geometry *mftGeo = Geometry::instance();

  Int_t sensorUniqueID, type, halfID, diskID, planeID, ladderID, sensorID;

  for (BimapType::right_map::const_iterator right_iter = mChipIDmap.right.begin(), iend = mChipIDmap.right.end(); right_iter != iend; ++right_iter) {
    if (right_iter->first == index) {
      sensorUniqueID = right_iter->second; 
    }
  }
  
  type     = mftGeo->getObjectType(sensorUniqueID);
  halfID   = mftGeo->getHalfID(sensorUniqueID);
  diskID   = mftGeo->getDiskID(sensorUniqueID);
  planeID  = mftGeo->getPlaneID(sensorUniqueID);
  ladderID = mftGeo->getLadderID(sensorUniqueID);
  sensorID = mftGeo->getSensorID(sensorUniqueID);

  TString path = Form("/cave_1/%s_0/",GeometryTGeo::getVolumeName());

  path += Form("%s_%d_%d/",GeometryTGeo::getHalfName(),halfID,halfID);
  path += Form("%s_%d_%d_%d/",GeometryTGeo::getDiskName(),halfID,diskID,diskID);
  path += Form("%s_%d_%d_%d_%d/",GeometryTGeo::getLadderName(),halfID,diskID,ladderID,ladderID);
  path += Form("%s_%d_%d_%d_%d",GeometryTGeo::getSensorName(),halfID,diskID,ladderID,sensorID);

  static TGeoHMatrix matTmp;
  gGeoManager->PushPath();

  if (!gGeoManager->cd(path.Data())) {
    gGeoManager->PopPath();
    LOG(ERROR) << "Error in cd-ing to " << path.Data() << FairLogger::endl;
    return nullptr;
  }

  matTmp = *gGeoManager->GetCurrentMatrix(); // matrix may change after cd

  // Restore the modeler state.
  gGeoManager->PopPath();

  return &matTmp;

}

//_____________________________________________________________________________
void GeometryTGeo::build(Bool_t loadSegmentations)
{

  if (!gGeoManager) {
    LOG(FATAL) << "Geometry is not loaded" << FairLogger::endl;
  } 
  Geometry *mftGeo = Geometry::instance();

  Segmentation *seg;
  if (loadSegmentations) {
    LOG(INFO) << "GeometryTGeo::build load segmentation from xml file!" << FairLogger::endl;
    seg = new Segmentation(gSystem->ExpandPathName("$(VMCWORKDIR)/Detectors/Geometry/MFT/data/Geometry.xml" ));
  } else {
    LOG(INFO) << "GeometryTGeo::build get segmentation from the geometry!" << FairLogger::endl;
    seg = mftGeo->getSegmentation();
  }

  mNumberOfDisks = Constants::sNDisks;

  // fill the map <sensorUniqueID,chipID>
  Int_t chipID = 0;
  Int_t sensorUniqueID;
  for (Int_t iHalf = 0; iHalf < 2; iHalf++) {
    HalfSegmentation * halfSeg = seg->getHalf(iHalf);
    for (Int_t iDisk = 0; iDisk < mNumberOfDisks; iDisk++) {
      HalfDiskSegmentation* halfDiskSeg = halfSeg->getHalfDisk(iDisk);
      for (Int_t iLadder = 0; iLadder < halfDiskSeg->getNLadders(); iLadder++) {
	LadderSegmentation* ladderSeg = halfDiskSeg->getLadder(iLadder);
	mNumberOfChips += ladderSeg->getNSensors();
	for (Int_t iSensor = 0; iSensor < ladderSeg->getNSensors(); iSensor++) {
	  ChipSegmentation * chipSeg = ladderSeg->getSensor(iSensor);
	  sensorUniqueID = chipSeg->GetUniqueID();
	  mChipIDmap.insert(BimapType::value_type(sensorUniqueID,chipID++));
	}
      }
    }
  }
  
  mChipHalfID = new Int_t[mNumberOfChips];
  mChipDiskID = new Int_t[mNumberOfChips];
  mChipPlaneID = new Int_t[mNumberOfChips];
  mChipLadderID = new Int_t[mNumberOfChips];
  mChipSensorID = new Int_t[mNumberOfChips];
  for (Int_t iHalf = 0; iHalf < 2; iHalf++) {
    HalfSegmentation * halfSeg = seg->getHalf(iHalf);
    for (Int_t iDisk = 0; iDisk < mNumberOfDisks; iDisk++) {
      HalfDiskSegmentation* halfDiskSeg = halfSeg->getHalfDisk(iDisk);
      for (Int_t iLadder = 0; iLadder < halfDiskSeg->getNLadders(); iLadder++) {
	LadderSegmentation* ladderSeg = halfDiskSeg->getLadder(iLadder);
	for (Int_t iSensor = 0; iSensor < ladderSeg->getNSensors(); iSensor++) {
	  ChipSegmentation * chipSeg = ladderSeg->getSensor(iSensor);
	  for (Int_t plane = 0; plane < 2; plane++) {
	    sensorUniqueID = mftGeo->getObjectID(Geometry::SensorType,iHalf,iDisk,plane,iLadder,iSensor);
	    for (BimapType::left_map::const_iterator left_iter = mChipIDmap.left.begin(), iend = mChipIDmap.left.end(); left_iter != iend; ++left_iter) {
	      if (left_iter->first == sensorUniqueID) {
		chipID = left_iter->second; 
		mChipHalfID[chipID] = iHalf;
		mChipDiskID[chipID] = iDisk;
		mChipPlaneID[chipID] = plane;
		mChipLadderID[chipID] = iLadder;
		mChipSensorID[chipID] = iSensor;
	      }
	    }
	  }
	}
      }
    }
  }
  
  fetchMatrices();

  if (loadSegmentations) {
    mSegmentations = new TObjArray();
    SegmentationPixel::loadSegmentations(mSegmentations, getMFTsegmentationFileName());
  }

}

//_____________________________________________________________________________
Int_t GeometryTGeo::getChipHalfID(Int_t index) const
{

  if (index >= 0 && index < mNumberOfChips) 
    return mChipHalfID[index];
  else 
    return -1;

}

//_____________________________________________________________________________
Int_t GeometryTGeo::getChipDiskID(Int_t index) const
{

  if (index >= 0 && index < mNumberOfChips) 
    return mChipDiskID[index];
  else 
    return -1;

}

//_____________________________________________________________________________
Int_t GeometryTGeo::getChipPlaneID(Int_t index) const
{

  if (index >= 0 && index < mNumberOfChips) 
    return mChipPlaneID[index];
  else 
    return -1;

}

//_____________________________________________________________________________
Int_t GeometryTGeo::getChipLadderID(Int_t index) const
{

  if (index >= 0 && index < mNumberOfChips) 
    return mChipLadderID[index];
  else 
    return -1;

}

//_____________________________________________________________________________
Int_t GeometryTGeo::getChipSensorID(Int_t index) const
{

  if (index >= 0 && index < mNumberOfChips) 
    return mChipSensorID[index];
  else 
    return -1;

}

