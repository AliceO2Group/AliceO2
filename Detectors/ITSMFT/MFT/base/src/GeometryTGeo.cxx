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
/// \author bogdan.vulpescu@clermont.in2p3.fr - adapted from ITS, 21.09.2017

#include "ITSMFTBase/SegmentationAlpide.h"

#include "MFTBase/GeometryTGeo.h"
#include "MFTBase/Geometry.h"

#include "DetectorsBase/GeometryManager.h"
#include "MathUtils/Cartesian3D.h"

#include "FairLogger.h" // for LOG

#include <TGeoBBox.h>         // for TGeoBBox
#include <TGeoManager.h>      // for gGeoManager, TGeoManager
#include <TGeoPhysicalNode.h> // for TGeoPNEntry, TGeoPhysicalNode
#include <TGeoShape.h>        // for TGeoShape
#include <TMath.h>            // for Nint, ATan2, RadToDeg
#include <TString.h>          // for TString, Form
#include <TClass.h>           // for TClass
#include <TGeoMatrix.h>       // for TGeoHMatrix
#include <TGeoNode.h>         // for TGeoNode, TGeoNodeMatrix
#include <TGeoVolume.h>       // for TGeoVolume
#include <TMathBase.h>        // for Max
#include <TObjArray.h>        // for TObjArray
#include <TObject.h>          // for TObject

#include <cctype>  // for isdigit
#include <cstdio>  // for snprintf, NULL, printf
#include <cstring> // for strstr, strlen

using namespace TMath;
using namespace o2::MFT;
using namespace o2::Base;
using namespace o2::Base::Utils;

using AlpideSegmentation = o2::ITSMFT::SegmentationAlpide;

ClassImp(o2::MFT::GeometryTGeo);

std::unique_ptr<o2::MFT::GeometryTGeo> GeometryTGeo::sInstance;

std::string GeometryTGeo::sVolumeName     = "MFT";           ///< 
std::string GeometryTGeo::sHalfName       = "MFT_H";         ///< 
std::string GeometryTGeo::sDiskName       = "MFT_D";         ///< 
std::string GeometryTGeo::sLadderName     = "MFT_L";         ///< 
std::string GeometryTGeo::sChipName       = "MFT_C";         ///< 
std::string GeometryTGeo::sSensorName     = "MFTSensor";     ///< 

//__________________________________________________________________________
GeometryTGeo::GeometryTGeo(Bool_t build, Int_t loadTrans)
  : o2::ITSMFT::GeometryTGeo(DetID::MFT)
{
  // default c-tor, if build is true, the structures will be filled and the transform matrices
  // will be cached
  if (sInstance) {
    LOG(FATAL) << "Invalid use of public constructor: o2::MFT::GeometryTGeo instance exists" << FairLogger::endl; 
    //throw std::runtime_error("Invalid use of public constructor: o2::MFT::GeometryTGeo instance exists");
  }
  
  if (build) {
    //loadTrans = kTRUE;
    Build(loadTrans);
  }
}

//__________________________________________________________________________
void GeometryTGeo::Build(Int_t loadTrans)
{
  if ( isBuilt() ) {
    LOG(WARNING) << "Already built" << FairLogger::endl;
    return; // already initialized
  }
  
  if (!gGeoManager) {
    // RSTODO: in future there will be a method to load matrices from the CDB
    LOG(FATAL) << "Geometry is not loaded" << FairLogger::endl;
  }

  mNumberOfHalves = extractNumberOfHalves();
  if (!mNumberOfHalves) {
    return;
  }

  //LOG(INFO) << "Number of halves " << mNumberOfHalves << FairLogger::endl;
  mNumberOfDisks.resize(mNumberOfHalves);

  mTotalNumberOfSensors = 0;
  for (Int_t i = 0; i < mNumberOfHalves; i++) {

    mNumberOfDisks[i] = extractNumberOfDisks(i);
    //LOG(INFO) << "Number of disks " << mNumberOfDisks[i] << " in half " << i << FairLogger::endl;

    // use one half only
    if (i == 0) {

      mNumberOfLadders.resize(mNumberOfDisks[i]);
      mNumberOfLaddersPerDisk.resize(mNumberOfDisks[i]);
      mLastSensorIndex.resize(mNumberOfDisks[i]);
      mLadderIndex2Id.resize(mNumberOfDisks[i]);
      mLadderId2Index.resize(mNumberOfDisks[i]);

      // loop over disks
      for (Int_t j = 0; j < mNumberOfDisks[i]; j++) {

        mNumberOfLadders[j].resize(MaxSensorsPerLadder);
        Int_t numberOfLadders = 0;
        for (Int_t nSensor = MinSensorsPerLadder; nSensor <= MaxSensorsPerLadder; nSensor++) {

          mNumberOfLadders[j][nSensor] = extractNumberOfLadders(i,j,nSensor);
          //LOG(INFO) << "Number of ladders with " << nSensor << " sensors is " << mNumberOfLadders[j][nSensor] << " in disk " << j << FairLogger::endl;

          numberOfLadders += mNumberOfLadders[j][nSensor];
          mTotalNumberOfSensors += mNumberOfLadders[j][nSensor]*nSensor;
            
        } // nSensor
        mLastSensorIndex[j] = mTotalNumberOfSensors - 1;
        mNumberOfLaddersPerDisk[j] = numberOfLadders;

        mLadderIndex2Id[j].resize(numberOfLadders);
        mLadderId2Index[j].resize(numberOfLadders);
        Int_t nL = 0;
        for (Int_t nSensor = MinSensorsPerLadder; nSensor <= MaxSensorsPerLadder; nSensor++) {
          if (mNumberOfLadders[j][nSensor] == 0) continue;
          Int_t n = extractNumberOfLadders(i,j,nSensor,nL);
        } // nSensor
    
      } // disk

    } // half = 0

  } // halves

  mTotalNumberOfSensors *= mNumberOfHalves;
  LOG(INFO) << "Total number of sensors " << mTotalNumberOfSensors << " in " << mNumberOfHalves << " detector halves" << FairLogger::endl;
  /*
  // checks
  for (Int_t i = 0; i < mTotalNumberOfSensors; i++) {
    Int_t ladder = getLadder(i);
    Int_t disk = getDisk(i);
    Int_t half = getHalf(i);
    Int_t ladderID = mLadderIndex2Id[disk][ladder];
    LOG(INFO) << "Index " << i << " half " << half << " disk " << disk << " ladder " << ladder << " geomap " << ladderID << FairLogger::endl;
  }
  */
  setSize(mTotalNumberOfSensors);

  fillMatrixCache(loadTrans); 
  /*
  // checks
  Int_t index;
  for (Int_t iH = 0; iH < mNumberOfHalves; iH++) {
    for (Int_t iD = 0; iD < mNumberOfDisks[iH]; iD++) {
      for (Int_t iL = 0; iL < mNumberOfLaddersPerDisk[iD]; iL++) {
        Int_t ladder = mLadderId2Index[iD][iL];
        Int_t nS = extractNumberOfSensorsPerLadder(iH,iD,iL);
        for (Int_t iS = 0; iS < nS; iS++) {
          index = getSensorIndex(iH,iD,iL,iS);
          LOG(INFO) << "Half " << iH << " disk " << iD << " ladder " << ladder << " ladderID " << iL << " sensor " << iS << " index " << index << FairLogger::endl;
        } // sensor
      } // ladder
    } // disk
  } // half
  */
}

//__________________________________________________________________________
Int_t GeometryTGeo::extractNumberOfSensorsPerLadder(Int_t half, Int_t disk, Int_t ladder) const
{
 
  Int_t numberOfSensors = 0;
  Char_t laddername[30];
  snprintf(laddername, 30, "%s_%d_%d_%d", getMFTLadderPattern(), half, disk, ladder);
  TGeoVolume* volLadder = gGeoManager->GetVolume(laddername);
  if (!volLadder) {
    LOG(FATAL) << "can't find volume " << laddername << FairLogger::endl;
  }
  // Loop on all ladder nodes, count sensor volumes by checking names
  Int_t nNodes = volLadder->GetNodes()->GetEntries();
  for (int j = 0; j < nNodes; j++) {
    //LOG(INFO) << "GeometryTGeo::extractNumberOfSensorsPerLadder " << half << " " << disk << " " << ladder << " " << volLadder->GetNodes()->At(j)->GetName() << FairLogger::endl;
    if (strstr(volLadder->GetNodes()->At(j)->GetName(), getMFTChipPattern())) {
      numberOfSensors++;
    }
  }
 
  return numberOfSensors;

}

//__________________________________________________________________________
Int_t GeometryTGeo::extractNumberOfLadders(Int_t half, Int_t disk, Int_t nsensor) const
{
  
  Int_t numberOfLadders = 0;
  Char_t diskname[30];
  snprintf(diskname, 30, "%s_%d_%d", getMFTDiskPattern(), half, disk);
  TGeoVolume* volDisk = gGeoManager->GetVolume(diskname);
  if (!volDisk) {
    LOG(FATAL) << "can't find volume " << diskname << FairLogger::endl;
  }
  // Loop on all disk nodes, count ladder volumes by checking names
  TObjArray* nodes = volDisk->GetNodes();
  Int_t nNodes = nodes->GetEntries();
  Int_t ladderID = -1;
  for (int j = 0; j < nNodes; j++) {
    TGeoNode* nd = (TGeoNode*)nodes->At(j);
    const Char_t* name = nd->GetName();
    if (strstr(name, getMFTLadderPattern())) {
      ladderID = extractVolumeCopy(name, Form("%s_%d_%d",getMFTLadderPattern(),half,disk));
      if (nsensor == extractNumberOfSensorsPerLadder(half,disk,ladderID)) { 
        numberOfLadders++;
      }
    }
  }
  
  return numberOfLadders;

}

//__________________________________________________________________________
Int_t GeometryTGeo::extractNumberOfLadders(Int_t half, Int_t disk, Int_t nsensor, Int_t& nL)
{
  
  Int_t numberOfLadders = 0;
  Char_t diskname[30];
  snprintf(diskname, 30, "%s_%d_%d", getMFTDiskPattern(), half, disk);
  TGeoVolume* volDisk = gGeoManager->GetVolume(diskname);
  if (!volDisk) {
    LOG(FATAL) << "can't find volume " << diskname << FairLogger::endl;
  }
  // Loop on all disk nodes, count ladder volumes by checking names
  TObjArray* nodes = volDisk->GetNodes();
  Int_t nNodes = nodes->GetEntries();
  Int_t ladderID = -1;
  for (int j = 0; j < nNodes; j++) {
    TGeoNode* nd = (TGeoNode*)nodes->At(j);
    const Char_t* name = nd->GetName();
    if (strstr(name, getMFTLadderPattern())) {
      ladderID = extractVolumeCopy(name, Form("%s_%d_%d",getMFTLadderPattern(),half,disk));
      if (nsensor == extractNumberOfSensorsPerLadder(half,disk,ladderID)) { 
        // map the new index with the one from the geometry
        mLadderIndex2Id[disk][nL] = ladderID;     
        mLadderId2Index[disk][ladderID] = nL;     
        //LOG(INFO) << "In disk " << disk << " ladder with " << nsensor << " sensors has matrix index " << nL << " and geometry index " << mLadderIndex2Id[disk][nL] << FairLogger::endl;
        nL++;
        //
        numberOfLadders++;
      }
    }
  }
  
  return numberOfLadders;

}

//__________________________________________________________________________
Int_t GeometryTGeo::extractNumberOfDisks(Int_t half) const
{

  Int_t numberOfDisks = 0;
  Char_t halfname[30];
  snprintf(halfname, 30, "%s_%d", getMFTHalfPattern(), half);
  TGeoVolume* volHalf = gGeoManager->GetVolume(halfname);
  if (!volHalf) {
    LOG(FATAL) << "can't find " << halfname << " volume" << FairLogger::endl;
    return -1;
  }

  // Loop on all half nodes, count disk volumes by checking names
  Int_t nNodes = volHalf->GetNodes()->GetEntries();
  for (Int_t j = 0; j < nNodes; j++) {
    if (strstr(volHalf->GetNodes()->At(j)->GetName(), getMFTDiskPattern())) {
      numberOfDisks++;
    }
  }

  return numberOfDisks;

}

//__________________________________________________________________________
Int_t GeometryTGeo::extractNumberOfHalves()
{
 
  Int_t numberOfHalves = 0;

  TGeoVolume* volMFT = gGeoManager->GetVolume(getMFTVolPattern());
  if (!volMFT) {
    LOG(FATAL) << "MFT volume " << getMFTVolPattern() << " is not in the geometry" << FairLogger::endl;
  }

  // Loop on all MFT nodes and count half detector volumes by checking names
  TObjArray* nodes = volMFT->GetNodes();
  int nNodes = nodes->GetEntriesFast();

  for (int j = 0; j < nNodes; j++) {
    Int_t halfID = -1;
    TGeoNode* nd = (TGeoNode*)nodes->At(j);
    const Char_t* name = nd->GetName();

    if (strstr(name, getMFTHalfPattern())) {
      numberOfHalves++;
      if ((halfID = extractVolumeCopy(name, getMFTHalfPattern())) < 0) {
        LOG(FATAL) << "Failed to extract half ID from the " << name << FairLogger::endl;
        exit(1);
      }
    } 
  }

  return numberOfHalves;

}

//__________________________________________________________________________
int GeometryTGeo::extractVolumeCopy(const char* name, const char* prefix) const
{

  TString nms = name;
  if (!nms.BeginsWith(prefix)) {
    return -1;
  }
  nms.Remove(0, strlen(prefix)+1);
  if (!isdigit(nms.Data()[0])) {
    return -1;
  }

  return nms.Atoi();

}

//__________________________________________________________________________
TGeoHMatrix* GeometryTGeo::extractMatrixSensor(Int_t index) const
{

  Int_t half, disk, ladder, sensor, ladderID;
  getSensorID(index, half, disk, ladder, sensor);
  ladderID = mLadderIndex2Id[disk][ladder];
  //LOG(INFO) << "extractMatrixSensor index " << index << " half " << half << " disk " << disk << " ladder " << ladder << " ladderID " << ladderID << FairLogger::endl;

  TString path = Form("/cave_1/%s_0/", getMFTVolPattern());
  path += Form("%s_%d_%d/%s_%d_%d_%d/%s_%d_%d_%d_%d/%s_%d_%d_%d_%d/%s_1",
               getMFTHalfPattern(),half,half,
               getMFTDiskPattern(),half,disk,disk,
               getMFTLadderPattern(),half,disk,ladderID,ladderID,
               getMFTChipPattern(),half,disk,ladderID,sensor,
               getMFTSensorPattern());
  //LOG(INFO) << "Volume path is " << path.Data() << FairLogger::endl;

  static TGeoHMatrix matTmp;
  gGeoManager->PushPath();

  if (!gGeoManager->cd(path.Data())) {
    gGeoManager->PopPath();
    LOG(ERROR) << "Error in cd-ing to " << path.Data() << FairLogger::endl;
    return nullptr;
  } // end if !gGeoManager

  matTmp = *gGeoManager->GetCurrentMatrix(); // matrix may change after cd

  // Restore the modeler state.
  gGeoManager->PopPath();

  // account for the difference between sensitive layer and physical sensor ticknesses
  static TGeoTranslation tra(0.,0.5*(AlpideSegmentation::SensorThickness-AlpideSegmentation::SensLayerThickness),0.);

  matTmp *= tra;

  return &matTmp;

}

//__________________________________________________________________________
void GeometryTGeo::fillMatrixCache(Int_t mask)
{

  // populate matrix cache for requested transformations
  //  
  if (mSize < 1) {    
    LOG(WARNING) << "The method Build was not called yet" << FairLogger::endl;
    Build(mask);
    return;
  }
  //LOG(INFO) << "mask " << mask << " bit2Mask " << bit2Mask(TransformType::L2G) << FairLogger::endl;
  // build matrices
  if ( (mask & bit2Mask(TransformType::L2G)) && !getCacheL2G().isFilled() ) {
    LOG(INFO) << "Loading MFT L2G matrices from TGeo" <<  FairLogger::endl;
    auto& cacheL2G = getCacheL2G();
    cacheL2G.setSize(mSize);
    for (Int_t i = 0; i < mSize; i++) {
      TGeoHMatrix* hm = extractMatrixSensor(i);
      cacheL2G.setMatrix( hm ? Mat3D(*hm) : Mat3D(), i);
    }
  } 

  if ( (mask & bit2Mask(TransformType::T2L)) && !getCacheT2L().isFilled() ) {
    // matrices for Tracking to Local frame transformation
    LOG(INFO) << "Loading MFT T2L matrices from TGeo" <<  FairLogger::endl;
    auto& cacheT2L = getCacheT2L();
    cacheT2L.setSize(mSize);
    for (int i=0;i<mSize;i++) {
      TGeoHMatrix& hm = createT2LMatrix(i);
      cacheT2L.setMatrix( Mat3D(hm) , i);
    }
  }

  if ( (mask & bit2Mask(TransformType::T2G)) && !getCacheT2G().isFilled() ) {
    // matrices for Tracking to Global frame transformation
    LOG(INFO) << "Loading MFT T2G matrices from TGeo" <<  FairLogger::endl;
    auto& cacheT2G = getCacheT2G();
    cacheT2G.setSize(mSize);
    for (int i=0;i<mSize;i++) {
      TGeoHMatrix& mat = createT2LMatrix(i);
      mat.MultiplyLeft( extractMatrixSensor(i) );
      cacheT2G.setMatrix( Mat3D(mat), i);
    }    
  }

}

//__________________________________________________________________________
TGeoHMatrix& GeometryTGeo::createT2LMatrix(Int_t index)
{

  // create for sensor isn the TGeo matrix for Tracking to Local frame transformations

  static TGeoHMatrix t2l;
  Float_t x = 0.f, alpha = 0.f;
  extractSensorXAlpha(index,x,alpha);
  t2l.Clear();
  /*
  t2l.RotateZ(alpha * RadToDeg()); // rotate in direction of normal to the sensor plane
  const TGeoHMatrix* matL2G = extractMatrixSensor(isn);
  t2l.MultiplyLeft(&matL2G->Inverse());
  */
  return t2l;

}

//__________________________________________________________________________
void GeometryTGeo::extractSensorXAlpha(int index, float &x, float &alpha)
{

}

//__________________________________________________________________________
Bool_t GeometryTGeo::getSensorID(Int_t index, Int_t& half, Int_t& disk, Int_t& ladder, Int_t& sensor) const
{

  if (index < 0 || index >= mTotalNumberOfSensors) return kFALSE;

  half = index/(mTotalNumberOfSensors/mNumberOfHalves);
  index = index%(mTotalNumberOfSensors/mNumberOfHalves);
  disk = 0;
  while (index > mLastSensorIndex[disk]) {
    disk++;
  }
  index -= getFirstSensorIndex(disk);
  Int_t nSensor = MinSensorsPerLadder;
  Int_t nFirstSensorIndex = 0, nFirstSensorIndexSave = 0;
  ladder = 0;
  while (index > ((nFirstSensorIndex += nSensor*mNumberOfLadders[disk][nSensor])-1)) {
    ladder += mNumberOfLadders[disk][nSensor];
    nFirstSensorIndexSave = nFirstSensorIndex;
    nSensor++; 
  }
  index -= nFirstSensorIndexSave;
  ladder += index/nSensor;
  sensor = index%nSensor;

  return kTRUE;

}

//__________________________________________________________________________
Int_t GeometryTGeo::getHalf(Int_t index) const
{

  Int_t half = index/(mTotalNumberOfSensors/mNumberOfHalves);

  return half;

}


//__________________________________________________________________________
Int_t GeometryTGeo::getDisk(Int_t index) const
{

  index = index%(mTotalNumberOfSensors/mNumberOfHalves);
  Int_t disk = 0;
  while (index > mLastSensorIndex[disk]) {
    disk++;
  }

  return disk;

}

//__________________________________________________________________________
Int_t GeometryTGeo::getLadder(Int_t index) const
{

  index = index%(mTotalNumberOfSensors/mNumberOfHalves);
  Int_t disk = 0;
  while (index > mLastSensorIndex[disk]) {
    disk++;
  }

  index -= getFirstSensorIndex(disk);

  Int_t nSensor = MinSensorsPerLadder;
  Int_t ladder = 0, nFirstSensorIndex = 0, nFirstSensorIndexSave = 0;
  while (index > ((nFirstSensorIndex += nSensor*mNumberOfLadders[disk][nSensor])-1)) {
    ladder += mNumberOfLadders[disk][nSensor];
    nFirstSensorIndexSave = nFirstSensorIndex;
    nSensor++; 
  }
  index -= nFirstSensorIndexSave;
  ladder += index/nSensor;

  return ladder;

}

//__________________________________________________________________________
Int_t GeometryTGeo::getSensorIndex(Int_t halfID, Int_t diskID, Int_t ladderID, Int_t sensorID) const
{

  Int_t index = 0;
  Int_t ladder = mLadderId2Index[diskID][ladderID];

  Int_t nL = 0;
  Int_t nS = MinSensorsPerLadder;
  while (ladder > ((nL += mNumberOfLadders[diskID][nS])-1)) {
    index += nS*mNumberOfLadders[diskID][nS];
    nS++;
  }
  ladder -= nL-mNumberOfLadders[diskID][nS];

  index += ladder*nS;
  index += sensorID;
  index += getFirstSensorIndex(diskID);
  index += halfID*mTotalNumberOfSensors/2;

  return index;

}
