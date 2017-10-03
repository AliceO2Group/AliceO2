// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file FindClusters.h
/// \brief Cluster finding from digits (ITS)
/// \author bogdan.vulpescu@cern.ch 
/// \date 03/05/2017

#include "ITSMFTBase/SegmentationPixel.h"

#include "MFTBase/Constants.h"
#include "MFTBase/Geometry.h"
#include "MFTSimulation/EventHeader.h"
#include "MFTReconstruction/ClustererTask.h"

#include "FairLogger.h"
#include "FairRootManager.h"
#include "TClonesArray.h"

ClassImp(o2::MFT::ClustererTask)

using o2::ITSMFT::SegmentationPixel;
using namespace o2::MFT;
using namespace o2::Base;
using namespace o2::Base::Utils;

//_____________________________________________________________________________
ClustererTask::ClustererTask() : FairTask("MFTClustererTask")
{

}

//_____________________________________________________________________________
ClustererTask::~ClustererTask()
{

  if (mClustersArray) {
    mClustersArray->Delete();
    delete mClustersArray;
  }

}

//_____________________________________________________________________________
InitStatus ClustererTask::Init()
{

  FairRootManager* mgr = FairRootManager::Instance();
  if (!mgr) {
    LOG(ERROR) << "Could not instantiate FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  TClonesArray *arr = dynamic_cast<TClonesArray*>(mgr->GetObject("MFTDigits"));
  if (!arr) {
    LOG(ERROR)<<"MFT digits not registered in the FairRootManager. Exiting ..."<<FairLogger::endl;
    return kERROR;
  }
  mReader.setDigitArray(arr);

  // Register output container
  mClustersArray = new TClonesArray("o2::ITSMFT::Cluster");
  mgr->Register("MFTClusters", "MFT", mClustersArray, kTRUE);

  GeometryTGeo* geom = GeometryTGeo::Instance();
  geom->fillMatrixCache( bit2Mask(TransformType::T2L) ); // make sure T2L matrices are loaded
  mGeometry = geom;
  mClusterer.setGeometry(geom);

  const SegmentationPixel* seg = (SegmentationPixel*)mGeometry->getSegmentationById(0);

  Float_t px = seg->cellSizeX();
  Float_t pz = seg->cellSizeZ(1);  //FIXME
  Float_t x0,z0; seg->detectorToLocal(0,0,x0,z0);
  mClusterer.setPixelGeometry(px,pz,x0,z0);

  return kSUCCESS;

}

//_____________________________________________________________________________
void ClustererTask::Exec(Option_t* /*opt*/) 
{

  mClustersArray->Clear();
  LOG(DEBUG) << "Running digitization on new event" << FairLogger::endl;

  mClusterer.process(mReader, *mClustersArray);

}

