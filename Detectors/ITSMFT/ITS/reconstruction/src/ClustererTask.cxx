// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  ClustererTask.cxx
/// \brief Implementation of the ITS cluster finder task

#include "ITSMFTBase/SegmentationPixel.h"
#include "ITSReconstruction/ClustererTask.h"

#include "FairLogger.h"      // for LOG
#include "FairRootManager.h" // for FairRootManager
#include "TClonesArray.h"    // for TClonesArray

ClassImp(o2::ITS::ClustererTask)

using o2::ITSMFT::SegmentationPixel;
using namespace o2::ITS;

//_____________________________________________________________________
ClustererTask::ClustererTask() : FairTask("ITSClustererTask"), mClustersArray(nullptr) {}

//_____________________________________________________________________
ClustererTask::~ClustererTask()
{
  if (mClustersArray) {
    mClustersArray->Delete();
    delete mClustersArray;
  }
}

//_____________________________________________________________________
/// \brief Init function
/// Inititializes the clusterer and connects input and output container
InitStatus ClustererTask::Init()
{
  FairRootManager* mgr = FairRootManager::Instance();
  if (!mgr) {
    LOG(ERROR) << "Could not instantiate FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  TClonesArray *arr = dynamic_cast<TClonesArray*>(mgr->GetObject("ITSDigit"));
  if (!arr) {
    LOG(ERROR)<<"ITS digits not registered in the FairRootManager. Exiting ..."<<FairLogger::endl;
    return kERROR;
  }
  mReader.setDigitArray(arr);
  
  // Register output container
  mClustersArray = new TClonesArray("o2::ITS::Cluster");
  mgr->Register("ITSCluster", "ITS", mClustersArray, kTRUE);

  mGeometry.Build(kTRUE);
  const SegmentationPixel* seg = (SegmentationPixel*)mGeometry.getSegmentationById(0);

  Float_t px = seg->cellSizeX();
  Float_t pz = seg->cellSizeZ(1);  //FIXME
  Float_t x0,z0; seg->detectorToLocal(0,0,x0,z0);
  mClusterer.setPixelGeometry(px,pz,x0,z0);
  
  return kSUCCESS;
}

//_____________________________________________________________________
void ClustererTask::Exec(Option_t* option)
{
  mClustersArray->Clear();
  LOG(DEBUG) << "Running digitization on new event" << FairLogger::endl;

  mClusterer.process(mReader, *mClustersArray);
}
