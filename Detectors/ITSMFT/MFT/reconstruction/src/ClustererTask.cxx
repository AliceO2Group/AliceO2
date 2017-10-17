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

#include "MFTBase/Constants.h"
#include "MFTBase/Geometry.h"
#include "MFTSimulation/EventHeader.h"
#include "MFTReconstruction/ClustererTask.h"
//#include "DetectorsBase/Utils.h"
//#include "MathUtils/Cartesian3D.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#include "FairLogger.h"
#include "FairRootManager.h"
#include "TClonesArray.h"

ClassImp(o2::MFT::ClustererTask)

using namespace o2::MFT;
using namespace o2::Base;
using namespace o2::Base::Utils;

//_____________________________________________________________________________
ClustererTask::ClustererTask(Bool_t useMCTruth) : FairTask("MFTClustererTask")
{

  if (useMCTruth)
    mClsLabels = new o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
  
}

//_____________________________________________________________________________
ClustererTask::~ClustererTask()
{

  if (mClustersArray) {
    mClustersArray->clear();
    delete mClustersArray;
  }
  if (mClsLabels) {
    mClsLabels->clear();
    delete mClsLabels;
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

  TClonesArray *arr = dynamic_cast<TClonesArray*>(mgr->GetObject("MFTDigit"));
  if (!arr) {
    LOG(ERROR)<<"MFT digits not registered in the FairRootManager. Exiting ..."<<FairLogger::endl;
    return kERROR;
  }
  mReader.setDigitArray(arr);

  // Register output container
  mgr->RegisterAny("MFTCluster", mClustersArray, kTRUE);

  // Register MC Truth container
  if (mClsLabels) {
    mgr->Register("MFTClusterMCTruth", "MFT", mClsLabels, kTRUE);
  }

  GeometryTGeo* geom = GeometryTGeo::Instance();
  geom->fillMatrixCache( bit2Mask(TransformType::T2L) ); // make sure T2L matrices are loaded
  mGeometry = geom;
  mClusterer.setGeometry(geom);
  mClusterer.setMCTruthContainer(mClsLabels);

  return kSUCCESS;

}

//_____________________________________________________________________________
void ClustererTask::Exec(Option_t* /*opt*/) 
{

  if (mClustersArray) mClustersArray->clear();
  if (mClsLabels)  mClsLabels->clear();
  LOG(DEBUG) << "Running digitization on new event" << FairLogger::endl;

  mClusterer.process(mReader, *mClustersArray);

}

