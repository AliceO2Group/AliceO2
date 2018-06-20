// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClustererTrack.h
/// \brief Cluster finding from digits (MFT)
/// \author bogdan.vulpescu@cern.ch
/// \date 03/05/2017

#include "DetectorsCommonDataFormats/DetID.h"
#include "MFTReconstruction/ClustererTask.h"
#include "MFTBase/Constants.h"
#include "MFTBase/Geometry.h"

#include "FairLogger.h"
#include "FairRootManager.h"

ClassImp(o2::MFT::ClustererTask);

using namespace o2::MFT;
using namespace o2::Base;
using namespace o2::utils;

//_____________________________________________________________________
ClustererTask::ClustererTask(bool useMC, bool raw) : FairTask("MFTClustererTask"),
                                                     mRawDataMode(raw),
                                                     mUseMCTruth(useMC && (!raw))
{
  LOG(INFO) << Class()->GetName() << ": MC digits mode: " << (mRawDataMode ? "OFF" : "ON")
            << " | Use MCtruth: " << (mUseMCTruth ? "ON" : "OFF") << FairLogger::endl;

  mClusterer.setNChips(o2::ITSMFT::ChipMappingMFT::getNChips());
}

//_____________________________________________________________________
ClustererTask::~ClustererTask()
{
  mFullClus.clear();
  mCompClus.clear();
  mClsLabels.clear();
}

//_____________________________________________________________________
InitStatus ClustererTask::Init()
{
  /// Inititializes the clusterer and connects input and output container

  if (mReader) {
    return kSUCCESS; // already initialized
  }

  // create reader according to requested raw of MC mode
  if (mRawDataMode) {
    mReaderRaw = std::make_unique<o2::ITSMFT::RawPixelReader<o2::ITSMFT::ChipMappingMFT>>();
    mReader = mReaderRaw.get();
  } else { // clusterizer of digits needs input from the FairRootManager (at the moment)
    mReaderMC = std::make_unique<o2::ITSMFT::DigitPixelReader>();
    mReader = mReaderMC.get();

    if (!isSelfManagedMode()) {
      attachFairManagerIO();
    }
  }

  GeometryTGeo* geom = GeometryTGeo::Instance();
  geom->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2L)); // make sure T2L matrices are loaded
  mGeometry = geom;
  mClusterer.setGeometry(geom);

  mClusterer.print();

  return kSUCCESS;
}

//_____________________________________________________________________
void ClustererTask::attachFairManagerIO()
{
  // attachs input/output containers from the FairManager

  if (mRawDataMode) {
    LOG(FATAL) << "FairRootManager I/O is not supported in raw data mode" << FairLogger::endl;
  }

  FairRootManager* mgr = FairRootManager::Instance();
  if (!mgr) {
    LOG(FATAL) << "Could not instantiate FairRootManager. Exiting ..." << FairLogger::endl;
  }

  const auto arrDig = mgr->InitObjectAs<const std::vector<o2::ITSMFT::Digit>*>("MFTDigit");
  if (!arrDig) {
    LOG(FATAL) << "MFT digits are not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
  }
  mReaderMC->setDigits(arrDig);

  if (mUseMCTruth && !(mClusterer.getWantFullClusters() || mClusterer.getWantCompactClusters())) {
    mUseMCTruth = false;
    LOG(WARNING) << "MFT clusters storage is not requested, suppressing MCTruth storage" << FairLogger::endl;
  }

  const auto arrDigLbl = mUseMCTruth ? mgr->InitObjectAs<const MCTruth*>("ITSDigitMCTruth") : nullptr;

  if (!arrDigLbl && mUseMCTruth) {
    LOG(WARNING) << "MFT digits labeals are not registered in the FairRootManager. Continue w/o MC truth ..."
                 << FairLogger::endl;
  }
  mReaderMC->setDigitsMCTruth(arrDigLbl);

  // Register output container
  if (mClusterer.getWantFullClusters()) {
    mFullClusPtr = &mFullClus;
    mgr->RegisterAny("MFTCluster", mFullClusPtr, kTRUE);
    LOG(INFO) << Class()->GetName() << " output of full clusters is requested " << FairLogger::endl;
  } else {
    LOG(INFO) << Class()->GetName() << " output of full clusters is not requested " << FairLogger::endl;
  }

  if (mClusterer.getWantCompactClusters()) {
    mCompClusPtr = &mCompClus;
    mgr->RegisterAny("MFTClusterComp", mCompClusPtr, kTRUE);
    LOG(INFO) << Class()->GetName() << " output of compact clusters is requested " << FairLogger::endl;
  } else {
    LOG(INFO) << Class()->GetName() << " output of compact clusters is not requested " << FairLogger::endl;
  }

  // Register output MC Truth container if there is an MC truth for digits
  if (arrDigLbl) {
    mClsLabelsPtr = &mClsLabels;
    mgr->RegisterAny("MFTClusterMCTruth", mClsLabelsPtr, kTRUE);
  }
  LOG(INFO) << Class()->GetName() << " | MCTruth: " << (arrDigLbl ? "ON" : "OFF") << FairLogger::endl;
}

//_____________________________________________________________________
void ClustererTask::Exec(Option_t* option)
{
  /// execution entry point used when managed by the FairRunAna
  if (mFullClusPtr) {
    mFullClusPtr->clear();
  }
  if (mCompClusPtr) {
    mCompClusPtr->clear();
  }
  if (mClsLabelsPtr) {
    mClsLabelsPtr->clear();
  }
  LOG(DEBUG) << "Running clusterization on new event" << FairLogger::endl;
  mReader->init(); // needed when we start with fresh input data, as in case of FairRoort input
  mClusterer.process(*mReader, mFullClusPtr, mCompClusPtr, mClsLabelsPtr);
}

//_____________________________________________________________________
void ClustererTask::run(const std::string inpName, const std::string outName, bool entryPerROF)
{
  // standalone execution
  setSelfManagedMode(true); // to prevent interference with FairRootManager io
  Init();                   // create reader, clusterer

  std::unique_ptr<TFile> outFile(TFile::Open(outName.data(), "recreate"));
  if (!outFile || outFile->IsZombie()) {
    LOG(FATAL) << "Failed to open output file " << outName << FairLogger::endl;
  }
  std::unique_ptr<TTree> outTree = std::make_unique<TTree>("o2sim", "MFT Clusters");

  if (mClusterer.getWantFullClusters()) {
    mFullClusPtr = &mFullClus;
    outTree->Branch("MFTCluster", &mFullClusPtr);
    LOG(INFO) << Class()->GetName() << " output of full clusters is requested " << FairLogger::endl;
  } else {
    LOG(INFO) << Class()->GetName() << " output of full clusters is not requested " << FairLogger::endl;
  }

  if (mClusterer.getWantCompactClusters()) {
    mCompClusPtr = &mCompClus;
    outTree->Branch("MFTClusterComp", &mCompClusPtr);
    LOG(INFO) << Class()->GetName() << " output of compact clusters is requested " << FairLogger::endl;
  } else {
    LOG(INFO) << Class()->GetName() << " output of compact clusters is not requested " << FairLogger::endl;
  }

  if (entryPerROF) {
    mClusterer.setOutputTree(outTree.get()); // this will force flushing at every ROF
  }

  if (mRawDataMode) {
    mReaderRaw->openInput(inpName);
    mClusterer.process(*mReaderRaw.get(), mFullClusPtr, mCompClusPtr);
  } else {
    mReaderMC->openInput(inpName, o2::detectors::DetID("MFT"));

    if (mUseMCTruth && !(mClusterer.getWantFullClusters() || mClusterer.getWantCompactClusters())) {
      mUseMCTruth = false;
      LOG(WARNING) << "MFT clusters storage is not requested, suppressing MCTruth storage" << FairLogger::endl;
    }

    if (mUseMCTruth && mReaderMC->getDigitsMCTruth()) {
      // digit labels are provided directly to clusterer
      mClsLabelsPtr = &mClsLabels;
      outTree->Branch("MFTClusterMCTruth", &mClsLabelsPtr);
    } else {
      mUseMCTruth = false;
    }
    LOG(INFO) << Class()->GetName() << " | MCTruth: " << (mUseMCTruth ? "ON" : "OFF") << FairLogger::endl;

    // loop over entries of the input tree
    while (mReaderMC->readNextEntry()) {
      mClusterer.process(*mReaderMC.get(), mFullClusPtr, mCompClusPtr, mClsLabelsPtr);
    }
  }

  if (!entryPerROF) { // Clustered was not managing the output, do it here
    outTree->Fill();  // in this mode all ROF will go to single entry of the tree
  }
  outTree->Write();
  outTree.reset(); // tree should be destroyed before the file is closed
  outFile->Close();

  mClusterer.clear();
}
