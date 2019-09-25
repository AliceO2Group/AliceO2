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
#include <TFile.h>
#include <TTree.h>

using namespace o2::mft;
using namespace o2::base;
using namespace o2::utils;

//_____________________________________________________________________
ClustererTask::ClustererTask(bool useMC, bool raw) : mRawDataMode(raw),
                                                     mUseMCTruth(useMC && (!raw))
{
  LOG(INFO) << Class()->GetName() << ": MC digits mode: " << (mRawDataMode ? "OFF" : "ON")
            << " | Use MCtruth: " << (mUseMCTruth ? "ON" : "OFF");

  mClusterer.setNChips(o2::itsmft::ChipMappingMFT::getNChips());
}

//_____________________________________________________________________
ClustererTask::~ClustererTask()
{
  mFullClus.clear();
  mCompClus.clear();
  mClsLabels.clear();
}

//_____________________________________________________________________
void ClustererTask::Init()
{
  /// Inititializes the clusterer and connects input and output container

  if (mReader) {
    return; // already initialized
  }

  // create reader according to requested raw of MC mode
  if (mRawDataMode) {
    mReaderRaw = std::make_unique<o2::itsmft::RawPixelReader<o2::itsmft::ChipMappingMFT>>();
    mReader = mReaderRaw.get();
  } else { // clusterizer of digits
    mReaderMC = std::make_unique<o2::itsmft::DigitPixelReader>();
    mReader = mReaderMC.get();
  }

  GeometryTGeo* geom = GeometryTGeo::Instance();
  geom->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2L)); // make sure T2L matrices are loaded
  mGeometry = geom;
  mClusterer.setGeometry(geom);

  mClusterer.print();

  return;
}

//_____________________________________________________________________
void ClustererTask::run(const std::string inpName, const std::string outName, bool entryPerROF)
{
  // standalone execution
  setSelfManagedMode(true);
  Init(); // create reader, clusterer

  std::unique_ptr<TFile> outFile(TFile::Open(outName.data(), "recreate"));
  if (!outFile || outFile->IsZombie()) {
    LOG(FATAL) << "Failed to open output file " << outName;
  }
  std::unique_ptr<TTree> outTree = std::make_unique<TTree>("o2sim", "MFT Clusters");
  std::unique_ptr<TTree> outTreeROF = std::make_unique<TTree>("MFTClustersROF", "ROF records tree");

  if (mClusterer.getWantFullClusters()) {
    mFullClusPtr = &mFullClus;
    outTree->Branch("MFTCluster", &mFullClusPtr);
    LOG(INFO) << Class()->GetName() << " output of full clusters is requested";
  } else {
    LOG(INFO) << Class()->GetName() << " output of full clusters is not requested";
  }

  if (mClusterer.getWantCompactClusters()) {
    mCompClusPtr = &mCompClus;
    outTree->Branch("MFTClusterComp", &mCompClusPtr);
    LOG(INFO) << Class()->GetName() << " output of compact clusters is requested";
  } else {
    LOG(INFO) << Class()->GetName() << " output of compact clusters is not requested";
  }

  mROFRecVecPtr = &mROFRecVec;
  outTreeROF->Branch("MFTClustersROF", mROFRecVecPtr);

  if (entryPerROF) {
    mClusterer.setOutputTree(outTree.get()); // this will force flushing at every ROF
  }

  if (mRawDataMode) {
    mReaderRaw->openInput(inpName);
    mClusterer.process(*mReaderRaw.get(), mFullClusPtr, mCompClusPtr, nullptr, mROFRecVecPtr);
  } else {
    mReaderMC->openInput(inpName, o2::detectors::DetID("MFT"));

    if (mUseMCTruth && !(mClusterer.getWantFullClusters() || mClusterer.getWantCompactClusters())) {
      mUseMCTruth = false;
      LOG(WARNING) << "MFT clusters storage is not requested, suppressing MCTruth storage";
    }

    if (mUseMCTruth && mReaderMC->getDigitsMCTruth()) {
      // digit labels are provided directly to clusterer
      mClsLabelsPtr = &mClsLabels;
      outTree->Branch("MFTClusterMCTruth", &mClsLabelsPtr);
    } else {
      mUseMCTruth = false;
    }
    LOG(INFO) << Class()->GetName() << " | MCTruth: " << (mUseMCTruth ? "ON" : "OFF");

    // loop over entries of the input tree
    while (mReaderMC->readNextEntry()) {
      mClusterer.process(*mReaderMC.get(), mFullClusPtr, mCompClusPtr, mClsLabelsPtr, mROFRecVecPtr);
    }
  }

  if (!mRawDataMode && mReaderMC->getMC2ROFRecords()) {
    std::unique_ptr<TTree> outTreeMC2ROF = std::make_unique<TTree>("MFTClustersMC2ROF", "MC->ROF records tree");
    auto mc2rof = *mReaderMC->getMC2ROFRecords(); // clone
    auto* mc2rofPtr = &mc2rof;
    outTreeMC2ROF->Branch("MFTClustersMC2ROF", &mc2rofPtr);
    outTreeMC2ROF->Fill();
    outTreeMC2ROF->Write();
  }

  if (!entryPerROF) { // Clustered was not managing the output, do it here
    outTree->Fill();  // in this mode all ROF will go to single entry of the tree
  }
  outTreeROF->Fill(); // ROF records are stored as a single vector

  outTree->Write();
  outTreeROF->Write();

  outTree.reset(); // tree should be destroyed before the file is closed
  outTreeROF.reset();

  outFile->Close();

  mClusterer.clear();
}
