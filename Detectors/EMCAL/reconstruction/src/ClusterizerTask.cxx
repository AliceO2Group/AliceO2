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

/// \file  ClusterizerTask.cxx
/// \brief Implementation of the EMCAL cluster finder task

#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <gsl/span>
#include <fairlogger/Logger.h> // for LOG
#include "FairRootManager.h" // for FairRootManager
#include "EMCALReconstruction/ClusterizerTask.h"
#include "DataFormatsEMCAL/Cluster.h"

#include <TFile.h>

//ClassImp(o2::emcal::ClusterizerTask);

using namespace o2::emcal;

//_____________________________________________________________________
template <class InputType>
ClusterizerTask<InputType>::ClusterizerTask(ClusterizerParameters* parameters) : mClusterizer(parameters->getTimeCut(), parameters->getTimeMin(), parameters->getTimeMax(), parameters->getGradientCut(), parameters->getDoEnergyGradientCut(), parameters->getThresholdSeedEnergy(), parameters->getThresholdCellEnergy())
{
}

//_____________________________________________________________________
/// \brief Init function
/// Inititializes the cell/digit reader & geometry in cluster finder
template <class InputType>
void ClusterizerTask<InputType>::init()
{
  // Initialize cell/digit reader
  if (!mInputReader) {
    mInputReader = std::make_unique<DigitReader<InputType>>();
  }

  // Get default geometry object if not yet set
  if (!mGeometry) {
    mGeometry = Geometry::GetInstanceFromRunNumber(223409); // NOTE: Hardcoded for run II run
  }
  if (!mGeometry) {
    LOG(error) << "Failure accessing geometry";
  }

  // Set geometry object in clusterizer
  if (!mClusterizer.getGeometry()) {
    mClusterizer.setGeometry(mGeometry);
  }
  if (!mClusterizer.getGeometry()) {
    LOG(error) << "Could not set geometry in clusterizer";
  }

  mClustersArray = new std::vector<o2::emcal::Cluster>();
  mClustersInputIndices = new std::vector<o2::emcal::ClusterIndex>();
  mClusterTriggerRecordsClusters = new std::vector<o2::emcal::TriggerRecord>();
  mClusterTriggerRecordsIndices = new std::vector<o2::emcal::TriggerRecord>();
}

//_____________________________________________________________________
template <class InputType>
void ClusterizerTask<InputType>::process(const std::string inputFileName, const std::string outputFileName)
{
  LOG(debug) << "Running clusterization on new event";

  // Create reader, initilize clusterizer geometry
  init();

  // Load output file
  std::unique_ptr<TFile> outFile(TFile::Open(outputFileName.data(), "recreate"));
  if (!outFile || outFile->IsZombie()) {
    LOG(fatal) << "Failed to open output file " << outputFileName;
  }

  // Create output tree
  std::unique_ptr<TTree> outTree = std::make_unique<TTree>("o2sim", "EMCAL clusters");
  outTree->Branch("EMCALCluster", &mClustersArray);
  outTree->Branch("EMCALClusterInputIndex", &mClustersInputIndices);
  outTree->Branch("EMCALClusterTRGR", &mClusterTriggerRecordsClusters);
  outTree->Branch("EMCIndicesTRGR", &mClusterTriggerRecordsIndices);

  mClustersArray->clear();
  mClustersInputIndices->clear();
  mClusterTriggerRecordsClusters->clear();
  mClusterTriggerRecordsIndices->clear();

  // Loop over entries of the input tree
  mInputReader->openInput(inputFileName);
  while (mInputReader->readNextEntry()) {

    auto InputVector = mInputReader->getInputArray();

    int currentStartClusters = mClustersArray->size();
    int currentStartIndices = mClustersInputIndices->size();

    for (auto iTrgRcrd = mInputReader->getTriggerArray()->begin(); iTrgRcrd != mInputReader->getTriggerArray()->end(); ++iTrgRcrd) {

      mClusterizer.findClusters(gsl::span<const InputType>(InputVector->data() + iTrgRcrd->getFirstEntry(), iTrgRcrd->getNumberOfObjects())); // Find clusters on cells/digits given in reader::mInputArray (pass by ref)

      // Get found clusters + cell/digit indices for output
      auto clusterstmp = mClusterizer.getFoundClusters();
      auto clusterIndecestmp = mClusterizer.getFoundClustersInputIndices();
      std::copy(clusterstmp->begin(), clusterstmp->end(), std::back_inserter(*mClustersArray));
      std::copy(clusterIndecestmp->begin(), clusterIndecestmp->end(), std::back_inserter(*mClustersInputIndices));

      mClusterTriggerRecordsClusters->emplace_back(iTrgRcrd->getBCData(), currentStartClusters, clusterstmp->size());
      mClusterTriggerRecordsIndices->emplace_back(iTrgRcrd->getBCData(), currentStartIndices, clusterIndecestmp->size());

      currentStartClusters = mClustersArray->size();
      currentStartIndices = mClustersInputIndices->size();
    }
    outTree->Fill();
  }

  // Write, close, and destroy tree/file
  outTree->Write();
  outTree.reset(); // here we reset the unique ptr, not the tree!
  outFile->Close();
}

template class o2::emcal::ClusterizerTask<o2::emcal::Cell>;
template class o2::emcal::ClusterizerTask<o2::emcal::Digit>;
