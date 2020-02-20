// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  ClusterizerTask.cxx
/// \brief Implementation of the EMCAL cluster finder task

#include "FairLogger.h"      // for LOG
#include "FairRootManager.h" // for FairRootManager
#include "EMCALReconstruction/ClusterizerTask.h"

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
    LOG(ERROR) << "Failure accessing geometry";
  }

  // Set geometry object in clusterizer
  if (!mClusterizer.getGeometry()) {
    mClusterizer.setGeometry(mGeometry);
  }
  if (!mClusterizer.getGeometry()) {
    LOG(ERROR) << "Could not set geometry in clusterizer";
  }
}

//_____________________________________________________________________
template <class InputType>
void ClusterizerTask<InputType>::process(const std::string inputFileName, const std::string outputFileName)
{
  LOG(DEBUG) << "Running clusterization on new event";

  // Create reader, initilize clusterizer geometry
  init();

  // Load output file
  std::unique_ptr<TFile> outFile(TFile::Open(outputFileName.data(), "recreate"));
  if (!outFile || outFile->IsZombie()) {
    LOG(FATAL) << "Failed to open output file " << outputFileName;
  }

  // Create output tree
  std::unique_ptr<TTree> outTree = std::make_unique<TTree>("o2sim", "EMCAL clusters");
  outTree->Branch("EMCALCluster", &mClustersArray);
  outTree->Branch("EMCALClusterInputIndices", &mClustersInputIndices);

  // Loop over entries of the input tree
  mInputReader->openInput(inputFileName);
  while (mInputReader->readNextEntry()) {
    mClusterizer.findClusters(*mInputReader->getInputArray()); // Find clusters on cells/digits given in reader::mInputArray (pass by ref)

    // Get found clusters + cell/digit indices for output
    mClustersArray = mClusterizer.getFoundClusters();
    mClustersInputIndices = mClusterizer.getFoundClustersInputIndices();
    outTree->Fill();
  }

  // Write, close, and destroy tree/file
  outTree->Write();
  outTree.reset(); // here we reset the unique ptr, not the tree!
  outFile->Close();
}

template class o2::emcal::ClusterizerTask<o2::emcal::Cell>;
template class o2::emcal::ClusterizerTask<o2::emcal::Digit>;
