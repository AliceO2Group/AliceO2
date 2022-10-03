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

/// \file  ClustererTask.cxx
/// \brief Implementation of the ITS cluster finder task

#include "DetectorsCommonDataFormats/DetID.h"
#include "ITSReconstruction/ClustererTask.h"
#include "MathUtils/Cartesian.h"
#include "MathUtils/Utils.h"
#include <fairlogger/Logger.h>
#include <TFile.h>
#include <TTree.h>

using namespace o2::its;

//_____________________________________________________________________
ClustererTask::ClustererTask(bool useMC, bool raw) : mRawDataMode(raw),
                                                     mUseMCTruth(useMC && (!raw))
{
  LOG(info) << Class()->GetName() << ": MC digits mode: " << (mRawDataMode ? "OFF" : "ON")
            << " | Use MCtruth: " << (mUseMCTruth ? "ON" : "OFF");

  mClusterer.setNChips(o2::itsmft::ChipMappingITS::getNChips());
}

//_____________________________________________________________________
ClustererTask::~ClustererTask()
{
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
    mReaderRaw = std::make_unique<o2::itsmft::RawPixelReader<o2::itsmft::ChipMappingITS>>();
    mReader = mReaderRaw.get();
  } else { // clusterizer of digits
    mReaderMC = std::make_unique<o2::itsmft::DigitPixelReader>();
    mReader = mReaderMC.get();
  }

  mClusterer.print();

  return;
}

//_____________________________________________________________________
void ClustererTask::run(const std::string inpName, const std::string outName)
{
  // standalone execution
  Init(); // create reader, clusterer

  if (mRawDataMode) {

    mReaderRaw->openInput(inpName);
    mClusterer.process(1, *mReaderRaw.get(), &mCompClus, &mPatterns, &mROFRecVec, nullptr);

    auto basename = outName.substr(0, outName.size() - sizeof("root"));
    auto nFiles = int(mROFRecVec.size() / maxROframe);
    int i = 0;
    for (; i < nFiles; i++) {
      writeTree(basename, i);
    }
    writeTree(basename, i); // The remainder

  } else {

    mReaderMC->openInput(inpName, o2::detectors::DetID("ITS"));

    TFile outFile(outName.data(), "new");
    if (!outFile.IsOpen()) {
      LOG(fatal) << "Failed to open output file " << outName;
    }

    TTree outTree("o2sim", "ITS Clusters");

    auto compClusPtr = &mCompClus;
    outTree.Branch("ITSClusterComp", &compClusPtr);

    auto rofRecVecPtr = &mROFRecVec;
    outTree.Branch("ITSClustersROF", &rofRecVecPtr);

    auto clsLabelsPtr = &mClsLabels;
    if (mUseMCTruth && mReaderMC->getDigitsMCTruth()) {
      // digit labels are provided directly to clusterer
      outTree.Branch("ITSClusterMCTruth", &clsLabelsPtr);
    } else {
      mUseMCTruth = false;
    }
    LOG(info) << Class()->GetName() << " | MCTruth: " << (mUseMCTruth ? "ON" : "OFF");

    outTree.Branch("ITSClusterPatt", &mPatterns);

    std::vector<o2::itsmft::MC2ROFRecord> mc2rof, *mc2rofPtr = &mc2rof;
    if (mUseMCTruth) {
      auto mc2rofOrig = mReaderMC->getMC2ROFRecords();
      mc2rof.reserve(mc2rofOrig.size());
      for (const auto& m2r : mc2rofOrig) { // clone from the span
        mc2rof.push_back(m2r);
      }
      outTree.Branch("ITSClustersMC2ROF", mc2rofPtr);
    }

    // loop over entries of the input tree
    while (mReaderMC->readNextEntry()) {
      mClusterer.process(1, *mReaderMC.get(), &mCompClus, &mPatterns, &mROFRecVec, &mClsLabels);
    }

    outTree.Fill();
    outTree.Write();
  }

  mClusterer.clear();
}

void ClustererTask::writeTree(std::string basename, int i)
{
  auto name = basename + std::to_string(i) + ".root";
  TFile outFile(name.data(), "new");
  if (!outFile.IsOpen()) {
    LOG(fatal) << "Failed to open output file " << name;
  }
  TTree outTree("o2sim", "ITS Clusters");

  size_t max = (i + 1) * maxROframe;
  auto lastf = (max < mROFRecVec.size()) ? mROFRecVec.begin() + max : mROFRecVec.end();
  std::vector<o2::itsmft::ROFRecord> rofRecBuffer(mROFRecVec.begin() + i * maxROframe, lastf);
  std::vector<o2::itsmft::ROFRecord>* rofRecPtr = &rofRecBuffer;
  outTree.Branch("ITSClustersROF", rofRecPtr);

  auto first = rofRecBuffer[0].getFirstEntry();
  auto last = rofRecBuffer.back().getFirstEntry() + rofRecBuffer.back().getNEntries();

  std::vector<CompClusterExt> compClusBuffer, *compClusPtr = &compClusBuffer;
  compClusBuffer.assign(&mCompClus[first], &mCompClus[last]);
  outTree.Branch("ITSClusterComp", &compClusPtr);
  outTree.Branch("ITSClusterPatt", &mPatterns);

  for (auto& rof : rofRecBuffer) {
    rof.setFirstEntry(rof.getFirstEntry() - first);
  }

  outTree.Fill();
  outTree.Write();
}
