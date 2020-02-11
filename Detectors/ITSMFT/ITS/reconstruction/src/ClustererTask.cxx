// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  ClustererTask.cxx
/// \brief Implementation of the ITS cluster finder task

#include "DetectorsCommonDataFormats/DetID.h"
#include "ITSReconstruction/ClustererTask.h"
#include "MathUtils/Cartesian3D.h"
#include "MathUtils/Utils.h"
#include "FairLogger.h"
#include <TFile.h>
#include <TTree.h>

using namespace o2::its;
using namespace o2::base;
using namespace o2::utils;

//_____________________________________________________________________
ClustererTask::ClustererTask(bool useMC, bool raw) : mRawDataMode(raw),
                                                     mUseMCTruth(useMC && (!raw))
{
  LOG(INFO) << Class()->GetName() << ": MC digits mode: " << (mRawDataMode ? "OFF" : "ON")
            << " | Use MCtruth: " << (mUseMCTruth ? "ON" : "OFF");

  mClusterer.setNChips(o2::itsmft::ChipMappingITS::getNChips());
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
    mReaderRaw = std::make_unique<o2::itsmft::RawPixelReader<o2::itsmft::ChipMappingITS>>();
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
void ClustererTask::run(const std::string inpName, const std::string outName)
{
  // standalone execution
  Init(); // create reader, clusterer

  if (mRawDataMode) {

    mReaderRaw->openInput(inpName);
    mClusterer.process(*mReaderRaw.get(), &mFullClus, &mCompClus, nullptr, &mROFRecVec);

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
      LOG(FATAL) << "Failed to open output file " << outName;
    }

    TTree outTree("o2sim", "ITS Clusters");

    auto fullClusPtr = &mFullClus;
    if (mClusterer.getWantFullClusters()) {
      outTree.Branch("ITSCluster", &fullClusPtr);
    } else {
      LOG(INFO) << Class()->GetName() << " output of full clusters is not requested";
    }

    auto compClusPtr = &mCompClus;
    if (mClusterer.getWantCompactClusters()) {
      outTree.Branch("ITSClusterComp", &compClusPtr);
    } else {
      LOG(INFO) << Class()->GetName() << " output of compact clusters is not requested";
    }

    auto rofRecVecPtr = &mROFRecVec;
    outTree.Branch("ITSClustersROF", &rofRecVecPtr);

    if (mUseMCTruth && !(mClusterer.getWantFullClusters() || mClusterer.getWantCompactClusters())) {
      mUseMCTruth = false;
      LOG(WARNING) << "ITS clusters storage is not requested, suppressing MCTruth storage";
    }

    auto clsLabelsPtr = &mClsLabels;
    if (mUseMCTruth && mReaderMC->getDigitsMCTruth()) {
      // digit labels are provided directly to clusterer
      outTree.Branch("ITSClusterMCTruth", &clsLabelsPtr);
    } else {
      mUseMCTruth = false;
    }
    LOG(INFO) << Class()->GetName() << " | MCTruth: " << (mUseMCTruth ? "ON" : "OFF");

    // loop over entries of the input tree
    while (mReaderMC->readNextEntry()) {
      mClusterer.process(*mReaderMC.get(), &mFullClus, &mCompClus, &mClsLabels, &mROFRecVec);
    }

    std::vector<o2::itsmft::MC2ROFRecord> mc2rof, *mc2rofPtr = &mc2rof;
    if (mUseMCTruth) {
      auto mc2rofOrig = mReaderMC->getMC2ROFRecords();
      mc2rof.reserve(mc2rofOrig.size());
      for (const auto& m2r : mc2rofOrig) { // clone from the span
        mc2rof.push_back(m2r);
      }
      outTree.Branch("ITSClustersMC2ROF", mc2rofPtr);
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
    LOG(FATAL) << "Failed to open output file " << name;
  }
  TTree outTree("o2sim", "ITS Clusters");

  auto max = (i + 1) * maxROframe;
  auto lastf = (max < mROFRecVec.size()) ? mROFRecVec.begin() + max : mROFRecVec.end();
  std::vector<o2::itsmft::ROFRecord> rofRecBuffer(mROFRecVec.begin() + i * maxROframe, lastf);
  std::vector<o2::itsmft::ROFRecord>* rofRecPtr = &rofRecBuffer;
  outTree.Branch("ITSClustersROF", rofRecPtr);

  auto first = rofRecBuffer[0].getFirstEntry();
  auto last = rofRecBuffer.back().getFirstEntry() + rofRecBuffer.back().getNEntries();

  std::vector<Cluster> fullClusBuffer, *fullClusPtr = &fullClusBuffer;
  if (mClusterer.getWantFullClusters()) {
    fullClusBuffer.assign(&mFullClus[first], &mFullClus[last]);
    outTree.Branch("ITSCluster", &fullClusPtr);
  } else {
    LOG(INFO) << Class()->GetName() << " output of full clusters is not requested";
  }

  std::vector<CompClusterExt> compClusBuffer, *compClusPtr = &compClusBuffer;
  if (mClusterer.getWantCompactClusters()) {
    compClusBuffer.assign(&mCompClus[first], &mCompClus[last]);
    outTree.Branch("ITSClusterComp", &compClusPtr);
  } else {
    LOG(INFO) << Class()->GetName() << " output of compact clusters is not requested";
  }

  for (auto& rof : rofRecBuffer) {
    rof.setFirstEntry(rof.getFirstEntry() - first);
  }

  outTree.Fill();
  outTree.Write();
}
