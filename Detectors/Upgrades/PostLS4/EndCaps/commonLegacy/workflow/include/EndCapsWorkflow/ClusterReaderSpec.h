// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   ClusterReaderSpec.h

#ifndef O2_ENDCAPS_CLUSTERREADER
#define O2_ENDCAPS_CLUSTERREADER

#include "TFile.h"
#include "TTree.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Headers/DataHeader.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsITSMFT/ROFRecord.h"

using namespace o2::framework;

namespace o2
{
namespace endcaps
{

class ClusterReader : public Task
{
 public:
  ClusterReader() = delete;
  ClusterReader(o2::detectors::DetID id, bool useMC = true, bool useClFull = true, bool useClComp = true, bool usePatterns = true);
  ~ClusterReader() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 protected:
  void connectTree(const std::string& filename);

  std::vector<o2::itsmft::ROFRecord> mClusROFRec, *mClusROFRecPtr = &mClusROFRec;
  std::vector<o2::itsmft::Cluster> mClusterArray, *mClusterArrayPtr = &mClusterArray;
  std::vector<o2::itsmft::CompClusterExt> mClusterCompArray, *mClusterCompArrayPtr = &mClusterCompArray;
  std::vector<unsigned char> mPatternsArray, *mPatternsArrayPtr = &mPatternsArray;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> mClusterMCTruth, *mClusterMCTruthPtr = &mClusterMCTruth;
  std::vector<o2::itsmft::MC2ROFRecord> mClusMC2ROFs, *mClusMC2ROFsPtr = &mClusMC2ROFs;

  o2::header::DataOrigin mOrigin = o2::header::gDataOriginInvalid;

  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTree;

  bool mUseMC = true;     // use MC truth
  bool mUseClFull = true; // use full clusters
  bool mUseClComp = true; // use compact clusters
  bool mUsePatterns = true; // send patterns

  std::string mDetName = "";
  std::string mDetNameLC = "";
  std::string mFileName = "";
  std::string mClusTreeName = "o2sim";
  std::string mClusROFBranchName = "ClustersROF";
  std::string mClusterBranchName = "Cluster";
  std::string mClusterPattBranchName = "ClusterPatt";
  std::string mClusterCompBranchName = "ClusterComp";
  std::string mClustMCTruthBranchName = "ClusterMCTruth";
  std::string mClustMC2ROFBranchName = "ClustersMC2ROF";
};

class EC0ClusterReader : public ClusterReader
{
 public:
  EC0ClusterReader(bool useMC = true, bool useClFull = true, bool useClComp = true)
    : ClusterReader(o2::detectors::DetID::EC0, useMC, useClFull, useClComp)
  {
    mOrigin = o2::header::gDataOriginITS;
  }
};

/// create a processor spec
/// read ITS/MFT cluster data from a root file
framework::DataProcessorSpec getEC0ClusterReaderSpec(bool useMC = true, bool useClFull = true, bool useClComp = true, bool usePatterns = true);

} // namespace endcaps
} // namespace o2

#endif /* O2_ENDCAPS_CLUSTERREADER */
