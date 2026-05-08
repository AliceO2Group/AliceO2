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

/// @file   ClusterReaderSpec.h

#ifndef O2_ITSMFT_CLUSTERREADER
#define O2_ITSMFT_CLUSTERREADER

#include <vector>

#include <TFile.h>
#include <TTree.h>

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Headers/DataHeader.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsITSMFT/DPLAlpideParam.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DetectorsCommonDataFormats/DetID.h"

using namespace o2::framework;

namespace o2::itsmft
{

template <int N>
class ClusterReader : public Task
{
 public:
  static constexpr o2::detectors::DetID ID{N == o2::detectors::DetID::ITS ? o2::detectors::DetID::ITS : o2::detectors::DetID::MFT};
  static constexpr o2::header::DataOrigin Origin{(N == o2::detectors::DetID::ITS) ? o2::header::gDataOriginITS : o2::header::gDataOriginMFT};

  ClusterReader() = delete;
  ClusterReader(bool useMC = true, bool doStag = false, bool usePatterns = true, bool triggers = true);
  ~ClusterReader() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 protected:
  void connectTree(const std::string& filename);
  template <typename Ptr>
  void setBranchAddress(const std::string& base, Ptr& addr, int layer);
  std::string getBranchName(const std::string& base, int index) const;

  std::vector<std::vector<o2::itsmft::ROFRecord>*> mClusROFRec{nullptr};
  std::vector<std::vector<o2::itsmft::CompClusterExt>*> mClusterCompArray{nullptr};
  std::vector<std::vector<unsigned char>*> mPatternsArray{nullptr};
  std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>*> mClusterMCTruth{nullptr};

  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTree;

  int mLayers = 1;
  bool mUseMC = true;         // use MC truth
  bool mDoStaggering = false; // read staggered data
  bool mUsePatterns = true;   // send patterns
  bool mTriggerOut = true;    // send dummy triggers vector

  std::string mDetName;
  std::string mDetNameLC;
  std::string mFileName;
  std::string mClusTreeName = "o2sim";
  std::string mClusROFBranchName = "ClustersROF";
  std::string mClusterPattBranchName = "ClusterPatt";
  std::string mClusterCompBranchName = "ClusterComp";
  std::string mClustMCTruthBranchName = "ClusterMCTruth";
};

class ITSClusterReader : public ClusterReader<o2::detectors::DetID::ITS>
{
 public:
  ITSClusterReader(bool useMC = true, bool doStag = false, bool usePatterns = true, bool triggerOut = true)
    : ClusterReader(useMC, doStag, usePatterns, triggerOut) {}
};

class MFTClusterReader : public ClusterReader<o2::detectors::DetID::MFT>
{
 public:
  MFTClusterReader(bool useMC = true, bool doStag = false, bool usePatterns = true, bool triggerOut = true)
    : ClusterReader(useMC, doStag, usePatterns, triggerOut) {}
};

/// create a processor spec
/// read ITS/MFT cluster data from a root file
framework::DataProcessorSpec getITSClusterReaderSpec(bool useMC = true, bool doStag = false, bool usePatterns = true, bool useTriggers = true);
framework::DataProcessorSpec getMFTClusterReaderSpec(bool useMC = true, bool doStag = false, bool usePatterns = true, bool useTriggers = true);

} // namespace o2::itsmft

#endif /* O2_ITSMFT_CLUSTERREADER */
