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

#ifndef O2_ITSMFT_CLUSTERREADER
#define O2_ITSMFT_CLUSTERREADER

#include "TFile.h"

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
namespace itsmft
{

class ClusterReader : public Task
{
 public:
  ClusterReader() = delete;
  ClusterReader(o2::detectors::DetID id, bool useMC = true, bool useClFull = true, bool useClComp = true);
  ~ClusterReader() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 protected:
  void accumulate();

  std::vector<o2::itsmft::ROFRecord>*mClusROFRecInp = nullptr, mClusROFRecOut;
  std::vector<o2::itsmft::Cluster>*mClusterArrayInp = nullptr, mClusterArrayOut;
  std::vector<o2::itsmft::CompClusterExt>*mClusterCompArrayInp = nullptr, mClusterCompArrayOut;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>*mClusterMCTruthInp = nullptr, mClusterMCTruthOut;

  o2::header::DataOrigin mOrigin = o2::header::gDataOriginInvalid;

  bool mFinished = false;

  bool mUseMC = true;     // use MC truth
  bool mUseClFull = true; // use full clusters
  bool mUseClComp = true; // use compact clusters

  std::string mDetName = "";
  std::string mDetNameLC = "";
  std::string mInputFileName = "";

  std::string mClusTreeName = "o2sim";
  std::string mClusROFTreeName = "ClustersROF";
  std::string mClusterBranchName = "Cluster";
  std::string mClusterCompBranchName = "ClusterComp";
  std::string mClustMCTruthBranchName = "ClusterMCTruth";
};

class ITSClusterReader : public ClusterReader
{
 public:
  ITSClusterReader(bool useMC = true, bool useClFull = true, bool useClComp = true)
    : ClusterReader(o2::detectors::DetID::ITS, useMC, useClFull, useClComp)
  {
    mOrigin = o2::header::gDataOriginITS;
  }
};

class MFTClusterReader : public ClusterReader
{
 public:
  MFTClusterReader(bool useMC = true, bool useClFull = true, bool useClComp = true)
    : ClusterReader(o2::detectors::DetID::MFT, useMC, useClFull, useClComp)
  {
    mOrigin = o2::header::gDataOriginMFT;
  }
};

/// create a processor spec
/// read ITS/MFT cluster data from a root file
framework::DataProcessorSpec getITSClusterReaderSpec(bool useMC = true, bool useClFull = true, bool useClComp = true);
framework::DataProcessorSpec getMFTClusterReaderSpec(bool useMC = true, bool useClFull = true, bool useClComp = true);

} // namespace itsmft
} // namespace o2

#endif /* O2_ITSMFT_CLUSTERREADER */
