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

#ifndef O2_TOF_CLUSTERREADER
#define O2_TOF_CLUSTERREADER

#include "TFile.h"
#include "TTree.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DataFormatsTOF/Cluster.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::framework;

namespace o2
{
namespace tof
{

class ClusterReader : public Task
{
 public:
  ClusterReader(bool useMC) : mUseMC(useMC) {}
  ~ClusterReader() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  void connectTree(const std::string& filename);

  bool mUseMC = true;
  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTree;
  std::string mFileName = "";

  std::vector<Cluster> mClusters, *mClustersPtr = &mClusters;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> mLabels, *mLabelsPtr = &mLabels;
};

/// create a processor spec
/// read simulated TOF digits from a root file
framework::DataProcessorSpec getClusterReaderSpec(bool useMC);

} // namespace tof
} // namespace o2

#endif /* O2_TOF_CLUSTERREADER */
