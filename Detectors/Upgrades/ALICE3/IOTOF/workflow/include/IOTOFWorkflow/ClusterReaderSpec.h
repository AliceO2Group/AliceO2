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

/// \file ClusterReaderSpec.h
/// \brief The TF3 clusters of o2-alice3-iotof-reco-workflow's cluster file
///        (ClusterWriterSpec.h), one timeframe per entry.

#ifndef O2_IOTOF_CLUSTERREADERSPEC_H
#define O2_IOTOF_CLUSTERREADERSPEC_H

#include "DataFormatsIOTOF/Cluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#include <TFile.h>
#include <TTree.h>

#include <memory>
#include <string>
#include <vector>

namespace o2::iotof
{

/// Publishes TF3/CLUSTERS, TF3/PATTERNS, TF3/CLUSTERSROF and, with MC,
/// TF3/CLUSTERSMCTR (subspec 0) from the tree `o2sim` of `--tf3-cluster-infile`,
/// as the clusterer publishes them; an empty timeframe past the tree's last
/// entry.
class ClusterReader : public o2::framework::Task
{
 public:
  explicit ClusterReader(bool useMC) : mUseMC(useMC) {}
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;

 private:
  void connectTree(const std::string& filename);

  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTree;
  std::vector<o2::iotof::Cluster>* mClusters = nullptr;
  std::vector<unsigned char>* mPatterns = nullptr;
  std::vector<o2::itsmft::ROFRecord>* mROFs = nullptr;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mLabels = nullptr;
  bool mUseMC = false;
};

/// The reader's device, `tf3-cluster-reader`; its file `--tf3-cluster-infile`
/// (tf3clusters.root by default).
o2::framework::DataProcessorSpec getIOTOFClusterReaderSpec(bool useMC);

} // namespace o2::iotof

#endif // O2_IOTOF_CLUSTERREADERSPEC_H
