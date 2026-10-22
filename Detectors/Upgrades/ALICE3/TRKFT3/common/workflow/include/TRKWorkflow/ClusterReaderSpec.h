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
/// @brief  Reads the TRK or FT3 clusters the cluster writer stored, one stream per layer

#ifndef O2_TRK_CLUSTERREADER
#define O2_TRK_CLUSTERREADER

#include <memory>
#include <string>
#include <vector>

#include "TFile.h"
#include "TTree.h"
#include "DataFormatsTRKFT3/Cluster.h"
#include "DataFormatsTRKFT3/ROFRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "TRKBase/AlmiraParam.h"

namespace o2::trk
{

template <int DetID>
class ClusterReader : public framework::Task
{
 public:
  explicit ClusterReader(bool useMC);
  void init(framework::InitContext& ic) final;
  void run(framework::ProcessingContext& pc) final;

  static constexpr int NLayers = DetID == o2::detectors::DetID::TRK ? static_cast<int>(o2::trk::AlmiraParam::kNLayers) : o2::trk::constants::MLOTDisks::nLayers;

 private:
  void connectTree(const std::string& filename);
  template <typename Ptr>
  void setBranchAddress(const std::string& base, int layer, Ptr& address);

  std::vector<std::vector<o2::trkft3::Cluster<DetID>>*> mClusters;
  std::vector<std::vector<unsigned char>*> mPatterns;
  std::vector<std::vector<o2::trkft3::ROFRecord>*> mROFs;
  std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>*> mLabels;

  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTree;
  bool mUseMC = true;
};

using TRKClusterReader = ClusterReader<o2::detectors::DetID::TRK>;
using FT3ClusterReader = ClusterReader<o2::detectors::DetID::FT3>;

/// The reader of \a defname, by default the cluster writer's file.
framework::DataProcessorSpec getTRKClusterReaderSpec(bool useMC = true, std::string defname = "o2clus_trk.root");
framework::DataProcessorSpec getFT3ClusterReaderSpec(bool useMC = true, std::string defname = "o2clus_ft3.root");

} // namespace o2::trk

#endif
