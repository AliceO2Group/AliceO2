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

/// @file   ClusterWriterSpec.cxx

#include <algorithm>
#include <cctype>
#include <memory>
#include <vector>
#include <format>

#include "TRKWorkflow/ClusterWriterSpec.h"
#include "Framework/ConcreteDataMatcher.h"
#include "Framework/DataRef.h"
#include "TRKBase/AlmiraParam.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "DataFormatsTRK/Cluster.h"
#include "DataFormatsTRK/ROFRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::framework;

namespace o2::trk
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
using ClustersType = std::vector<o2::trk::Cluster>;
using PatternsType = std::vector<unsigned char>;
using ROFrameType = std::vector<o2::trk::ROFRecord>;
using LabelsType = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
using ROFRecLblType = std::vector<o2::trk::MC2ROFRecord>;

DataProcessorSpec getClusterWriterSpec(bool useMC)
{
  static constexpr o2::header::DataOrigin Origin{o2::header::gDataOriginTRK};
  static constexpr int nLayers = o2::trk::AlmiraParam::kNLayers;
  const auto detName = Origin.as<std::string>();

  auto compClusterSizes = std::make_shared<std::vector<size_t>>(nLayers, 0);
  auto compClustersSizeGetter = [compClusterSizes](ClustersType const& compClusters, DataRef const& ref) {
    auto const* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    (*compClusterSizes)[dh->subSpecification] = compClusters.size();
  };
  auto logger = [detName, compClusterSizes](ROFrameType const& rofs, DataRef const& ref) {
    auto const* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    const auto i = dh->subSpecification;
    LOG(info) << detName << "ClusterWriter on layer " << i
              << " pulled " << (*compClusterSizes)[i] << " clusters, in " << rofs.size() << " RO frames";
  };
  auto getIndex = [](DataRef const& ref) -> size_t {
    auto const* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    return static_cast<size_t>(dh->subSpecification);
  };
  auto getName = [](std::string base, size_t index) -> std::string {
    return base + "_" + std::to_string(index);
  };
  auto detNameLC = detName;
  std::transform(detNameLC.begin(), detNameLC.end(), detNameLC.begin(), [](unsigned char c) { return std::tolower(c); });

  std::vector<InputSpec> vecInpSpecClus, vecInpSpecPatt, vecInpSpecROF, vecInpSpecLbl;
  vecInpSpecClus.reserve(nLayers);
  vecInpSpecPatt.reserve(nLayers);
  vecInpSpecROF.reserve(nLayers);
  vecInpSpecLbl.reserve(nLayers);
  for (int iLayer = 0; iLayer < nLayers; iLayer++) {
    vecInpSpecClus.emplace_back(getName("compclus", iLayer), Origin, "COMPCLUSTERS", iLayer);
    vecInpSpecPatt.emplace_back(getName("patterns", iLayer), Origin, "PATTERNS", iLayer);
    vecInpSpecROF.emplace_back(getName("ROframes", iLayer), Origin, "CLUSTERSROF", iLayer);
    vecInpSpecLbl.emplace_back(getName("labels", iLayer), Origin, "CLUSTERSMCTR", iLayer);
  }

  return MakeRootTreeWriterSpec(std::format("{}-cluster-writer", detNameLC).c_str(),
                                "o2clus_trk.root",
                                MakeRootTreeWriterSpec::TreeAttributes{.name = "o2sim", .title = "Tree with TRK clusters"},
                                BranchDefinition<ClustersType>{vecInpSpecClus,
                                                               "TRKClusterComp", "compact-cluster-branch",
                                                               nLayers,
                                                               compClustersSizeGetter,
                                                               getIndex,
                                                               getName},
                                BranchDefinition<PatternsType>{vecInpSpecPatt,
                                                               "TRKClusterPatt", "cluster-pattern-branch",
                                                               nLayers,
                                                               getIndex,
                                                               getName},
                                BranchDefinition<ROFrameType>{vecInpSpecROF,
                                                              "TRKClustersROF", "cluster-rof-branch",
                                                              nLayers,
                                                              logger,
                                                              getIndex,
                                                              getName},
                                BranchDefinition<LabelsType>{vecInpSpecLbl,
                                                             "TRKClusterMCTruth", "cluster-label-branch",
                                                             (useMC ? nLayers : 0),
                                                             getIndex,
                                                             getName})();
}

} // namespace o2::trk
