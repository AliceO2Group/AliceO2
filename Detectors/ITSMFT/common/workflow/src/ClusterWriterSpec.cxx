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

#include "Framework/ConcreteDataMatcher.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "ITSMFTWorkflow/ClusterWriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::framework;

namespace o2::itsmft
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
using CompClusType = std::vector<o2::itsmft::CompClusterExt>;
using PatternsType = std::vector<unsigned char>;
using ROFrameRType = std::vector<o2::itsmft::ROFRecord>;
using LabelsType = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
using ROFRecLblT = std::vector<o2::itsmft::MC2ROFRecord>;
using namespace o2::header;

template <int N>
DataProcessorSpec getClusterWriterSpec(bool useMC)
{
  static constexpr o2::header::DataOrigin Origin{N == o2::detectors::DetID::ITS ? o2::header::gDataOriginITS : o2::header::gDataOriginMFT};
  constexpr int NLayers = (DPLAlpideParam<N>::supportsStaggering()) ? DPLAlpideParam<N>::getNLayers() : 1;
  const auto detName = Origin.as<std::string>();
  // Spectators for logging
  auto compClusterSizes = std::make_shared<std::array<size_t, NLayers>>();
  auto compClustersSizeGetter = [compClusterSizes](CompClusType const& compClusters, DataRef const& ref) {
    auto const* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    (*compClusterSizes)[dh->subSpecification] = compClusters.size();
  };
  auto logger = [detName, compClusterSizes](std::vector<o2::itsmft::ROFRecord> const& rofs, DataRef const& ref) {
    auto const* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    const auto i = dh->subSpecification;
    LOG(info) << detName << "ClusterWriter:" << i << " pulled " << (*compClusterSizes)[i] << " clusters, in " << rofs.size() << " RO frames";
  };
  auto getIndex = [](DataRef const& ref) -> size_t {
    auto const* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    return static_cast<size_t>(dh->subSpecification);
  };
  auto getName = [](std::string base, size_t index) -> std::string {
    if constexpr (DPLAlpideParam<N>::supportsStaggering()) {
      return base += "_" + std::to_string(index);
    }
    return base;
  };
  auto detNameLC = detName;
  std::transform(detNameLC.begin(), detNameLC.end(), detNameLC.begin(), [](unsigned char c) { return std::tolower(c); });
  return MakeRootTreeWriterSpec(std::format("{}-cluster-writer", detNameLC).c_str(),
                                (o2::detectors::DetID::ITS == N) ? "o2clus_its.root" : "mftclusters.root",
                                MakeRootTreeWriterSpec::TreeAttributes{.name = "o2sim", .title = std::format("Tree with {} clusters", detName)},
                                BranchDefinition<CompClusType>{InputSpec{"compclus", ConcreteDataTypeMatcher{Origin, "COMPCLUSTERS"}},
                                                               (detName + "ClusterComp").c_str(), "compact-cluster-branch",
                                                               NLayers,
                                                               compClustersSizeGetter,
                                                               getIndex,
                                                               getName},
                                BranchDefinition<PatternsType>{InputSpec{"patterns", ConcreteDataTypeMatcher{Origin, "PATTERNS"}},
                                                               (detName + "ClusterPatt").c_str(), "cluster-pattern-branch",
                                                               NLayers,
                                                               getIndex,
                                                               getName},
                                BranchDefinition<ROFrameRType>{InputSpec{"ROframes", ConcreteDataTypeMatcher{Origin, "CLUSTERSROF"}},
                                                               (detName + "ClustersROF").c_str(), "cluster-rof-branch",
                                                               NLayers,
                                                               logger,
                                                               getIndex,
                                                               getName},
                                BranchDefinition<LabelsType>{InputSpec{"labels", ConcreteDataTypeMatcher{Origin, "CLUSTERSMCTR"}},
                                                             (detName + "ClusterMCTruth").c_str(), "cluster-label-branch",
                                                             (useMC ? NLayers : 0),
                                                             getIndex,
                                                             getName},
                                BranchDefinition<ROFRecLblT>{InputSpec{"MC2ROframes", ConcreteDataTypeMatcher{Origin, "CLUSTERSMC2ROF"}},
                                                             (detName + "ClustersMC2ROF").c_str(), "cluster-mc2rof-branch",
                                                             (useMC ? NLayers : 0),
                                                             getIndex,
                                                             getName})();
}

framework::DataProcessorSpec getITSClusterWriterSpec(bool useMC) { return getClusterWriterSpec<o2::detectors::DetID::ITS>(useMC); }
framework::DataProcessorSpec getMFTClusterWriterSpec(bool useMC) { return getClusterWriterSpec<o2::detectors::DetID::MFT>(useMC); }

} // namespace o2::itsmft
