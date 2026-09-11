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
#include "TRKBase/Specs.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "DataFormatsTRKFT3/Cluster.h"
#include "DataFormatsTRKFT3/ROFRecord.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "Headers/DataHeader.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::framework;

namespace o2::trk
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
using PatternsType = std::vector<unsigned char>;
using ROFrameType = std::vector<o2::trkft3::ROFRecord>;
using LabelsType = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;

template <int DetID>
DataProcessorSpec getClusterWriterSpecT(bool useMC)
{
  static_assert(DetID == o2::detectors::DetID::TRK || DetID == o2::detectors::DetID::FT3, "only TRK and FT3 cluster writers are supported");
  using ClustersType = std::vector<o2::trkft3::Cluster<DetID>>;
  static constexpr o2::header::DataOrigin Origin = DetID == o2::detectors::DetID::TRK ? o2::header::gDataOriginTRK : o2::header::gDataOriginFT3;
  const int nLayers = DetID == o2::detectors::DetID::TRK ? o2::trk::AlmiraParam::kNLayers : o2::trk::constants::MLOTDisks::nLayers;
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
    vecInpSpecClus.emplace_back(getName(detName + "compclus", iLayer), Origin, "COMPCLUSTERS", iLayer);
    vecInpSpecPatt.emplace_back(getName(detName + "patterns", iLayer), Origin, "PATTERNS", iLayer);
    vecInpSpecROF.emplace_back(getName(detName + "ROframes", iLayer), Origin, "CLUSTERSROF", iLayer);
    vecInpSpecLbl.emplace_back(getName(detName + "labels", iLayer), Origin, "CLUSTERSMCTR", iLayer);
  }

  return MakeRootTreeWriterSpec(std::format("{}-cluster-writer", detNameLC).c_str(),
                                std::format("o2clus_{}.root", detNameLC).c_str(),
                                MakeRootTreeWriterSpec::TreeAttributes{.name = "o2sim", .title = "Tree with " + detName + " clusters"},
                                BranchDefinition<ClustersType>{vecInpSpecClus,
                                                               detName + "ClusterComp", "compact-cluster-branch",
                                                               nLayers,
                                                               compClustersSizeGetter,
                                                               getIndex,
                                                               getName},
                                BranchDefinition<PatternsType>{vecInpSpecPatt,
                                                               detName + "ClusterPatt", "cluster-pattern-branch",
                                                               nLayers,
                                                               getIndex,
                                                               getName},
                                BranchDefinition<ROFrameType>{vecInpSpecROF,
                                                              detName + "ClustersROF", "cluster-rof-branch",
                                                              nLayers,
                                                              logger,
                                                              getIndex,
                                                              getName},
                                BranchDefinition<LabelsType>{vecInpSpecLbl,
                                                             detName + "ClusterMCTruth", "cluster-label-branch",
                                                             (useMC ? nLayers : 0),
                                                             getIndex,
                                                             getName})();
}

DataProcessorSpec getTRKClusterWriterSpec(bool useMC)
{
  return getClusterWriterSpecT<o2::detectors::DetID::TRK>(useMC);
}

DataProcessorSpec getFT3ClusterWriterSpec(bool useMC)
{
  return getClusterWriterSpecT<o2::detectors::DetID::FT3>(useMC);
}

DataProcessorSpec getClusterWriterSpec(bool useMC)
{
  return getTRKClusterWriterSpec(useMC);
}

} // namespace o2::trk
