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

#include "IOTOFWorkflow/ClusterWriterSpec.h"
#include "Framework/ConcreteDataMatcher.h"
#include "Framework/DataRef.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "DataFormatsIOTOF/Cluster.h"
#include "DataFormatsIOTOF/Digit.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/IOMCTruthContainerView.h"

using namespace o2::framework;

namespace o2::iotof
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
using ClustersType = std::vector<o2::iotof::Cluster>;
using PatternsType = std::vector<unsigned char>;
using ROFrameType = std::vector<o2::itsmft::ROFRecord>;
using LabelsType = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;

DataProcessorSpec getClusterWriterSpec(bool mctruth, bool dec, o2::header::DataOrigin detOrig, o2::detectors::DetID detId)
{
  std::string detStr = o2::detectors::DetID::getName(detId);
  std::string detStrL = dec ? "o2_" : ""; // for decoded digits prepend by o2
  detStrL += detStr;
  std::transform(detStrL.begin(), detStrL.end(), detStrL.begin(), ::tolower);
  auto logger = [](std::vector<o2::iotof::Cluster> const& inClusters) {
    LOG(info) << "RECEIVED CLUSTERS SIZE " << inClusters.size();
  };

  return MakeRootTreeWriterSpec((detStr + "ClusterWriter" + (dec ? "_dec" : "")).c_str(),
                                (detStrL + "clusters.root").c_str(),
                                MakeRootTreeWriterSpec::TreeAttributes{.name = "o2sim", .title = "Tree with TF3 clusters"},
                                BranchDefinition<ClustersType>{InputSpec{"tf3_compclus", detOrig, "COMPCLUSTERS", 0},
                                                               (detStr + "ClusterComp").c_str(),
                                                               logger},
                                BranchDefinition<PatternsType>{InputSpec{"tf3_patterns", detOrig, "PATTERNS", 0},
                                                               (detStr + "ClusterPatt").c_str()},
                                BranchDefinition<ROFrameType>{InputSpec{"tf3_ROframes", detOrig, "CLUSTERSROF", 0},
                                                              (detStr + "ClusterROF").c_str(), "cluster-rof-branch"},
                                BranchDefinition<LabelsType>{InputSpec{"tf3_labels", detOrig, "CLUSTERSMCTR", 0},
                                                             (detStr + "ClusterMCTruth").c_str()})();
}

DataProcessorSpec getIOTOFClusterWriterSpec(bool mctruth, bool dec)
{
  return getClusterWriterSpec(mctruth, dec, o2::header::gDataOriginTF3, o2::detectors::DetID::TF3);
}

} // namespace o2::iotof
