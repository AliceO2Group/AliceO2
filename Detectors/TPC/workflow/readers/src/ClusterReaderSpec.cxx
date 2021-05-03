// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   ClusterReaderSpec1.cxx
/// @author David Rohr

#include "Framework/WorkflowSpec.h"
#include "DPLUtils/RootTreeReader.h"
#include "TPCReaderWorkflow/PublisherSpec.h"
#include "SimulationDataFormat/IOMCTruthContainerView.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "TPCBase/Sector.h"

using namespace o2::framework;

namespace o2
{
namespace tpc
{

framework::DataProcessorSpec getClusterReaderSpec(bool useMC, const std::vector<int>* tpcSectors, const std::vector<int>* laneConfiguration)
{
  static RootTreeReader::SpecialPublishHook hook{[](std::string_view name, ProcessingContext& context, o2::framework::Output const& output, char* data) -> bool {
    if (TString(name.data()).Contains("TPCDigitMCTruth") || TString(name.data()).Contains("TPCClusterHwMCTruth") || TString(name.data()).Contains("TPCClusterNativeMCTruth")) {
      auto storedlabels = reinterpret_cast<o2::dataformats::IOMCTruthContainerView const*>(data);
      o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel> flatlabels;
      storedlabels->copyandflatten(flatlabels);
      //LOG(INFO) << "PUBLISHING CONST LABELS " << flatlabels.getNElements();
      context.outputs().snapshot(output, flatlabels);
      return true;
    }
    return false;
  }};

  std::vector<int> defaultSectors;
  std::vector<int> defaultLaneConfig;
  if (tpcSectors == nullptr) {
    defaultSectors.resize(Sector::MAXSECTOR);
    std::iota(defaultSectors.begin(), defaultSectors.end(), 0);
    tpcSectors = &defaultSectors;
  }
  if (laneConfiguration == nullptr) {
    defaultLaneConfig = *tpcSectors;
    laneConfiguration = &defaultLaneConfig;
  }

  return std::move(o2::tpc::getPublisherSpec(PublisherConf{
                                               "tpc-native-cluster-reader",
                                               "tpc-native-clusters.root",
                                               "tpcrec",
                                               {"clusterbranch", "TPCClusterNative", "Branch with TPC native clusters"},
                                               {"clustermcbranch", "TPCClusterNativeMCTruth", "MC label branch"},
                                               OutputSpec{"TPC", "CLUSTERNATIVE"},
                                               OutputSpec{"TPC", "CLNATIVEMCLBL"},
                                               *tpcSectors,
                                               *laneConfiguration,
                                               &hook},
                                             useMC));
}

} // end namespace tpc
} // end namespace o2
