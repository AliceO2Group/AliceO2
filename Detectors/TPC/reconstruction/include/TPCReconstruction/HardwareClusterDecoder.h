// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TPCCATracking.h
/// \brief Decoder to convert TPC ClusterHardware to ClusterNative
/// \author David Rohr
#ifndef ALICEO2_TPC_HARDWARECLUSTERDECODER_H_
#define ALICEO2_TPC_HARDWARECLUSTERDECODER_H_

#include <vector>
#include "TPCReconstruction/DigitalCurrentClusterIntegrator.h"

namespace o2 { namespace DataFormat { namespace TPC {
class ClusterNative;
class ClusterHardwareContainer;
class ClusterNativeContainer;
}}}

namespace o2 { namespace dataformats { template <typename TruthElement> class MCTruthContainer; } class MCCompLabel; }

namespace o2 { namespace TPC {

//Class to convert a list of input buffers containing TPC clusters of type ClusterHardware to type ClusterNative.
class HardwareClusterDecoder
{
public:
  HardwareClusterDecoder() = default;
  ~HardwareClusterDecoder() = default;
  
  int decodeClusters(std::vector<std::pair<const o2::DataFormat::TPC::ClusterHardwareContainer*, std::size_t>>& inputClusters, std::vector<o2::DataFormat::TPC::ClusterNativeContainer>& outputClusters,
    const std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>* inMCLabels = nullptr, std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>* outMCLabels = nullptr);
  static void sortClustersAndMC(std::vector<o2::DataFormat::TPC::ClusterNative> clusters, o2::dataformats::MCTruthContainer<o2::MCCompLabel> mcTruth);

private:
  DigitalCurrentClusterIntegrator mIntegrator;
};

}}
#endif
