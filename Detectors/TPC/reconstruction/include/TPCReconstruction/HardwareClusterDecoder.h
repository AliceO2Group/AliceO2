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

namespace o2 { namespace DataFormat { namespace TPC {
class ClusterHardwareContainer;
class ClusterNativeContainer;
}}}

namespace o2 { namespace TPC {

class HardwareClusterDecoder
{
public:
  HardwareClusterDecoder() = default;
  ~HardwareClusterDecoder() = default;
  
  int decodeClusters(std::vector<std::pair<const o2::DataFormat::TPC::ClusterHardwareContainer*, std::size_t>>& inputClusters, std::vector<o2::DataFormat::TPC::ClusterNativeContainer>& outputClusters);

private:
};

}}
#endif
