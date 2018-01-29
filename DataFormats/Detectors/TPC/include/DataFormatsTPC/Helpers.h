// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Helpers.h
/// \brief Helper class for memory management of TPC Data Formats, external from the actual data type classes to keep them simple in order to run on all kinds of accelerators
/// \author David Rohr
#ifndef ALICEO2_DATAFORMATSTPC_HELPERS_H
#define ALICEO2_DATAFORMATSTPC_HELPERS_H

#include "DataFormatsTPC/ClusterHardware.h"
#include "DataFormatsTPC/ClusterNative.h"
#include <memory>

namespace o2 { class MCCompLabel; namespace dataformats { template <class T> class MCTruthContainer; }}

namespace o2 { namespace DataFormat { namespace TPC{

class TPCClusterFormatHelper
{
public:
  //Helper function to create a ClusterNativeAccessFullTPC structure from a std::vector of ClusterNative containers
  //This is not contained in the ClusterNative class itself to reduce the dependencies of the class
  static std::unique_ptr<ClusterNativeAccessFullTPC> accessNativeContainerArray(std::vector<ClusterNativeContainer>& clusters, std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>* mcTruth = nullptr);
    
};

template <unsigned int size>
class ClusterHardwareContainerFixedSize {
  //We cannot use a union because that prevents ROOT streaming, so we just block 8 kb and reinterpret_cast to ClusterHardwareContainer
public:
  ClusterHardwareContainer* getContainer() {return(reinterpret_cast<ClusterHardwareContainer*>(this));}
  int getMaxNumberOfClusters() {return((sizeof(*this) - sizeof(ClusterHardwareContainer)) / sizeof(ClusterHardware));}
  ClusterHardwareContainerFixedSize() {memset(this, 0, size);}
    
private:
  uint8_t mFixSize[size];

  static_assert(size <= 8192, "Size must be below 8 kb");
  static_assert(size >= sizeof(ClusterHardwareContainer), "Size must be at least sizeof(ClusterHardwareContainer)");
};
typedef ClusterHardwareContainerFixedSize<8192> ClusterHardwareContainer8kb;

}}}

#endif
