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
/// \brief Helper class for memory management of TPC Data Formats, external from the actual data type classes to keep
/// them simple in order to run on all kinds of accelerators
/// \author David Rohr
#ifndef ALICEO2_DATAFORMATSTPC_HELPERS_H
#define ALICEO2_DATAFORMATSTPC_HELPERS_H

#include <memory>
#include <cstring> // for memset
#include "DataFormatsTPC/ClusterHardware.h"

namespace o2
{
class MCCompLabel;
namespace dataformats
{
template <class T>
class MCTruthContainer;
}
} // namespace o2

namespace o2
{
namespace tpc
{
template <unsigned int size>
class ClusterHardwareContainerFixedSize
{
  // We cannot use a union because that prevents ROOT streaming, so we just block 8 kb and reinterpret_cast to
  // ClusterHardwareContainer
 public:
  ClusterHardwareContainer* getContainer() { return (reinterpret_cast<ClusterHardwareContainer*>(mFixSize)); }
  ClusterHardwareContainer const* getContainer() const { return (reinterpret_cast<ClusterHardwareContainer const*>(mFixSize)); }
  constexpr int getMaxNumberOfClusters()
  {
    static_assert(sizeof(*this) == sizeof(mFixSize));
    return ((sizeof(mFixSize) - sizeof(ClusterHardwareContainer)) / sizeof(ClusterHardware));
  }
  ClusterHardwareContainerFixedSize() { memset(this, 0, size); }

 private:
  uint8_t mFixSize[size];

  static_assert(size <= 8192, "Size must be below 8 kb");
  static_assert(size >= sizeof(ClusterHardwareContainer), "Size must be at least sizeof(ClusterHardwareContainer)");
};
typedef ClusterHardwareContainerFixedSize<8192> ClusterHardwareContainer8kb;
} // namespace tpc
} // namespace o2

#endif
