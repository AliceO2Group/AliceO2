// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CompressedClustersHelpers.h
/// \brief Helper for the CompressedClusters container
/// \author Matthias Richter

#ifndef ALICEO2_DATAFORMATSTPC_COMPRESSEDCLUSTERSHELPERS_H
#define ALICEO2_DATAFORMATSTPC_COMPRESSEDCLUSTERSHELPERS_H

#include <type_traits>
#include <cassert>
#include <stdexcept>
#include <vector>
#include "Algorithm/FlattenRestore.h"
#include "DataFormatsTPC/CompressedClusters.h"

namespace flatten = o2::algorithm::flatten;

namespace o2::tpc
{
struct CompressedClustersHelpers {
  /// Apply an operation to the layout of the class
  /// This method is the single place defining the mapping of the class layout
  /// to an underlying buffer for the flatten/restore functionality
  /// @param op       the operation to be applied, e.g. calculate raw buffer size,
  ///                 copy to and restore from buffer
  /// @param ptr      buffer pointer, passed onto to operation
  /// @param counters the counters object CompressedClustersCounters
  template <typename Op, typename BufferType>
  static size_t apply(Op op, BufferType& ptr, CompressedClustersROOT& c)
  {
    size_t size = 0;
    size += op(ptr, c.nAttachedClusters, c.qTotA, c.qMaxA, c.flagsA, c.sigmaPadA, c.sigmaTimeA);
    size += op(ptr, c.nUnattachedClusters, c.qTotU, c.qMaxU, c.flagsU, c.padDiffU, c.timeDiffU, c.sigmaPadU, c.sigmaTimeU);
    size += op(ptr, c.nAttachedClustersReduced, c.rowDiffA, c.sliceLegDiffA, c.padResA, c.timeResA);
    size += op(ptr, c.nTracks, c.qPtA, c.rowA, c.sliceA, c.timeA, c.padA, c.nTrackClusters);
    size += op(ptr, c.nSliceRows, c.nSliceRowClusters);
    return size;
  }

  /// Create a flat copy of the class
  /// The target container is resized accordingly
  template <typename ContainerType>
  static size_t flattenTo(ContainerType& container, CompressedClustersROOT& clusters)
  {
    static_assert(sizeof(typename ContainerType::value_type) == 1);
    char* dummyptr = nullptr;
    auto calc_size = [](auto&... args) { return flatten::calc_size(args...); };
    auto copy_to = [](auto&... args) { return flatten::copy_to(args...); };
    size_t size = apply(calc_size, dummyptr, clusters);
    container.resize(size);
    auto wrtptr = container.data();
    size_t copySize = apply(copy_to, wrtptr, clusters);
    assert(copySize == size);
    return copySize;
  }

  /// Restore the array pointers from the data in the container
  template <typename ContainerType>
  static size_t restoreFrom(ContainerType& container, CompressedClustersROOT& clusters)
  {
    static_assert(sizeof(typename ContainerType::value_type) == 1);
    char* dummyptr = nullptr;
    auto calc_size = [](auto&... args) { return flatten::calc_size(args...); };
    auto set_from = [](auto&... args) { return flatten::set_from(args...); };
    size_t size = apply(calc_size, dummyptr, clusters);
    if (container.size() != size) {
      // for the moment we don not support changes in the member layout, we can implement the
      // handling in the custom streamer
      throw std::runtime_error("mismatch between raw buffer size and counters, schema evolution is not yet supported");
    }
    auto readptr = container.data();
    size_t checkSize = apply(set_from, readptr, clusters);
    assert(checkSize == size);
    return checkSize;
  }
};

} // namespace o2::tpc

#endif // ALICEO2_DATAFORMATSTPC_COMPRESSEDCLUSTERSHELPERS_H
