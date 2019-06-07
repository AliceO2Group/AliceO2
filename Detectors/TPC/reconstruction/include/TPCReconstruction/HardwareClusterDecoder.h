// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HardwareClusterDecoder.h
/// \brief Decoder to convert TPC ClusterHardware to ClusterNative
/// \author David Rohr
#ifndef ALICEO2_TPC_HARDWARECLUSTERDECODER_H_
#define ALICEO2_TPC_HARDWARECLUSTERDECODER_H_

#include <vector>
#include <functional>
#include "TPCReconstruction/DigitalCurrentClusterIntegrator.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/ClusterNativeHelper.h"

namespace o2
{
namespace tpc
{
class ClusterHardwareContainer;
class MCCompLabel;
namespace dataformats
{
template <typename TruthElement>
class MCTruthContainer;
}

/// @class HardwareClusterDecoder
/// @brief Class to convert a list of input buffers containing TPC clusters of type ClusterHardware
/// to type ClusterNative.
///
/// This class transforms the hardware cluster raw format to the native cluster format. The output is
/// organized in blocks per sector-padrow, the sequence of cluster blocks is stored in a flat structure.
/// Cluster blocks consist of the property header ClusterGroupHeader with members sector, globalPadrow
/// and nClusters, and the clusters following.
///
/// An allocator needs to be provided to allocate a char buffer of specified size from the decoding,
/// function the cluster blocks are initialized inside the provided binary buffer.
///
/// Usage:
///
///     HardwareClusterDecoder decoder;
///     std::vector<char> outputBuffer;
///     auto outputAllocator = [&outputBuffer](size_t size) -> char* {
///       outputBuffer.resize(size);
///       return outputBuffer.data();
///     }
///     decoder( {pointer, n}, outputAllocator, &mcIn, &mcOut);
///
/// FIXME: The class should be in principle stateless, but right now it has an instance of the
/// integrator for the digittal currents.
class HardwareClusterDecoder
{
public:
  HardwareClusterDecoder() = default;
  ~HardwareClusterDecoder() = default;

  /// @brief Allocator function object to provide the output buffer
  using OutputAllocator = std::function<char*(size_t)>;

  /// @brief Decode clusters provided in raw pages
  /// The function uses an allocator object to request a raw char buffer of needed size. Inside this buffer,
  /// flat structures of type ClusterNativeBuffer are initialized, each block starting with a property header
  /// containing also the number of clusters immediately following the the property header.
  ///
  /// @param inputClusters   list of input pages, each entry a pair of pointer to first page and number of pages
  /// @param outputAllocator allocator object to provide the output buffer of specified size
  /// @param inMCLabels      optional pointer to MC label container
  /// @param outMCLabels     optional pointer to MC output container
  int decodeClusters(std::vector<std::pair<const o2::tpc::ClusterHardwareContainer*, std::size_t>>& inputClusters,
                     OutputAllocator outputAllocator,
                     const std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>* inMCLabels = nullptr,
                     std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>* outMCLabels = nullptr);

  /// @brief Sort clusters and MC labels in place
  /// ClusterNative defines the smaller-than relation used in the sorting, with time being the more significant
  /// condition in the comparison.
  static void sortClustersAndMC(ClusterNative* clusters, size_t nClusters,
                                o2::dataformats::MCTruthContainer<o2::MCCompLabel> mcTruth);

 private:
  std::unique_ptr<DigitalCurrentClusterIntegrator> mIntegrator;
};

} // namespace tpc
} // namespace o2
#endif
