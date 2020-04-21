// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TPCEntropyEncoder.cxx
/// @author Michael Lettrich
/// @since  Apr 30, 2020
/// @brief static class that performs rans encoding of tpc::Compressed clustets and serializes to TTree.

#include "TPCEntropyCoding/TPCEntropyEncoder.h"

#include <Compression.h>

namespace o2
{
namespace tpc
{

std::unique_ptr<EncodedClusters> TPCEntropyEncoder::encode(const CompressedClusters& clusters)
{
  auto encodedClusters = std::make_unique<o2::tpc::EncodedClusters>();

  // header
  encodedClusters->header = new o2::tpc::EncodedClusters::Header{0, 1};

  //counters
  encodedClusters->counters = new o2::tpc::EncodedClusters::Counters{clusters.nTracks,
                                                                     clusters.nAttachedClusters,
                                                                     clusters.nUnattachedClusters,
                                                                     clusters.nAttachedClustersReduced,
                                                                     clusters.nSliceRows};

  // vector of metadata
  encodedClusters->metadata = new std::vector<o2::tpc::EncodedClusters::Metadata>;

  //encode data
  compress("qTotA", clusters.qTotA, clusters.qTotA + clusters.nAttachedClusters, sProbabilityBits16Bit, *encodedClusters);
  compress("qMaxA", clusters.qMaxA, clusters.qMaxA + clusters.nAttachedClusters, sProbabilityBits16Bit, *encodedClusters);
  compress("flagsA", clusters.flagsA, clusters.flagsA + clusters.nAttachedClusters, sProbabilityBits8Bit, *encodedClusters);
  compress("rowDiffA", clusters.rowDiffA, clusters.rowDiffA + clusters.nAttachedClustersReduced, sProbabilityBits8Bit, *encodedClusters);
  compress("sliceLegDiffA", clusters.sliceLegDiffA, clusters.sliceLegDiffA + clusters.nAttachedClustersReduced, sProbabilityBits8Bit, *encodedClusters);
  compress("padResA", clusters.padResA, clusters.padResA + clusters.nAttachedClustersReduced, sProbabilityBits16Bit, *encodedClusters);
  compress("timeResA", clusters.timeResA, clusters.timeResA + clusters.nAttachedClustersReduced, sProbabilityBits25Bit, *encodedClusters);
  compress("sigmaPadA", clusters.sigmaPadA, clusters.sigmaPadA + clusters.nAttachedClusters, sProbabilityBits8Bit, *encodedClusters);
  compress("sigmaTimeA", clusters.sigmaTimeA, clusters.sigmaTimeA + clusters.nAttachedClusters, sProbabilityBits8Bit, *encodedClusters);
  compress("qPtA", clusters.qPtA, clusters.qPtA + clusters.nTracks, sProbabilityBits8Bit, *encodedClusters);
  compress("rowA", clusters.rowA, clusters.rowA + clusters.nTracks, sProbabilityBits8Bit, *encodedClusters);
  compress("sliceA", clusters.sliceA, clusters.sliceA + clusters.nTracks, sProbabilityBits8Bit, *encodedClusters);
  compress("timeA", clusters.timeA, clusters.timeA + clusters.nTracks, sProbabilityBits25Bit, *encodedClusters);
  compress("padA", clusters.padA, clusters.padA + clusters.nTracks, sProbabilityBits16Bit, *encodedClusters);
  compress("qTotU", clusters.qTotU, clusters.qTotU + clusters.nUnattachedClusters, sProbabilityBits16Bit, *encodedClusters);
  compress("qMaxU", clusters.qMaxU, clusters.qMaxU + clusters.nUnattachedClusters, sProbabilityBits16Bit, *encodedClusters);
  compress("flagsU", clusters.flagsU, clusters.flagsU + clusters.nUnattachedClusters, sProbabilityBits8Bit, *encodedClusters);
  compress("padDiffU", clusters.padDiffU, clusters.padDiffU + clusters.nUnattachedClusters, sProbabilityBits16Bit, *encodedClusters);
  compress("timeDiffU", clusters.timeDiffU, clusters.timeDiffU + clusters.nUnattachedClusters, sProbabilityBits25Bit, *encodedClusters);
  compress("sigmaPadU", clusters.sigmaPadU, clusters.sigmaPadU + clusters.nUnattachedClusters, sProbabilityBits8Bit, *encodedClusters);
  compress("sigmaTimeU", clusters.sigmaTimeU, clusters.sigmaTimeU + clusters.nUnattachedClusters, sProbabilityBits8Bit, *encodedClusters);
  compress("nTrackClusters", clusters.nTrackClusters, clusters.nTrackClusters + clusters.nTracks, sProbabilityBits16Bit, *encodedClusters);
  compress("nSliceRowClusters", clusters.nSliceRowClusters, clusters.nSliceRowClusters + clusters.nSliceRows, sProbabilityBits25Bit, *encodedClusters);

  return std::move(encodedClusters);
}

void TPCEntropyEncoder::appendToTTree(TTree& tree, EncodedClusters& encodedClusters)
{
  tree.Branch("Header", &encodedClusters.header);
  tree.Branch("Counters", &encodedClusters.counters);
  tree.Branch("Metadata", &encodedClusters.metadata);

  for (size_t i = 0; i < encodedClusters.NUM_ARRAYS; i++) {
    const auto name = encodedClusters.NAMES[i];
    const auto dictName = std::string(name) + "Dict";

    auto& dicts = encodedClusters.dicts;
    auto& buffers = encodedClusters.buffers;

    assert(dicts[i]);
    assert(buffers[i]);

    tree.Branch(dictName.c_str(), &dicts[i]);
    auto branch = tree.Branch(name, &buffers[i]);
    branch->SetCompressionLevel(ROOT::RCompressionSetting::ELevel::kUncompressed);
  }

  // fill and write
  tree.Fill();
}

size_t TPCEntropyEncoder::calculateMaxBufferSize(size_t num, size_t rangeBits, size_t sizeofStreamT)
{
  return std::ceil(1.20 * (num * rangeBits * 1.0) / (sizeofStreamT * 8.0));
}

} // namespace tpc
} // namespace o2
