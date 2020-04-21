// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   EncodedClusters.h
/// @author Michael Lettrich
/// @since  Apr 19, 2020
/// @brief

#ifndef TPCENTROPYCODING_ENCODEDCLUSTERS_H_
#define TPCENTROPYCODING_ENCODEDCLUSTERS_H_

#include <cstdint>
#include <array>
#include <vector>

namespace o2
{
namespace tpc
{

struct EncodedClusters {

  struct Header {
    uint8_t majorVersion;
    uint8_t minorVersion;
    ClassDefNV(Header, 2);
  };

  struct Metadata {
    uint8_t coderType = 0;
    uint8_t streamSize = 0;
    uint8_t probabilityBits = 0;

    int32_t min = 0;
    int32_t max = 0;
    ClassDefNV(Metadata, 2);
  };

  struct Counters {
    unsigned int nTracks = 0;
    unsigned int nAttachedClusters = 0;
    unsigned int nUnattachedClusters = 0;
    unsigned int nAttachedClustersReduced = 0;
    unsigned int nSliceRows = 0;
    unsigned char nComppressionModes = 0;
    ClassDefNV(Counters, 2);
  };

  EncodedClusters() = default;
  ~EncodedClusters()
  {
    if (header) {
      delete header;
    }
    if (counters) {
      delete counters;
    }

    if (metadata) {
      delete metadata;
    }

    for (auto dict : dicts) {
      if (dict) {
        delete dict;
      }
    }

    for (auto buffer : buffers) {
      if (buffer) {
        delete buffer;
      }
    }
  }

  // version information
  Header* header = nullptr;
  // count of each object
  Counters* counters = nullptr;

  static constexpr size_t NUM_ARRAYS = 23;
  static constexpr std::array<const char*, NUM_ARRAYS> NAMES{"qTotA", "qMaxA",
                                                             "flagsA", "rowDiffA", "sliceLegDiffA", "padResA", "timeResA", "sigmaPadA",
                                                             "sigmaTimeA", "qPtA", "rowA", "sliceA", "timeA", "padA", "qTotU", "qMaxU",
                                                             "flagsU", "padDiffU", "timeDiffU", "sigmaPadU", "sigmaTimeU", "nTrackClusters", "nSliceRowClusters"};
  std::vector<Metadata>* metadata = nullptr;
  std::array<std::vector<uint32_t>*, NUM_ARRAYS> dicts;
  std::array<std::vector<uint32_t>*, NUM_ARRAYS> buffers;
};
} // namespace tpc
} // namespace o2

#endif /* TPCENTROPYCODING_ENCODEDCLUSTERS_H_ */
