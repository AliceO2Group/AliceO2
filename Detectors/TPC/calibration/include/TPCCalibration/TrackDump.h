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

#ifndef O2_TPC_TrackDump_H_
#define O2_TPC_TrackDump_H_

#include <Rtypes.h>
#include <cstdint>
#include <memory>
#include <string>
#include <gsl/span>
#include <vector>

#include "DataFormatsTPC/TrackTPC.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "CommonUtils/TreeStreamRedirector.h"

/// \file TrackDump.h
/// \author Jens Wiechula (Jens.Wiechula@ikf.uni-frankfurt.de)

namespace o2::tpc
{

class ClusterNativeAccess;

/// The class can be used to dump track and associated clusters to a tree to easily iterate over them and draw them
class TrackDump
{
 public:
  struct ClusterGlobal {
    float gx{};
    float gy{};
    uint16_t qMax; //< QMax of the cluster
    uint16_t qTot; //< Total charge of the cluster
    uint8_t sector = 0;
    uint8_t padrow = 0;

    ClassDefNV(ClusterGlobal, 1);
  };

  struct ClusterNativeAdd : public ClusterNative {
    ClusterNativeAdd() = default;
    ClusterNativeAdd(const ClusterNative& cl) : ClusterNative(cl){};
    ~ClusterNativeAdd() = default;

    // float z = 0.f;

    float tgl = 0.f;
    float snp = 0.f;
    uint8_t sector = 0;
    uint8_t padrow = 0;

    float lx() const;
    float ly() const;
    float gx() const;
    float gy() const;
    float cpad() const;

    ClassDefNV(ClusterNativeAdd, 1);
  };

  struct TrackInfo : public TrackTPC {
    TrackInfo() = default;
    TrackInfo(const TrackTPC& track) : TrackTPC(track){};
    TrackInfo(const TrackInfo&) = default;
    ~TrackInfo() = default;

    std::vector<ClusterNativeAdd> clusters{};

    ClassDefNV(TrackInfo, 1);
  };

  void filter(const gsl::span<const TrackTPC> tracks, ClusterNativeAccess const& clusterIndex, const gsl::span<const o2::tpc::TPCClRefElem> clRefs, const gsl::span<const MCCompLabel> mcLabels);
  void finalize();

  std::string outputFileName{"filtered-tracks-and-clusters.root"}; ///< Name of the output file with the tree
  bool writeTracks{true};                                          ///< write global cluster information for quick drawing
  bool writeGlobal{false};                                         ///< write global cluster information for quick drawing
  bool writeMC{false};                                             ///< write MC track information for quick drawing
 private:
  std::unique_ptr<o2::utils::TreeStreamRedirector> mTreeDump; ///< Tree writer
};
} // namespace o2::tpc
#endif
