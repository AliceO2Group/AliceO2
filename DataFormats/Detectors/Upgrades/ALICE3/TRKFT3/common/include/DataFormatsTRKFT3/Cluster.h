// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_DATAFORMATSTRKFT3_CLUSTER_H
#define ALICEO2_DATAFORMATSTRKFT3_CLUSTER_H

#include "DetectorsCommonDataFormats/DetID.h"
#include <Rtypes.h>
#include <cstdint>
#include <sstream>
#include <string>

namespace o2::trkft3
{

template <int DetID>
struct Cluster {
  static_assert(DetID == o2::detectors::DetID::TRK || DetID == o2::detectors::DetID::FT3, "only TRK and FT3 clusters are supported");

  uint16_t chipID = 0;
  uint16_t row = 0;
  uint16_t col = 0;
  uint16_t size = 1;
  int16_t subDetID = -1;
  int16_t layer = -1;

  std::string asString() const
  {
    std::ostringstream stream;
    stream << o2::detectors::DetID(DetID).getName() << " cluster chip=" << chipID
           << " row=" << row << " col=" << col << " size=" << size
           << " subDet=" << subDetID << " layer=" << layer;
    return stream.str();
  }

  ClassDefNV(Cluster, 1);
};

using TRKCluster = Cluster<o2::detectors::DetID::TRK>;
using FT3Cluster = Cluster<o2::detectors::DetID::FT3>;

} // namespace o2::trkft3

#endif
