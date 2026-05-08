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

#ifndef ALICEO2_DATAFORMATSTRK_CLUSTER_H
#define ALICEO2_DATAFORMATSTRK_CLUSTER_H

#include <Rtypes.h>
#include <cstdint>
#include <string>

namespace o2::trk
{

struct Cluster {
  uint16_t chipID = 0;
  uint16_t row = 0;
  uint16_t col = 0;
  uint16_t size = 1;
  int16_t subDetID = -1;
  int16_t layer = -1;
  int16_t disk = -1;

  std::string asString() const;

  ClassDefNV(Cluster, 1);
};

} // namespace o2::trk

#endif
