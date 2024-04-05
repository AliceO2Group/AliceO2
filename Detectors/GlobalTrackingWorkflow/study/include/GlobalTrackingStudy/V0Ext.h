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

// class for extended V0 info (for debugging)

#ifndef ALICEO2_V0EXT_H
#define ALICEO2_V0EXT_H

#include "ReconstructionDataFormats/V0.h"

namespace o2
{
namespace dataformats
{

struct ProngInfoExt {
  o2::track::TrackParCov trackTPC;
  int nClTPC = 0;
  int nClITS = 0;
  int pattITS = 0;
  float chi2ITSTPC = 0.f;
  ClassDefNV(ProngInfoExt, 1);
};

struct V0Ext {
  V0 v0;
  V0Index v0ID;
  std::array<ProngInfoExt, 2> prInfo{};
  const ProngInfoExt& getPrInfo(int i) const { return prInfo[i]; }
  ClassDefNV(V0Ext, 1);
};

} // namespace dataformats
} // namespace o2

#endif
