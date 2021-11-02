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

#ifndef ALICEO2_BASE_STF_HEADER_
#define ALICEO2_BASE_STF_HEADER_

#include <cstdint>
#include <string>

namespace o2::header
{

struct STFHeader { // fake header to mimic DD SubTimeFrame::Header sent with DISTSUBTIMEFRAME message
  uint64_t id = uint64_t(-1);
  uint32_t firstOrbit = uint32_t(-1);
  std::uint32_t runNumber = 0;

  std::string asString();
};

} // namespace o2::header

#endif
