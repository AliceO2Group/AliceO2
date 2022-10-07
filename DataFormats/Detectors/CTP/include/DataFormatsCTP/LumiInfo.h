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

#ifndef _ALICEO2_CTP_LUMIINFO_H_
#define _ALICEO2_CTP_LUMIINFO_H_
#include "CommonDataFormat/InteractionRecord.h"

/// \brief Luminosity information used of online TPC calibration

namespace o2
{
namespace ctp
{
struct LumiInfo {
  LumiInfo() = default;
  InteractionRecord ir;    // timestamp of start of lumi interval
  size_t mNHBFCounted = 0; // length of interval in HB
  size_t mCounts = 0;      //  counts in the interval
};
} // namespace ctp

} // namespace o2

#endif // _ALICEO2_CTP_LUMIINFO_H_
