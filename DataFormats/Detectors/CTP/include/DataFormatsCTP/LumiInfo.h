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
#include "CommonConstants/LHCConstants.h"
#include <Rtypes.h>
#include <cmath>

/// \brief Luminosity information as a moving average over certain number of TFs

namespace o2
{
namespace ctp
{
struct LumiInfo {
  LumiInfo() = default;
  uint32_t orbit = 0;       // orbit of TF when was updated
  uint32_t nHBFCounted = 0; // length of interval in HB
  uint32_t nHBFCountedFV0 = 0;
  uint64_t counts = 0;      // counts in the interval for the nominal lumi detector (FT0)
  uint64_t countsFV0 = 0;   // connts for FV0 (less reliable)
  int inp1 = 3;             // TVX
  int inp2 = 6;             // VBA
  float getLumi() const { return nHBFCounted > 0 ? float(counts / (nHBFCounted * o2::constants::lhc::LHCOrbitMUS * 1e-6)) : 0.f; }
  float getLumiFV0() const { return nHBFCountedFV0 > 0 ? float(countsFV0 / (nHBFCountedFV0 * o2::constants::lhc::LHCOrbitMUS * 1e-6)) : 0.f; }
  float getLumiAlt() const { return getLumiFV0(); }
  float getLumiError() const { return nHBFCounted > 0 ? float(std::sqrt(counts) / (nHBFCounted * o2::constants::lhc::LHCOrbitMUS * 1e-6)) : 0.f; }
  float getLumiFV0Error() const { return nHBFCountedFV0 > 0 ? float(std::sqrt(countsFV0) / (nHBFCountedFV0 * o2::constants::lhc::LHCOrbitMUS * 1e-6)) : 0.f; }
  float getLumiAltError() const { return getLumiFV0Error(); }
  void printInputs() const;
  ClassDefNV(LumiInfo, 3);
};
} // namespace ctp

} // namespace o2

#endif // _ALICEO2_CTP_LUMIINFO_H_
