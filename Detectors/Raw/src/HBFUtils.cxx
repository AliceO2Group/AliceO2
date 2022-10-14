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

#include "Framework/Logger.h"
#include "DetectorsRaw/HBFUtils.h"
#include <fairlogger/Logger.h>
#include <bitset>
#include <cassert>
#include <exception>

using namespace o2::raw;

O2ParamImpl(o2::raw::HBFUtils);

//_________________________________________________
uint32_t HBFUtils::getHBF(const IR& rec) const
{
  ///< get HBF ID corresponding to this IR
  auto diff = rec.differenceInBC(getFirstIR());
  if (diff < 0) {
    LOG(error) << "IR " << rec.bc << '/' << rec.orbit << " is ahead of the reference IR "
               << "0/" << orbitFirst;
    throw std::runtime_error("Requested IR is ahead of the reference IR");
  }
  return diff / o2::constants::lhc::LHCMaxBunches;
}

//_________________________________________________
int HBFUtils::fillHBIRvector(std::vector<IR>& dst, const IR& fromIR, const IR& toIR) const
{
  // Fill provided vector (cleaned) by interaction records (bc/orbit) for HBFs, considering
  // BCs between interaction records "fromIR" and "toIR" (inclusive).
  dst.clear();
  int hb0 = getHBF(fromIR), hb1 = getHBF(toIR);
  if (fromIR.bc != 0) { // unless we are just starting the HBF of fromIR, it was already counted
    hb0++;
  }
  for (int ihb = hb0; ihb <= hb1; ihb++) {
    dst.emplace_back(getIRHBF(ihb));
  }
  return dst.size();
}

//_________________________________________________
void HBFUtils::checkConsistency() const
{
  if (orbitFirstSampled < orbitFirst) {
    auto s = fmt::format("1st sampled orbit ({}) < 1st orbit of run ({})", orbitFirstSampled, orbitFirst);
    LOG(error) << s;
    throw std::runtime_error(s);
  }
}