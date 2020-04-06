// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/Logger.h"
#include "DetectorsRaw/HBFUtils.h"
#include "DetectorsRaw/RDHUtils.h"
#include <FairLogger.h>
#include <bitset>
#include <cassert>
#include <exception>

using namespace o2::raw;

O2ParamImpl(o2::raw::HBFUtils);

//_________________________________________________
int64_t HBFUtils::getHBF(const IR& rec) const
{
  ///< get HBF ID corresponding to this IR
  auto diff = rec.differenceInBC(getFirstIR());
  if (diff < 0) {
    LOG(ERROR) << "IR " << rec.bc << '/' << rec.orbit << " is ahead of the reference IR "
               << bcFirst << '/' << orbitFirst;
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
  if (fromIR.bc != bcFirst) { // unless we are just starting the HBF of fromIR, it was already counted
    hb0++;
  }
  for (int ihb = hb0; ihb <= hb1; ihb++) {
    dst.emplace_back(getIRHBF(ihb));
  }
  return dst.size();
}
