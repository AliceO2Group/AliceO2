// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Steer/HBFSampler.h"
#include <FairLogger.h>

using namespace o2::steer;

//_________________________________________________
int HBFSampler::getHBF(const IR& rec) const
{
  ///< get HBF ID corresponding to this IR
  auto diff = rec.differenceInBC(mFirstIR);
  if (diff < 0) {
    LOG(FATAL) << "IR " << rec.bc << '/' << rec.orbit << " is ahead of the reference IR "
               << mFirstIR.bc << '/' << mFirstIR.orbit;
  }
  return diff / o2::constants::lhc::LHCMaxBunches;
}

//_________________________________________________
int HBFSampler::fillHBIRvector(std::vector<IR>& dst, const IR& fromIR, const IR& toIR) const
{
  // Fill provided vector (cleaned) by interaction records (bc/orbit) for HBFs, considering
  // BCs between interaction records "fromIR" and "toIR" (inclusive).
  dst.clear();
  int hb0 = getHBF(fromIR), hb1 = getHBF(toIR);
  if (fromIR.bc != mFirstIR.bc) { // unless we just starting the HBF of fromIR, it was already counted
    hb0++;
  }
  for (int ihb = hb0; ihb <= hb1; ihb++) {
    dst.emplace_back(getIR(ihb));
  }
  return dst.size();
}

//_________________________________________________
void HBFSampler::print() const
{
  printf("%d HBF per TF, starting from ", mNHBFPerTF);
  mFirstIR.print();
}
