// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "CommonUtils/HBFUtils.h"
#include <FairLogger.h>
#include <bitset>
#include <cassert>

using namespace o2::utils;

//_________________________________________________
int HBFUtils::getHBF(const IR& rec) const
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
int HBFUtils::fillHBIRvector(std::vector<IR>& dst, const IR& fromIR, const IR& toIR) const
{
  // Fill provided vector (cleaned) by interaction records (bc/orbit) for HBFs, considering
  // BCs between interaction records "fromIR" and "toIR" (inclusive).
  dst.clear();
  int hb0 = getHBF(fromIR), hb1 = getHBF(toIR);
  if (fromIR.bc != mFirstIR.bc) { // unless we just starting the HBF of fromIR, it was already counted
    hb0++;
  }
  for (int ihb = hb0; ihb <= hb1; ihb++) {
    dst.emplace_back(getIRHBF(ihb));
  }
  return dst.size();
}

//_________________________________________________
void HBFUtils::print() const
{
  printf("%d HBF per TF, starting from ", mNHBFPerTF);
  mFirstIR.print();
}

//_________________________________________________
void HBFUtils::printRDH(const o2::header::RAWDataHeaderV4& rdh)
{
  std::bitset<32> trb(rdh.triggerType);
  printf(
    "EP:%d CRU:0x%04x Packet:%-3d Link:%-3d MemSize:%-5d OffsNext:%-5d  prio.:%d FEEID:0x%04x BL:%-5d HS:%-2d HV:%d\n"
    "HBOrb:%-9u TrOrb:%-9u "
    "Trg:%32s HBBC:%-4d TrBC:%-4d "
    "Page:%-5d Stop:%d Par:%-5d DetFld:0x%04x\n",
    int(rdh.endPointID), int(rdh.cruID), int(rdh.packetCounter), int(rdh.linkID), int(rdh.memorySize),
    int(rdh.offsetToNext), int(rdh.priority), int(rdh.feeId), int(rdh.blockLength), int(rdh.headerSize), int(rdh.version),
    //
    rdh.heartbeatOrbit, rdh.triggerOrbit,
    //
    trb.to_string().c_str(), int(rdh.heartbeatBC), int(rdh.triggerBC),
    //
    int(rdh.pageCnt), int(rdh.stop), int(rdh.par), int(rdh.detectorField));
}

//_________________________________________________
void HBFUtils::printRDH(const o2::header::RAWDataHeaderV5& rdh)
{
  std::bitset<32> trb(rdh.triggerType);
  printf(
    "EP:%d CRU:0x%04x Packet:%-3d Link:%-3d MemSize:%-5d OffsNext:%-5d  prio.:%d FEEID:0x%04x HS:%-2d HV:%d\n"
    "Orbit:%-9u BC:%-4d "
    "Stop:%d Page:%-5d Trg:%32s "
    "Par:%-5d DetFld:0x%04x\n",
    int(rdh.endPointID), int(rdh.cruID), int(rdh.packetCounter), int(rdh.linkID), int(rdh.memorySize),
    int(rdh.offsetToNext), int(rdh.priority), int(rdh.feeId), int(rdh.headerSize), int(rdh.version),
    //
    rdh.orbit, int(rdh.bunchCrossing),
    //
    int(rdh.stop), int(rdh.pageCnt), trb.to_string().c_str(),
    //
    int(rdh.detectorPAR), int(rdh.detectorField));
}

//_________________________________________________
void HBFUtils::dumpRDH(const o2::header::RAWDataHeaderV5& rdh)
{
  constexpr int hsz = sizeof(o2::header::RAWDataHeaderV5);
  static_assert(hsz == 64, "Expect 64 bytes long header");
  const uint32_t* w32 = reinterpret_cast<const uint32_t*>(&rdh);
  for (int iw128 = 0; iw128 < hsz / 16; iw128++) {
    printf("[rdh%d] 0x:", iw128);
    for (int iw = hsz / sizeof(uint32_t); iw--;) {
      printf(" %08x", w32[iw128 * 16 / sizeof(uint32_t) + iw]);
    }
    printf("\n");
  }
}
