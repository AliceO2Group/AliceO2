// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// @brief Class for operations with RawDataHeader
// @author ruben.shahoyan@cern.ch

#ifndef ALICEO2_RDHUTILS_H
#define ALICEO2_RDHUTILS_H

#include <Rtypes.h>
#include "CommonDataFormat/InteractionRecord.h"
#include "Headers/RAWDataHeader.h"

namespace o2
{
namespace raw
{
using LinkSubSpec_t = uint32_t;
using IR = o2::InteractionRecord;

struct RDHUtils {
  static constexpr int GBTWord = 16; // length of GBT word
  static constexpr int MAXCRUPage = 512 * GBTWord;

  // some fields of the same meaning have different names in the RDH of different versions
  static uint32_t getHBOrbit(const void* rdhP);
  static uint16_t getHBBC(const void* rdhP);
  static IR getHBIR(const void* rdhP);

  static void printRDH(const void* rdhP);
  static void dumpRDH(const void* rdhP);

  static bool checkRDH(const void* rdhP, bool verbose = true);

  static uint32_t getHBOrbit(const o2::header::RAWDataHeaderV4& rdh) { return rdh.heartbeatOrbit; }

  static uint32_t getHBOrbit(const o2::header::RAWDataHeaderV5& rdh) { return rdh.orbit; }

  static uint16_t getHBBC(const o2::header::RAWDataHeaderV4& rdh) { return rdh.heartbeatBC; }
  static uint16_t getHBBC(const o2::header::RAWDataHeaderV5& rdh) { return rdh.bunchCrossing; }

  static IR getHBIR(const o2::header::RAWDataHeaderV4& rdh) { return {uint16_t(rdh.heartbeatBC), uint32_t(rdh.heartbeatOrbit)}; }
  static IR getHBIR(const o2::header::RAWDataHeaderV5& rdh) { return {uint16_t(rdh.bunchCrossing), uint32_t(rdh.orbit)}; }

  static void printRDH(const o2::header::RAWDataHeaderV5& rdh);
  static void printRDH(const o2::header::RAWDataHeaderV4& rdh);
  static void dumpRDH(const o2::header::RAWDataHeaderV5& rdh) { dumpRDH(&rdh); }
  static void dumpRDH(const o2::header::RAWDataHeaderV4& rdh) { dumpRDH(&rdh); }

  static bool checkRDH(const o2::header::RAWDataHeaderV4& rdh, bool verbose = true);
  static bool checkRDH(const o2::header::RAWDataHeaderV5& rdh, bool verbose = true);

  static LinkSubSpec_t getSubSpec(uint16_t cru, uint8_t link, uint8_t endpoint, uint16_t feeId);
  static LinkSubSpec_t getSubSpec(const o2::header::RAWDataHeaderV4& rdh) { return getSubSpec(rdh.cruID, rdh.linkID, rdh.endPointID, rdh.feeId); }
  static LinkSubSpec_t getSubSpec(const o2::header::RAWDataHeaderV5& rdh) { return getSubSpec(rdh.cruID, rdh.linkID, rdh.endPointID, rdh.feeId); }

 private:
  static uint32_t fletcher32(const uint16_t* data, int len);

  ClassDefNV(RDHUtils, 1);
};

//_____________________________________________________________________
inline LinkSubSpec_t RDHUtils::getSubSpec(uint16_t cru, uint8_t link, uint8_t endpoint, uint16_t feeId)
{
  /*
  // RS Temporarily suppress this way since such a subspec does not define the TOF/TPC links in a unique way
  // define subspecification as in DataDistribution
  int linkValue = (LinkSubSpec_t(link) + 1) << (endpoint == 1 ? 8 : 0);
  return (LinkSubSpec_t(cru) << 16) | linkValue;
  */
  // RS Temporarily suppress this way since such a link is ambiguous
  uint16_t seq[3] = {cru, uint16_t((uint16_t(link) << 8) | endpoint), feeId};
  return fletcher32(seq, 3);
}

} // namespace raw
} // namespace o2

#endif //ALICEO2_RDHUTILS_H
