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
#include "DetectorsRaw/RDHUtils.h"
#include <FairLogger.h>
#include <bitset>
#include <cassert>
#include <exception>

using namespace o2::raw;

//_________________________________________________
void RDHUtils::printRDH(const o2::header::RAWDataHeaderV4& rdh)
{
  std::bitset<32> trb(rdh.triggerType);
  LOGF(INFO, "EP:%d CRU:0x%04x Link:%-3d FEEID:0x%04x Packet:%-3d MemSize:%-4d OffsNext:%-4d prio.:%d BL:%-5d HS:%-2d HV:%d",
       int(rdh.endPointID), int(rdh.cruID), int(rdh.linkID), int(rdh.feeId), int(rdh.packetCounter), int(rdh.memorySize),
       int(rdh.offsetToNext), int(rdh.priority), int(rdh.blockLength), int(rdh.headerSize), int(rdh.version));
  LOGF(INFO, "HBOrb:%-9u TrOrb:%-9u Trg:%32s HBBC:%-4d TrBC:%-4d Page:%-5d Stop:%d Par:%-5d DetFld:0x%04x", //
       rdh.heartbeatOrbit, rdh.triggerOrbit, trb.to_string().c_str(), int(rdh.heartbeatBC), int(rdh.triggerBC),
       int(rdh.pageCnt), int(rdh.stop), int(rdh.par), int(rdh.detectorField));
}

//_________________________________________________
void RDHUtils::printRDH(const o2::header::RAWDataHeaderV5& rdh)
{
  std::bitset<32> trb(rdh.triggerType);
  LOGF(INFO, "EP:%d CRU:0x%04x Link:%-3d FEEID:0x%04x Packet:%-3d MemSize:%-5d OffsNext:%-5d  prio.:%d HS:%-2d HV:%d",
       int(rdh.endPointID), int(rdh.cruID), int(rdh.linkID), int(rdh.feeId), int(rdh.packetCounter), int(rdh.memorySize),
       int(rdh.offsetToNext), int(rdh.priority), int(rdh.headerSize), int(rdh.version));
  LOGF(INFO, "Orbit:%-9u BC:%-4d Stop:%d Page:%-5d Trg:%32s Par:%-5d DetFld:0x%04x",
       rdh.orbit, int(rdh.bunchCrossing), int(rdh.stop), int(rdh.pageCnt), trb.to_string().c_str(),
       int(rdh.detectorPAR), int(rdh.detectorField));
}

//_________________________________________________
void RDHUtils::printRDH(const void* rdhP)
{
  int version = (reinterpret_cast<const char*>(rdhP))[0];
  switch (version) {
    case 4:
      printRDH(*reinterpret_cast<const o2::header::RAWDataHeaderV4*>(rdhP));
      break;
    case 5:
      printRDH(*reinterpret_cast<const o2::header::RAWDataHeaderV5*>(rdhP));
      break;
    default:
      LOG(ERROR) << "Unexpected RDH version " << version << " from";
      dumpRDH(rdhP);
      throw std::runtime_error("invalid RDH provided");
      break;
  };
}

//_________________________________________________
void RDHUtils::dumpRDH(const void* rdhP)
{
  const uint32_t* w32 = reinterpret_cast<const uint32_t*>(rdhP);
  for (int i = 0; i < 4; i++) {
    int l = 4 * i;
    LOGF(INFO, "[rdh%d] 0x%08x 0x%08x 0x%08x 0x%08x", i, w32[l + 3], w32[l + 2], w32[l + 1], w32[l]);
  }
}

//_________________________________________________
o2::InteractionRecord RDHUtils::getHBIR(const void* rdhP)
{
  int version = (reinterpret_cast<const char*>(rdhP))[0];
  if (version == 4) {
    return getHBIR(*reinterpret_cast<const o2::header::RAWDataHeaderV4*>(rdhP));
  } else if (version == 5) {
    return getHBIR(*reinterpret_cast<const o2::header::RAWDataHeaderV5*>(rdhP));
  }
  LOG(ERROR) << "Unexpected RDH version " << version << " from";
  dumpRDH(rdhP);
  throw std::runtime_error("invalid RDH provided");
}

//_________________________________________________
bool RDHUtils::checkRDH(const void* rdhP, bool verbose)
{
  int version = (reinterpret_cast<const char*>(rdhP))[0];
  bool ok = true;
  switch (version) {
    case 3:
    case 4:
      ok = checkRDH(*reinterpret_cast<const o2::header::RAWDataHeaderV4*>(rdhP), verbose);
      break;
    case 5:
      ok = checkRDH(*reinterpret_cast<const o2::header::RAWDataHeaderV5*>(rdhP), verbose);
      break;
    default:
      ok = false;
      if (verbose) {
        LOG(ERROR) << "Unexpected RDH version " << version << " from";
      }
      break;
  };
  if (!ok && verbose) {
    dumpRDH(rdhP);
  }
  return ok;
}

//_________________________________________________
uint32_t RDHUtils::getHBOrbit(const void* rdhP)
{
  int version = (reinterpret_cast<const char*>(rdhP))[0];
  if (version == 4) {
    return getHBOrbit(*reinterpret_cast<const o2::header::RAWDataHeaderV4*>(rdhP));
  } else if (version == 5) {
    return getHBOrbit(*reinterpret_cast<const o2::header::RAWDataHeaderV5*>(rdhP));
  }
  LOG(ERROR) << "Unexpected RDH version " << version << " from";
  dumpRDH(rdhP);
  throw std::runtime_error("invalid RDH provided");
}

//_________________________________________________
uint16_t RDHUtils::getHBBC(const void* rdhP)
{
  int version = (reinterpret_cast<const char*>(rdhP))[0];
  if (version == 4) {
    return getHBBC(*reinterpret_cast<const o2::header::RAWDataHeaderV4*>(rdhP));
  } else if (version == 5) {
    return getHBBC(*reinterpret_cast<const o2::header::RAWDataHeaderV5*>(rdhP));
  }
  LOG(ERROR) << "Unexpected RDH version " << version << " from";
  dumpRDH(rdhP);
  throw std::runtime_error("invalid RDH provided");
}

//_____________________________________________________________________
bool RDHUtils::checkRDH(const o2::header::RAWDataHeaderV4& rdh, bool verbose)
{
  // check if rdh conforms with RDH4 fields
  bool ok = true;
  if (rdh.version != 4 && rdh.version != 3) {
    if (verbose) {
      LOG(ERROR) << "RDH version 4 is expected instead of " << int(rdh.version);
    }
    ok = false;
  }
  if (rdh.headerSize != 64) {
    if (verbose) {
      LOG(ERROR) << "RDH with header size of 64 B is expected instead of " << int(rdh.headerSize);
    }
    ok = false;
  }
  if (rdh.memorySize < 64 || rdh.offsetToNext < 64 || rdh.memorySize > MAXCRUPage || rdh.offsetToNext > MAXCRUPage) {
    if (verbose) {
      LOG(ERROR) << "RDH expected to have memorySize/offsetToNext in 64 : 8192 bytes range instead of "
                 << int(rdh.memorySize) << '/' << int(rdh.offsetToNext);
    }
    ok = false;
  }
  if (rdh.zero0 || rdh.word3 || rdh.zero41 || rdh.zero42 || rdh.word5 || rdh.zero6 || rdh.word7) {
    if (verbose) {
      LOG(ERROR) << "Some reserved fields of RDH v4 are not empty";
    }
    ok = false;
  }
  if (!ok && verbose) {
    dumpRDH(rdh);
  }
  return ok;
}

//_____________________________________________________________________
bool RDHUtils::checkRDH(const o2::header::RAWDataHeaderV5& rdh, bool verbose)
{
  // check if rdh conforms with RDH5 fields
  bool ok = true;
  if (rdh.version != 5) {
    if (verbose) {
      LOG(ERROR) << "RDH version 5 is expected instead of " << int(rdh.version);
    }
    ok = false;
  }
  if (rdh.headerSize != 64) {
    if (verbose) {
      LOG(ERROR) << "RDH with header size of 64 B is expected instead of " << int(rdh.headerSize);
    }
    ok = false;
  }
  if (rdh.memorySize < 64 || rdh.offsetToNext < 64) {
    if (verbose) {
      LOG(ERROR) << "RDH expected to have memory size and offset to next >= 64 B instead of "
                 << int(rdh.memorySize) << '/' << int(rdh.offsetToNext);
    }
    ok = false;
  }
  if (rdh.zero0 || rdh.word3 || rdh.zero4 || rdh.word5 || rdh.zero6 || rdh.word7) {
    if (verbose) {
      LOG(ERROR) << "Some reserved fields of RDH v5 are not empty";
    }
    ok = false;
  }
  if (!ok && verbose) {
    dumpRDH(rdh);
  }
  return ok;
}

/// temporary: provide a hashcode (checksum) for RDH fields to be used as susbspec
/// until unique soureID / FeeID is implemented
/// Source: https://en.wikipedia.org/wiki/Fletcher%27s_checksum
uint32_t RDHUtils::fletcher32(const uint16_t* data, int len)
{
  uint32_t c0, c1;
  // We similarly solve for n > 0 and n * (n+1) / 2 * (2^16-1) < (2^32-1) here.
  // On modern computers, using a 64-bit c0/c1 could allow a group size of 23726746.
  for (c0 = c1 = 0; len > 0; len -= 360) {
    int blocklen = len < 360 ? len : 360;
    for (int i = 0; i < blocklen; ++i) {
      c0 = c0 + *data++;
      c1 = c1 + c0;
    }
    c0 = c0 % 65535;
    c1 = c1 % 65535;
  }
  return (c1 << 16 | c0);
}
