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
#include "DetectorsRaw/RDHUtils.h"
#include "CommonUtils/StringUtils.h"
#include <fairlogger/Logger.h>
#include <bitset>
#include <cassert>
#include <exception>

using namespace o2::raw;
using namespace o2::header;

//=================================================

//_________________________________________________
void RDHUtils::printRDH(const RAWDataHeaderV4& rdh)
{
  std::bitset<32> trb(rdh.triggerType);
  LOGF(info, "EP:%d CRU:0x%04x Link:%-3d FEEID:0x%04x Packet:%-3d MemSize:%-4d OffsNext:%-4d prio.:%d BL:%-5d HS:%-2d HV:%d",
       int(rdh.endPointID), int(rdh.cruID), int(rdh.linkID), int(rdh.feeId), int(rdh.packetCounter), int(rdh.memorySize),
       int(rdh.offsetToNext), int(rdh.priority), int(rdh.blockLength), int(rdh.headerSize), int(rdh.version));
  LOGF(info, "HBOrb:%-9u TrOrb:%-9u Trg:%32s HBBC:%-4d TrBC:%-4d Page:%-5d Stop:%d Par:%-5d DetFld:0x%04x", //
       rdh.heartbeatOrbit, rdh.triggerOrbit, trb.to_string().c_str(), int(rdh.heartbeatBC), int(rdh.triggerBC),
       int(rdh.pageCnt), int(rdh.stop), int(rdh.par), int(rdh.detectorField));
}

//_________________________________________________
void RDHUtils::printRDH(const RAWDataHeaderV5& rdh)
{
  std::bitset<32> trb(rdh.triggerType);
  LOGF(info, "EP:%d CRU:0x%04x Link:%-3d FEEID:0x%04x Packet:%-3d MemSize:%-5d OffsNext:%-5d  prio.:%d HS:%-2d HV:%d",
       int(rdh.endPointID), int(rdh.cruID), int(rdh.linkID), int(rdh.feeId), int(rdh.packetCounter), int(rdh.memorySize),
       int(rdh.offsetToNext), int(rdh.priority), int(rdh.headerSize), int(rdh.version));
  LOGF(info, "Orbit:%-9u BC:%-4d Stop:%d Page:%-5d Trg:%32s Par:%-5d DetFld:0x%04x",
       rdh.orbit, int(rdh.bunchCrossing), int(rdh.stop), int(rdh.pageCnt), trb.to_string().c_str(),
       int(rdh.detectorPAR), int(rdh.detectorField));
}

//_________________________________________________
void RDHUtils::printRDH(const RAWDataHeaderV7& rdh)
{
  std::bitset<32> trb(rdh.triggerType);
  LOGF(info, "EP:%d CRU:0x%04x Link:%-3d FEEID:0x%04x SrcID:%s[%d] Packet:%-3d MemSize:%-5d OffsNext:%-5d  prio.:%d HS:%-2d HV:%d",
       int(rdh.endPointID), int(rdh.cruID), int(rdh.linkID), int(rdh.feeId), DAQID::DAQtoO2(rdh.sourceID).str, int(rdh.sourceID), int(rdh.packetCounter),
       int(rdh.memorySize), int(rdh.offsetToNext), int(rdh.priority), int(rdh.headerSize), int(rdh.version));
  LOGF(info, "Orbit:%-9u BC:%-4d DataFormat:%-2d Stop:%d Page:%-5d Trg:%32s Par:%-5d DetFld:0x%04x",
       rdh.orbit, int(rdh.bunchCrossing), int(rdh.dataFormat), int(rdh.stop), int(rdh.pageCnt), trb.to_string().c_str(),
       int(rdh.detectorPAR), int(rdh.detectorField));
}

//_________________________________________________
void RDHUtils::printRDH(const RAWDataHeaderV6& rdh)
{
  std::bitset<32> trb(rdh.triggerType);
  LOGF(info, "EP:%d CRU:0x%04x Link:%-3d FEEID:0x%04x SrcID:%s[%d] Packet:%-3d MemSize:%-5d OffsNext:%-5d  prio.:%d HS:%-2d HV:%d",
       int(rdh.endPointID), int(rdh.cruID), int(rdh.linkID), int(rdh.feeId), DAQID::DAQtoO2(rdh.sourceID).str, int(rdh.sourceID), int(rdh.packetCounter),
       int(rdh.memorySize), int(rdh.offsetToNext), int(rdh.priority), int(rdh.headerSize), int(rdh.version));
  LOGF(info, "Orbit:%-9u BC:%-4d Stop:%d Page:%-5d Trg:%32s Par:%-5d DetFld:0x%04x",
       rdh.orbit, int(rdh.bunchCrossing), int(rdh.stop), int(rdh.pageCnt), trb.to_string().c_str(),
       int(rdh.detectorPAR), int(rdh.detectorField));
}

//_________________________________________________
void RDHUtils::printRDH(const void* rdhP)
{
  int version = getVersion(rdhP);
  switch (version) {
    case 3:
    case 4:
      printRDH(*reinterpret_cast<const RAWDataHeaderV4*>(rdhP));
      break;
    case 5:
      printRDH(*reinterpret_cast<const RAWDataHeaderV5*>(rdhP));
      break;
    case 6:
      printRDH(*reinterpret_cast<const RAWDataHeaderV6*>(rdhP));
      break;
    case 7:
      printRDH(*reinterpret_cast<const RAWDataHeaderV7*>(rdhP));
      break;
    default:
      LOG(error) << "Unexpected RDH version " << version << " from";
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
    LOGF(info, "[rdh%d] 0x%08x 0x%08x 0x%08x 0x%08x", i, w32[l + 3], w32[l + 2], w32[l + 1], w32[l]);
  }
}

//_________________________________________________
bool RDHUtils::checkRDH(const void* rdhP, bool verbose, bool checkZeros)
{
  int version = getVersion(rdhP);
  bool ok = true;
  switch (version) {
    case 7:
      ok = checkRDH(*reinterpret_cast<const RAWDataHeaderV7*>(rdhP), verbose, checkZeros);
      break;
    case 6:
      ok = checkRDH(*reinterpret_cast<const RAWDataHeaderV6*>(rdhP), verbose, checkZeros);
      break;
    case 3:
    case 4:
      ok = checkRDH(*reinterpret_cast<const RAWDataHeaderV4*>(rdhP), verbose, checkZeros);
      break;
    case 5:
      ok = checkRDH(*reinterpret_cast<const RAWDataHeaderV5*>(rdhP), verbose, checkZeros);
      break;
    default:
      ok = false;
      if (verbose) {
        LOG(alarm) << "Unexpected RDH version " << version << " from";
      }
      break;
  };
  if (!ok && verbose) {
    dumpRDH(rdhP);
  }
  return ok;
}

//_____________________________________________________________________
bool RDHUtils::checkRDH(const RAWDataHeaderV4& rdh, bool verbose, bool checkZeros)
{
  // check if rdh conforms with RDH4 fields
  bool ok = true;
  if (rdh.version != 4 && rdh.version != 3) {
    if (verbose) {
      LOG(alarm) << "RDH version 4 is expected instead of " << int(rdh.version);
    }
    ok = false;
  }
  if (rdh.headerSize != 64) {
    if (verbose) {
      LOG(alarm) << "RDH with header size of 64 B is expected instead of " << int(rdh.headerSize);
    }
    ok = false;
  }
  if (rdh.memorySize < 64 || rdh.offsetToNext < 64 || rdh.memorySize > MAXCRUPage || rdh.offsetToNext > MAXCRUPage) {
    if (verbose) {
      LOG(alarm) << "RDH expected to have memorySize/offsetToNext in 64 : 8192 bytes range instead of "
                 << int(rdh.memorySize) << '/' << int(rdh.offsetToNext);
    }
    ok = false;
  }
  if (checkZeros && (rdh.zero0 || rdh.word3 || rdh.zero41 || rdh.zero42 || rdh.word5 || rdh.zero6 || rdh.word7)) {
    if (verbose) {
      LOG(alarm) << "Some reserved fields of RDH v4 are not empty";
    }
    ok = false;
  }
  if (!ok && verbose) {
    dumpRDH(rdh);
  }
  return ok;
}

//_____________________________________________________________________
bool RDHUtils::checkRDH(const RAWDataHeaderV5& rdh, bool verbose, bool checkZeros)
{
  // check if rdh conforms with RDH5 fields
  bool ok = true;
  if (rdh.version != 5) {
    if (verbose) {
      LOG(alarm) << "RDH version 5 is expected instead of " << int(rdh.version);
    }
    ok = false;
  }
  if (rdh.headerSize != 64) {
    if (verbose) {
      LOG(alarm) << "RDH with header size of 64 B is expected instead of " << int(rdh.headerSize);
    }
    ok = false;
  }
  if (rdh.memorySize < 64 || rdh.offsetToNext < 64) {
    if (verbose) {
      LOG(alarm) << "RDH expected to have memory size and offset to next >= 64 B instead of "
                 << int(rdh.memorySize) << '/' << int(rdh.offsetToNext);
    }
    ok = false;
  }
  if (checkZeros && (rdh.zero0 || rdh.word3 || rdh.zero4 || rdh.word5 || rdh.zero6 || rdh.word7)) {
    if (verbose) {
      LOG(alarm) << "Some reserved fields of RDH v5 are not empty";
    }
    ok = false;
  }
  if (!ok && verbose) {
    dumpRDH(rdh);
  }
  return ok;
}

//_____________________________________________________________________
bool RDHUtils::checkRDH(const RAWDataHeaderV6& rdh, bool verbose, bool checkZeros)
{
  // check if rdh conforms with RDH6 fields
  bool ok = true;
  if (rdh.version != 6) {
    if (verbose) {
      LOG(alarm) << "RDH version 6 is expected instead of " << int(rdh.version);
    }
    ok = false;
  }
  if (rdh.headerSize != 64) {
    if (verbose) {
      LOG(alarm) << "RDH with header size of 64 B is expected instead of " << int(rdh.headerSize);
    }
    ok = false;
  }
  if (rdh.memorySize < 64 || rdh.offsetToNext < 64) {
    if (verbose) {
      LOG(alarm) << "RDH expected to have memory size and offset to next >= 64 B instead of "
                 << int(rdh.memorySize) << '/' << int(rdh.offsetToNext);
    }
    ok = false;
  }
  if (checkZeros && (rdh.zero0 || rdh.word3 || rdh.zero4 || rdh.word5 || rdh.zero6 || rdh.word7)) {
    if (verbose) {
      LOG(alarm) << "Some reserved fields of RDH v6 are not empty";
    }
    ok = false;
  }
  if (!ok && verbose) {
    dumpRDH(rdh);
  }
  return ok;
}

//_____________________________________________________________________
bool RDHUtils::checkRDH(const RAWDataHeaderV7& rdh, bool verbose, bool checkZeros)
{
  // check if rdh conforms with RDH7 fields
  bool ok = true;
  if (rdh.version != 7) {
    if (verbose) {
      LOG(alarm) << "RDH version 7 is expected instead of " << int(rdh.version);
    }
    ok = false;
  }
  if (rdh.headerSize != 64) {
    if (verbose) {
      LOG(alarm) << "RDH with header size of 64 B is expected instead of " << int(rdh.headerSize);
    }
    ok = false;
  }
  if (rdh.memorySize < 64 || rdh.offsetToNext < 64) {
    if (verbose) {
      LOG(alarm) << "RDH expected to have memory size and offset to next >= 64 B instead of "
                 << int(rdh.memorySize) << '/' << int(rdh.offsetToNext);
    }
    ok = false;
  }
  if (checkZeros && (rdh.zero0 || rdh.zero3 || rdh.zero4 || rdh.word5 || rdh.zero6 || rdh.word7)) {
    if (verbose) {
      LOG(alarm) << "Some reserved fields of RDH v7 are not empty";
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

/// process access to non-existing field
void RDHUtils::processError(int v, const char* field)
{
  LOG(alarm) << "Wrong field " << field << " for RDHv" << v;
  throw std::runtime_error("wrong RDH field accessed");
}
