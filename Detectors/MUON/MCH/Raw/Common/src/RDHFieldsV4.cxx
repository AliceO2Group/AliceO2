// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Headers/RAWDataHeader.h"
#include "MCHRawCommon/RDHFields.h"

namespace o2::mch::raw
{

template <>
uint32_t rdhOrbit(const o2::header::RAWDataHeaderV4& rdh)
{
  return rdh.heartbeatOrbit;
}

template <>
void rdhOrbit(o2::header::RAWDataHeaderV4& rdh, uint32_t orbit)
{
  rdh.heartbeatOrbit = orbit;
}

template <>
uint16_t rdhPayloadSize(const o2::header::RAWDataHeaderV4& rdh)
{
  return rdh.memorySize - sizeof(rdh);
}

template <>
uint8_t rdhLinkId(const o2::header::RAWDataHeaderV4& rdh)
{
  return rdh.linkID + 12 * rdh.endPointID;
}

template <>
void rdhLinkId(o2::header::RAWDataHeaderV4& rdh, uint8_t linkId)
{
  rdh.linkID = linkId % 12;
  rdh.endPointID = linkId / 12;
}

template <>
uint16_t rdhCruId(const o2::header::RAWDataHeaderV4& rdh)
{
  return rdh.cruID;
}

template <>
void rdhCruId(o2::header::RAWDataHeaderV4& rdh, uint16_t cruId)
{
  rdh.cruID = cruId;
}

template <>
uint16_t rdhBunchCrossing(const o2::header::RAWDataHeaderV4& rdh)
{
  return static_cast<uint16_t>(rdh.heartbeatBC & 0xFFF);
}

template <>
void rdhBunchCrossing(o2::header::RAWDataHeaderV4& rdh, uint16_t bc)
{
  rdh.heartbeatBC = bc;
}

template <>
uint8_t rdhPacketCounter(const o2::header::RAWDataHeaderV4& rdh)
{
  return rdh.packetCounter;
}

template <>
void rdhPacketCounter(o2::header::RAWDataHeaderV4& rdh, uint8_t count)
{
  rdh.packetCounter = count;
}

template <>
uint16_t rdhFeeId(const o2::header::RAWDataHeaderV4& rdh)
{
  return rdh.feeId;
}

template <>
void rdhFeeId(o2::header::RAWDataHeaderV4& rdh, uint16_t feeId)
{
  rdh.feeId = feeId;
}

template <>
uint16_t rdhPageCounter(const o2::header::RAWDataHeaderV4& rdh)
{
  return rdh.pageCnt;
}

template <>
void rdhPageCounter(o2::header::RAWDataHeaderV4& rdh, uint16_t pageCnt)
{
  rdh.pageCnt = pageCnt;
}

template <>
uint16_t rdhOffsetToNext(const o2::header::RAWDataHeaderV4& rdh)
{
  return rdh.offsetToNext;
}

template <>
void rdhOffsetToNext(o2::header::RAWDataHeaderV4& rdh, uint16_t offset)
{
  rdh.offsetToNext = offset;
}

template <>
uint16_t rdhMemorySize(const o2::header::RAWDataHeaderV4& rdh)
{
  return rdh.memorySize;
}

template <>
void rdhMemorySize(o2::header::RAWDataHeaderV4& rdh, uint16_t memorySize)
{
  rdh.memorySize = memorySize;
}

template <>
void rdhStop(o2::header::RAWDataHeaderV4& rdh, uint8_t stop)
{
  rdh.stop = stop;
}

template <>
uint8_t rdhStop(const o2::header::RAWDataHeaderV4& rdh)
{
  return rdh.stop;
}

template <>
uint32_t rdhTriggerType(const o2::header::RAWDataHeaderV4& rdh)
{
  return rdh.triggerType;
}

template <>
void rdhTriggerType(o2::header::RAWDataHeaderV4& rdh, uint32_t triggerType)
{
  rdh.triggerType = triggerType;
}

template <>
uint8_t rdhEndpoint(const o2::header::RAWDataHeaderV4& rdh)
{
  return rdh.endPointID;
}
} // namespace o2::mch::raw
