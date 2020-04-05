// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_RDH_FIELDS_H
#define O2_MCH_RAW_RDH_FIELDS_H

#include <cstdint>

namespace o2
{
namespace mch
{
namespace raw
{
template <typename RDH>
uint16_t rdhPayloadSize(const RDH& rdh);

template <typename RDH>
uint16_t rdhCruId(const RDH& rdh);

template <typename RDH>
void rdhCruId(RDH& rdh, uint16_t cruId);

template <typename RDH>
uint8_t rdhLinkId(const RDH& rdh);

template <typename RDH>
void rdhLinkId(RDH& rdh, uint8_t linkId);

template <typename RDH>
uint32_t rdhOrbit(const RDH& rdh);

template <typename RDH>
void rdhOrbit(RDH& rdh, uint32_t orbit);

template <typename RDH>
uint16_t rdhBunchCrossing(const RDH& rdh);

template <typename RDH>
void rdhBunchCrossing(RDH& rdh, uint16_t bc);

template <typename RDH>
uint16_t rdhFeeId(const RDH& rdh);

template <typename RDH>
void rdhFeeId(RDH& rdh, uint16_t feeId);

template <typename RDH>
uint16_t rdhPageCounter(const RDH& rdh);

template <typename RDH>
void rdhPageCounter(RDH& rdh, uint16_t pageCnt);

template <typename RDH>
uint8_t rdhPacketCounter(const RDH& rhd);

template <typename RDH>
void rdhPacketCounter(RDH& rdh, uint8_t count);

template <typename RDH>
uint16_t rdhOffsetToNext(const RDH& rdh);

template <typename RDH>
void rdhOffsetToNext(RDH& rdh, uint16_t offset);

template <typename RDH>
uint16_t rdhMemorySize(const RDH& rdh);

template <typename RDH>
void rdhMemorySize(RDH& rdh, uint16_t memorySize);

template <typename RDH>
void rdhStop(RDH& rdh, uint8_t stop);

template <typename RDH>
uint8_t rdhStop(const RDH& rdh);

template <typename RDH>
uint32_t rdhTriggerType(const RDH& rdh);

template <typename RDH>
void rdhTriggerType(RDH& rdh, uint32_t triggerType);

template <typename RDH>
uint8_t rdhEndpoint(const RDH& rdh);
} // namespace raw
} // namespace mch
} // namespace o2

#endif
