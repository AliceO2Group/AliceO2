// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_RDH_MANIP_H
#define O2_MCH_RAW_RDH_MANIP_H

#include <cstdint>
#include <iostream>
#include <vector>
#include <gsl/span>
#include <string_view>
#include <functional>

namespace o2
{
namespace mch
{
namespace raw
{

template <typename RDH>
void assertRDH(const RDH& rdh);

template <typename RDH>
void appendRDH(std::vector<uint8_t>& buffer, const RDH& rdh);

template <typename RDH>
void appendRDH(std::vector<uint32_t>& buffer, const RDH& rdh);

template <typename RDH>
RDH createRDH(gsl::span<uint8_t> buffer);

template <typename RDH>
RDH createRDH(gsl::span<uint32_t> buffer);

template <typename RDH>
RDH createRDH(uint16_t cruId, uint8_t linkId, uint16_t solarId, uint32_t orbit, uint16_t bunchCrossing, uint16_t payloadSize);

template <typename RDH>
bool isValid(const RDH& rdh);

template <typename RDH>
size_t rdhPayloadSize(const RDH& rdh);

template <typename RDH>
uint8_t rdhLinkId(const RDH& rdh);

template <typename RDH>
uint32_t rdhOrbit(const RDH& rdh);

template <typename RDH>
uint16_t rdhBunchCrossing(const RDH& rdh);

template <typename RDH>
int countRDHs(gsl::span<uint8_t> buffer);

template <typename RDH>
int showRDHs(gsl::span<uint32_t> buffer);

template <typename RDH>
int showRDHs(gsl::span<uint8_t> buffer);

void dumpRDHBuffer(gsl::span<uint32_t> buffer, std::string_view indent);

template <typename RDH>
int forEachRDH(gsl::span<uint32_t> buffer, std::function<void(RDH&)> f);

template <typename RDH>
int countRDHs(gsl::span<uint32_t> buffer);

template <typename RDH>
int countRDHs(gsl::span<uint8_t> buffer);

} // namespace raw
} // namespace mch
} // namespace o2

#endif
