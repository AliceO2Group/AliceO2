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
#include <functional>
#include <gsl/span>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#include "MCHRawCommon/RDHFields.h"
#include <cstddef>

namespace o2::mch::raw
{

template <typename RDH>
void assertRDH(const RDH& rdh);

template <typename RDH>
void appendRDH(std::vector<std::byte>& buffer, const RDH& rdh);

template <typename RDH>
RDH createRDH(gsl::span<const std::byte> buffer);

template <typename RDH>
RDH createRDH(uint16_t cruId, uint8_t linkId, uint16_t feeId, uint32_t orbit, uint16_t bunchCrossing, uint16_t payloadSize);

template <typename RDH>
bool isValid(const RDH& rdh);

template <typename RDH>
int countRDHs(gsl::span<const std::byte> buffer);

template <typename RDH>
int showRDHs(gsl::span<const std::byte> buffer);

std::string triggerTypeAsString(uint32_t triggerType);

template <typename RDH>
int forEachRDH(gsl::span<const std::byte> buffer, std::function<void(const RDH&)> f);

template <typename RDH>
int forEachRDH(gsl::span<const std::byte> buffer, std::function<void(const RDH&, gsl::span<const std::byte>::size_type offset)> f);

// beware : this version might modify the RDH, hence the buffer
template <typename RDH>
int forEachRDH(gsl::span<std::byte> buffer, std::function<void(RDH&, gsl::span<std::byte>::size_type offset)> f);

} // namespace o2::mch::raw

#endif
