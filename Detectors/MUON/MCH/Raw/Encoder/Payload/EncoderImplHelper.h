// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_ENCODER_IMPL_HELPER_H
#define O2_MCH_RAW_ENCODER_IMPL_HELPER_H

#include "MCHRawCommon/SampaCluster.h"
#include "MCHRawCommon/SampaHeader.h"
#include <cstdlib>
#include <gsl/span>
#include <vector>

namespace o2::mch::raw

{

namespace impl
{
void append(std::vector<uint10_t>& b10, uint50_t value);

SampaHeader buildSampaHeader(uint8_t elinkId, uint8_t chId, gsl::span<const SampaCluster> data);

void fillUserLogicBuffer10(std::vector<uint10_t>& b10,
                           gsl::span<const SampaCluster> clusters,
                           uint8_t elinkId,
                           uint8_t chId,
                           bool addSync);

void b10to64(std::vector<uint10_t> b10, std::vector<uint64_t>& b64, uint16_t prefix14);

} // namespace impl
} // namespace o2::mch::raw

#endif
