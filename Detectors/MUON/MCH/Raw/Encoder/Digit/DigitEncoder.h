// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_ENCODER_DIGIT_ENCODER_H
#define O2_MCH_RAW_ENCODER_DIGIT_ENCODER_H

#include "MCHBase/Digit.h"
#include "MCHRawElecMap/DsDetId.h"
#include "MCHRawElecMap/DsElecId.h"
#include "MCHRawElecMap/Mapper.h"
#include <cstdint>
#include <functional>
#include <gsl/span>
#include <optional>

namespace o2::mch::raw
{

using DigitEncoder = std::function<void(gsl::span<o2::mch::Digit> digits,
                                        std::vector<std::byte>& buffer,
                                        uint32_t orbit,
                                        uint16_t bc)>;

DigitEncoder createDigitEncoder(bool userLogic, Det2ElecMapper det2elec);

using Digit2ElecMapper = std::function<std::optional<std::pair<DsElecId, int>>(const o2::mch::Digit& digit)>;

Digit2ElecMapper createDigit2ElecMapper(Det2ElecMapper);

} // namespace o2::mch::raw
#endif
