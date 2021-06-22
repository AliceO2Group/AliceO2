// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_ENCODER_DIGIT_TO_ELECMAPPER_H
#define O2_MCH_RAW_ENCODER_DIGIT_TO_ELECMAPPER_H

#include "MCHRawElecMap/Mapper.h"
#include "MCHRawElecMap/DsDetId.h"
#include "MCHRawElecMap/DsElecId.h"
#include "DataFormatsMCH/Digit.h"
#include <cstdint>
#include <functional>
#include <optional>
#include <utility>

namespace o2::mch::raw
{

using Digit2ElecMapper = std::function<std::optional<std::pair<DsElecId, int>>(const o2::mch::Digit& digit)>;

Digit2ElecMapper createDigit2ElecMapper(Det2ElecMapper d2e);
} // namespace o2::mch::raw

#endif
