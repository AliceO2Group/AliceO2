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

#ifndef O2_MCH_DIGITFILTERING_DIGITFILTER_H_
#define O2_MCH_DIGITFILTERING_DIGITFILTER_H_

#include "DataFormatsMCH/Digit.h"
#include "MCHStatus/StatusMap.h"
#include <functional>

namespace o2::mch
{
typedef std::function<bool(const Digit&)> DigitFilter;

DigitFilter createDigitFilter(uint32_t minADC,
                              bool rejectBackground,
                              bool selectSignal,
                              const StatusMap& statusMap = {},
                              uint32_t statusMask = 0);

} // namespace o2::mch

#endif
