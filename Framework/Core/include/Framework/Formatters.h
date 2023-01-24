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

#ifndef O2_FRAMEWORK_DPLFORMATTERS_H_
#define O2_FRAMEWORK_DPLFORMATTERS_H_

#include <fmt/format.h>
#include "Framework/Lifetime.h"

template <>
struct fmt::formatter<o2::framework::Lifetime> : fmt::formatter<std::string_view> {
  char presentation = 's';

  template <typename FormatContext>
  auto format(o2::framework::Lifetime const& h, FormatContext& ctx)
  {
    std::string_view s = "unknown";
    switch (h) {
      case o2::framework::Lifetime::Timeframe:
        s = "timeframe";
        break;
      case o2::framework::Lifetime::Condition:
        s = "condition";
        break;
      case o2::framework::Lifetime::QA:
        s = "qos";
        break;
      case o2::framework::Lifetime::Transient:
        s = "transient";
        break;
        // Complete the rest of the enum
      case o2::framework::Lifetime::Timer:
        s = "timer";
        break;
      case o2::framework::Lifetime::Enumeration:
        s = "enumeration";
        break;
      case o2::framework::Lifetime::Signal:
        s = "signal";
        break;
      case o2::framework::Lifetime::Optional:
        s = "optional";
        break;
      case o2::framework::Lifetime::OutOfBand:
        s = "out-of-band";
        break;
    };
    return formatter<string_view>::format(s, ctx);
  }
};

#endif // O2_FRAMEWORK_DPLFORMATTERS_H_
