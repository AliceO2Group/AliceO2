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

#ifndef O2_BASE_DATA_HEADER_HELPERS_
#define O2_BASE_DATA_HEADER_HELPERS_

#include "Headers/DataHeader.h"
#include <fmt/format.h>

template <typename T>
struct fmt::formatter<T, std::enable_if_t<o2::header::is_descriptor<T>::value, char>> {
  // Presentation format: 'f' - fixed, 'e' - exponential.
  char presentation = 's';

  // Parses format specifications of the form ['f' | 'e'].
  constexpr auto parse(format_parse_context& ctx)
  {
    auto it = ctx.begin(), end = ctx.end();
    if (it != end && (*it == 's')) {
      presentation = *it++;
    }

    // Check if reached the end of the range:
    if (it != end && *it != '}') {
      throw format_error("invalid pick format");
    }

    // Return an iterator past the end of the parsed range:
    return it;
  }

  template <typename FormatContext>
  auto format(const T& p, FormatContext& ctx)
  {
    return fmt::format_to(ctx.out(), "{}", p.template as<std::string>());
  }
};

template <>
struct fmt::formatter<o2::header::DataHeader> {
  // Presentation format: 'f' - fixed, 'e' - exponential.
  char presentation = 's';

  // Parses format specifications of the form ['f' | 'e'].
  constexpr auto parse(format_parse_context& ctx)
  {
    auto it = ctx.begin(), end = ctx.end();
    if (it != end && (*it == 's')) {
      presentation = *it++;
    }

    // Check if reached the end of the range:
    if (it != end && *it != '}') {
      throw format_error("invalid format");
    }

    // Return an iterator past the end of the parsed range:
    return it;
  }

  template <typename FormatContext>
  auto format(const o2::header::DataHeader& h, FormatContext& ctx)
  {
    auto res = fmt::format("Data header version {}, flags: {}\n", h.headerVersion, h.flags) +
               fmt::format("  origin       : {}\n", h.dataOrigin.str) +
               fmt::format("  serialization: {}\n", h.payloadSerializationMethod.str) +
               fmt::format("  description  : {}\n", h.dataDescription.str) +
               fmt::format("  sub spec.    : {}\n", (long long unsigned int)h.subSpecification) +
               fmt::format("  header size  : {}\n", h.headerSize) +
               fmt::format("  payloadSize  : {}\n", (long long unsigned int)h.payloadSize) +
               fmt::format("  firstTForbit : {}\n", h.firstTForbit) +
               fmt::format("  tfCounter    : {}\n", h.tfCounter) +
               fmt::format("  runNumber    : {}\n", h.runNumber);
    return fmt::format_to(ctx.out(), "{}", res);
  }
};

#endif // O2_BASE_DATA_HEADER_HELPERS_
