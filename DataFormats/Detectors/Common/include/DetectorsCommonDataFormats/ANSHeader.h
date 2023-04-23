// Copyright 2019-2023 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ANSHeader.h
/// \author michael.lettrich@cern.ch
/// \brief representation of ANS Version number in a comparable way

#ifndef _ALICEO2_ANSHEADER_H_
#define _ALICEO2_ANSHEADER_H_

#include <cstdint>
#include <string>
#include <Rtypes.h>
#include <fmt/format.h>

namespace o2::ctf
{

struct ANSHeader {
  uint8_t majorVersion;
  uint8_t minorVersion;

  void clear() { majorVersion = minorVersion = 0; }
  inline constexpr operator uint16_t() const noexcept
  {
    uint16_t major = majorVersion;
    uint16_t minor = minorVersion;
    return (major << 8) | minor;
  };

  inline operator std::string() const
  {
    return fmt::format("{}.{}", majorVersion, minorVersion);
  };
  inline constexpr uint32_t version() const noexcept { return static_cast<uint16_t>(*this); };
  inline constexpr bool operator==(const ANSHeader& other) const noexcept { return static_cast<uint16_t>(*this) == static_cast<uint16_t>(other); };
  inline constexpr bool operator!=(const ANSHeader& other) const noexcept { return static_cast<uint16_t>(*this) != static_cast<uint16_t>(other); };
  inline constexpr bool operator<(const ANSHeader& other) const noexcept { return static_cast<uint16_t>(*this) < static_cast<uint16_t>(other); };
  inline constexpr bool operator>(const ANSHeader& other) const noexcept { return static_cast<uint16_t>(*this) > static_cast<uint16_t>(other); };
  inline constexpr bool operator>=(const ANSHeader& other) const noexcept { return static_cast<uint16_t>(*this) >= static_cast<uint16_t>(other); };
  inline constexpr bool operator<=(const ANSHeader& other) const noexcept { return static_cast<uint16_t>(*this) <= static_cast<uint16_t>(other); };
  ClassDefNV(ANSHeader, 2);
};

inline constexpr ANSHeader ANSVersionUnspecified{0, 0};
inline constexpr ANSHeader ANSVersionCompat{0, 1};
inline constexpr ANSHeader ANSVersion1{1, 0};

inline ANSHeader ansVersionFromString(const std::string& ansVersionString)
{
  if (ansVersionString == "0.1" || ansVersionString == "compat") {
    return ctf::ANSVersionCompat;
  } else if (ansVersionString == "1.0") {
    return ctf::ANSVersion1;
  } else {
    return ctf::ANSVersionUnspecified;
  }
}

} // namespace o2::ctf

#endif /* _ALICEO2_ANSHEADER_H_ */
