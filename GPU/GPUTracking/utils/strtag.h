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

/// \file strtag.h
/// \author David Rohr

#ifndef STRTAG_H
#define STRTAG_H

#include <cstring>
#include <cstdint>
#include <string>
#include <type_traits>

template <class T = uint64_t, std::size_t N>
constexpr T qStr2Tag(const char (&str)[N])
{
  static_assert(std::is_trivially_copyable_v<T>);
  static_assert(N - 1 == sizeof(T), "Invalid tag length");
  T value{};
  for (std::size_t i = 0; i < sizeof(T); ++i) {
    value |= T(static_cast<unsigned char>(str[i])) << (i * 8);
  }
  return value;
}

template <class T>
std::string qTag2Str(const T tag)
{
  T str[2];
  str[0] = tag;
  str[1] = 0;
  return std::string((const char*)str);
}

#endif
