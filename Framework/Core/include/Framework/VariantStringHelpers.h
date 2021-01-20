// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_VARIANTSTRINGHELPERS_H
#define FRAMEWORK_VARIANTSTRINGHELPERS_H

#include "Framework/Variant.h"
#include <regex>

namespace o2::framework
{

template <typename T>
T lexical_cast(std::string const& input)
{
  if constexpr (std::is_same_v<T, int>) {
    return std::stoi(input, nullptr);
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return std::stol(input, nullptr);
  } else if constexpr (std::is_same_v<T, float>) {
    return std::stof(input, nullptr);
  } else if constexpr (std::is_same_v<T, double>) {
    return std::stod(input, nullptr);
  } else if constexpr (std::is_same_v<T, bool>) {
    return static_cast<bool>(std::stoi(input, nullptr));
  }
}

template <typename T>
std::vector<T> stringToVector(std::string const& input)
{
  std::vector<T> result;
  //check if the array string has correct array type symbol
  assert(input[0] == variant_array_symbol<T>::symbol);
  std::regex nmatch(R"((?:(?!=,)|(?!=\[))[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?(?=,|\]))");
  auto end = std::sregex_iterator();
  auto values = std::sregex_iterator(input.begin(), input.end(), nmatch);
  for (auto& v = values; v != end; ++v) {
    result.push_back(lexical_cast<T>(v->str()));
  }
  return result;
}

template <>
std::vector<std::string> stringToVector(std::string const& input)
{
  std::vector<std::string> result;
  //check if the array string has correct array type symbol
  assert(input[0] == variant_array_symbol<std::string>::symbol);
  std::regex smatch(R"((?:(?!=,)|(?!=\[))\w+(?=,|\]))");
  auto end = std::sregex_iterator();
  auto values = std::sregex_iterator(input.begin(), input.end(), smatch);
  for (auto v = values; v != end; ++v) {
    result.push_back(v->str());
  }
  return result;
}

template <typename T>
Array2D<T> stringToArray2D(std::string const& input)
{
  std::vector<T> cache;
  assert(input[0] == variant_array_symbol<T>::symbol);
  std::regex mrows(R"(\[[^\[\]]+\])");
  std::regex marray(R"((?:(?!=,)|(?!=\[))[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?(?=,|\]))");
  auto end = std::sregex_iterator();
  auto rows = std::sregex_iterator(input.begin(), input.end(), mrows);
  uint32_t nrows = 0;
  uint32_t ncols = 0;
  bool first = true;
  for (auto& row = rows; row != end; ++row) {
    auto str = row->str();
    auto values = std::sregex_iterator(str.begin(), str.end(), marray);
    if (first) {
      ncols = 0;
    }
    for (auto& v = values; v != end; ++v) {
      cache.push_back(lexical_cast<T>(v->str()));
      if (first) {
        ++ncols;
      }
    }
    if (first) {
      first = false;
    }
    ++nrows;
  }
  return Array2D<T>{cache, nrows, ncols};
}

} // namespace o2::framework

#endif // FRAMEWORK_VARIANTSTRINGHELPERS_H
