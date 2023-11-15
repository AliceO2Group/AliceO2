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

#include "DetectorsDCS/AliasExpander.h"
#include "DetectorsDCS/DataPointGenerator.h"
#include "DetectorsDCS/DataPointCreator.h"
#include "DetectorsDCS/DataPointCompositeObject.h"
#include "DetectorsDCS/StringUtils.h"
#include "Framework/Logger.h"
#include <fmt/format.h>
#include <random>
#include <utility>
#include <type_traits>
#include <cstdint>

namespace
{
std::pair<uint32_t, uint16_t> getDate(const std::string& refDate)
{

  uint32_t seconds;
  if (refDate.empty()) {
    auto current = std::time(nullptr);
    auto t = std::localtime(&current);
    seconds = mktime(t);
  } else {
    std::tm t{};
    std::istringstream ss(refDate);
    ss >> std::get_time(&t, "%Y-%b-%d %H:%M:%S");
    if (ss.fail()) { // let's see if it was passed as a TDatime, as SQL string
      std::tm tt{};
      std::istringstream sss(refDate);
      sss >> std::get_time(&tt, "%Y-%m-%d %H:%M:%S");
      if (sss.fail()) {
        std::tm ttt{};
        std::istringstream ssss(refDate);
        ssss >> std::get_time(&tt, "%Y-%B-%d %H:%M:%S");
        if (ssss.fail()) {
          LOG(error) << "We cannot parse the date";
        }
        seconds = mktime(&ttt);
      } else {
        seconds = mktime(&tt);
      }
    } else {
      seconds = mktime(&t);
    }
  }
  uint16_t msec = 5;
  return std::make_pair(seconds, msec);
}

} // namespace

namespace o2::dcs
{

// std::enable_if_t<std::is_arithmetic<T>::value, bool> = true>

template <typename T>
std::vector<o2::dcs::DataPointCompositeObject>
  generateRandomDataPoints(const std::vector<std::string>& aliases,
                           T minValue, T maxValue, std::string refDate)
{
  std::vector<o2::dcs::DataPointCompositeObject> dpcoms;
  static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
  using distType = std::conditional_t<std::is_integral<T>::value,
                                      std::uniform_int_distribution<long long>,
                                      std::uniform_real_distribution<T>>;
  std::random_device rd;
  std::mt19937 mt(rd());
  distType dist{minValue, maxValue};
  auto [seconds, msec] = getDate(refDate);
  for (auto alias : expandAliases(aliases)) {
    T value = dist(mt);
    dpcoms.emplace_back(o2::dcs::createDataPointCompositeObject(alias, value, seconds, msec));
  }
  return dpcoms;
}

// only specialize the functions for the types we support :
//
// - double
// - float
// - uint32_t
// - int32_t
// - char
// - bool
//
// - std::string

template std::vector<o2::dcs::DataPointCompositeObject> generateRandomDataPoints<double>(const std::vector<std::string>& aliases, double minValue, double maxValue, std::string);

template std::vector<o2::dcs::DataPointCompositeObject> generateRandomDataPoints<float>(const std::vector<std::string>& aliases, float minValue, float maxValue, std::string);

template std::vector<o2::dcs::DataPointCompositeObject> generateRandomDataPoints<uint32_t>(const std::vector<std::string>& aliases, uint32_t minValue, uint32_t maxValue, std::string);

template std::vector<o2::dcs::DataPointCompositeObject> generateRandomDataPoints<int32_t>(const std::vector<std::string>& aliases, int32_t minValue, int32_t maxValue, std::string);

template std::vector<o2::dcs::DataPointCompositeObject> generateRandomDataPoints<long long>(const std::vector<std::string>& aliases, long long minValue, long long maxValue, std::string);

template std::vector<o2::dcs::DataPointCompositeObject> generateRandomDataPoints<char>(const std::vector<std::string>& aliases, char minValue, char maxValue, std::string);

/** Need a specific specialization for bool as got into trouble compiling uniform_int_distribution<bool>
 * on some platform (e.g. CC7).
 */
template <>
std::vector<o2::dcs::DataPointCompositeObject> generateRandomDataPoints<bool>(const std::vector<std::string>& aliases, bool minValue, bool maxValue, std::string refDate)
{
  std::vector<o2::dcs::DataPointCompositeObject> dpcoms;
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution dist{0, 1};
  auto [seconds, msec] = getDate(refDate);
  for (auto alias : expandAliases(aliases)) {
    bool value = dist(mt);
    dpcoms.emplace_back(o2::dcs::createDataPointCompositeObject(alias, value, seconds, msec));
  }
  return dpcoms;
}

/**
 * Generate data points of type string, where each string is random, with
 * a length between the length of the two input strings (minLength,maxLength)
 */
template <>
std::vector<o2::dcs::DataPointCompositeObject> generateRandomDataPoints<std::string>(const std::vector<std::string>& aliases, std::string minLength, std::string maxLength, std::string refDate)
{
  std::vector<o2::dcs::DataPointCompositeObject> dpcoms;
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::string::size_type> dist{minLength.size(), maxLength.size()};
  auto [seconds, msec] = getDate(refDate);
  for (auto alias : expandAliases(aliases)) {
    std::string value = o2::dcs::random_string2(dist(mt));
    dpcoms.emplace_back(o2::dcs::createDataPointCompositeObject(alias, value, seconds, msec));
  }
  return dpcoms;
}
} // namespace o2::dcs
