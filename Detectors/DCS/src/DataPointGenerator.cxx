// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DetectorsDCS/AliasExpander.h"
#include "DetectorsDCS/DataPointGenerator.h"
#include "DetectorsDCS/DataPointCreator.h"
#include "DetectorsDCS/DataPointCompositeObject.h"
#include "DetectorsDCS/StringUtils.h"
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
    uint32_t seconds = mktime(t);
  } else {
    std::tm t{};
    std::istringstream ss(refDate);
    ss >> std::get_time(&t, "%Y-%b-%d %H:%M:%S");
    seconds = mktime(&t);
  }
  uint16_t msec = 5;
  return std::make_pair(seconds, msec);
}

} // namespace

namespace o2::dcs
{

//std::enable_if_t<std::is_arithmetic<T>::value, bool> = true>

template <typename T>
std::vector<o2::dcs::DataPointCompositeObject>
  generateRandomDataPoints(const std::vector<std::string>& aliases,
                           T minValue, T maxValue, std::string refDate)
{
  std::vector<o2::dcs::DataPointCompositeObject> dpcoms;
  static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
  typedef typename std::conditional<std::is_integral<T>::value,
                                    std::uniform_int_distribution<T>,
                                    std::uniform_real_distribution<T>>::type distType;

  std::random_device rd;
  std::mt19937 mt(rd());
  distType dist{minValue, maxValue};
  auto [seconds, msec] = getDate(refDate);
  for (auto alias : expandAliases(aliases)) {
    auto value = dist(mt);
    dpcoms.emplace_back(o2::dcs::createDataPointCompositeObject(alias, value, seconds, msec));
  }
  return dpcoms;
}

// only specialize the functions for the types we support :
//
// - double
// - uint32_t
// - int32_t
// - char
// - bool
//
// - std::string

template std::vector<o2::dcs::DataPointCompositeObject> generateRandomDataPoints<double>(const std::vector<std::string>& aliases, double minValue, double maxValue, std::string);

template std::vector<o2::dcs::DataPointCompositeObject> generateRandomDataPoints<uint32_t>(const std::vector<std::string>& aliases, uint32_t minValue, uint32_t maxValue, std::string);

template std::vector<o2::dcs::DataPointCompositeObject> generateRandomDataPoints<int32_t>(const std::vector<std::string>& aliases, int32_t minValue, int32_t maxValue, std::string);

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
    auto value = o2::dcs::random_string2(dist(mt));
    dpcoms.emplace_back(o2::dcs::createDataPointCompositeObject(alias, value, seconds, msec));
  }
  return dpcoms;
}
} // namespace o2::dcs
