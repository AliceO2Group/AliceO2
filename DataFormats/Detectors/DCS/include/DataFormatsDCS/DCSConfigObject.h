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

/// \file DCSConfigObject.h
/// \bried Data format to store DCS configurations

#include <vector>
#include <string>
#include <iostream>
#include <iterator>
#include <regex>

#include <TString.h>

namespace o2
{
namespace dcs
{

typedef std::vector<char> DCSconfigObject_t;

template <typename T>
inline void addConfigItem(DCSconfigObject_t& configVector, std::string key, const T value)
{
  std::string keyValue = key + ":" + std::to_string(value) + ",";
  std::copy(keyValue.begin(), keyValue.end(), std::back_inserter(configVector));
}

// explicit specialization for std::string
template <>
inline void addConfigItem(DCSconfigObject_t& configVector, std::string key, const std::string value)
{
  std::string keyValue = key + ":" + value + ",";
  std::copy(keyValue.begin(), keyValue.end(), std::back_inserter(configVector));
}

// explicit specialization for char
template <>
inline void addConfigItem(DCSconfigObject_t& configVector, std::string key, const char value)
{
  std::string keyValue = key + ":" + value + ",";
  std::copy(keyValue.begin(), keyValue.end(), std::back_inserter(configVector));
}

// explicit specialization for char*
template <>
inline void addConfigItem(DCSconfigObject_t& configVector, std::string key, const char* value)
{
  std::string keyValue = key + ":" + value + ",";
  std::copy(keyValue.begin(), keyValue.end(), std::back_inserter(configVector));
}

// explicit specialization for TString
template <>
inline void addConfigItem(DCSconfigObject_t& configVector, std::string key, const TString value)
{
  std::string keyValue = key + ":" + value.Data() + ",";
  std::copy(keyValue.begin(), keyValue.end(), std::back_inserter(configVector));
}

inline void printDCSConfig(const DCSconfigObject_t& configVector)
{
  std::string sConfig(configVector.begin(), configVector.end());
  std::cout << "string "
            << " --> " << sConfig << std::endl;
  auto const re = std::regex{R"(,+)"};
  auto vecRe = std::vector<std::string>(std::sregex_token_iterator{begin(sConfig), end(sConfig), re, -1},
                                        std::sregex_token_iterator{});
  for (size_t i = 0; i < vecRe.size(); ++i) {
    //      vecRe[i].erase(vecRe[i].end() - 1);
    std::cout << i << " --> " << vecRe[i] << std::endl;
  }
}

} // namespace dcs
} // namespace o2
