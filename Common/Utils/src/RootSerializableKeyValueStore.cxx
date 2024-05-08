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

#include "CommonUtils/RootSerializableKeyValueStore.h"
#include <iostream>

using namespace o2::utils;

namespace
{
template <typename T>
std::string stringFromType(char* buffer)
{
  T value;
  std::memcpy(&value, buffer, sizeof(T));
  return std::to_string(value);
}
} // namespace

void RootSerializableKeyValueStore::print(bool includetypeinfo) const
{
  for (auto& p : mStore) {
    const auto& key = p.first;
    const auto info = p.second;
    auto tinfo = info.typeinfo_name;

    std::string value("unknown-value");
    // let's try to decode the value as a string if we can
    if (tinfo == typeid(int).name()) {
      value = stringFromType<int>(info.bufferptr);
    }
    // let's try to decode the value as a string if we can
    else if (tinfo == typeid(unsigned int).name()) {
      value = stringFromType<unsigned int>(info.bufferptr);
    }
    // let's try to decode the value as a string if we can
    else if (tinfo == typeid(short).name()) {
      value = stringFromType<short>(info.bufferptr);
    }
    // let's try to decode the value as a string if we can
    else if (tinfo == typeid(unsigned short).name()) {
      value = stringFromType<unsigned short>(info.bufferptr);
    }
    // let's try to decode the value as a string if we can
    else if (tinfo == typeid(double).name()) {
      value = stringFromType<double>(info.bufferptr);
    }
    // let's try to decode the value as a string if we can
    else if (tinfo == typeid(float).name()) {
      value = stringFromType<float>(info.bufferptr);
    }
    // let's try to decode the value as a string if we can
    else if (tinfo == typeid(std::string).name()) {
      value = *(get<std::string>(key));
    }
    std::cout << "key: " << key << " value: " << value;
    if (includetypeinfo) {
      std::cout << " type: " << info.typeinfo_name;
    }
    std::cout << "\n";
  }
}
