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

#ifndef ALICEO2_FIT_DCSDPVALUES_H_
#define ALICEO2_FIT_DCSDPVALUES_H_

#include <Rtypes.h>
#include "Framework/Logger.h"

namespace o2
{
namespace fit
{
struct DCSDPValues {
  std::vector<std::pair<uint64_t, int>> values;

  DCSDPValues()
  {
    values = std::vector<std::pair<uint64_t, int>>();
  }

  void add(uint64_t timestamp, int value)
  {
    values.push_back(std::pair<uint64_t, int>(timestamp, value));
  }

  bool empty()
  {
    return values.empty();
  }

  void makeEmpty()
  {
    values.clear();
  }

  void print(const bool verbose = false) const
  {
    LOG(info) << values.size() << " value(s)";
    if (verbose && !values.empty()) {
      for (auto& val : values) {
        LOG(info) << "timestamp = " << val.first << ", value = " << val.second;
      }
    } else if (!values.empty()) {
      LOG(info) << "First value: "
                << "timestamp = " << values.front().first << ", value = " << values.front().second;
      LOG(info) << "Last value: "
                << "timestamp = " << values.back().first << ", value = " << values.back().second;
    }
  }

  ClassDefNV(DCSDPValues, 2);
};

} // namespace fit
} // namespace o2

#endif