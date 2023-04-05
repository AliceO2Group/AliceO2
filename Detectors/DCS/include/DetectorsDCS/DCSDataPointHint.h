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

#ifndef O2_DCS_TEST_WORKFLOW_DATAPOINTHINT_H
#define O2_DCS_TEST_WORKFLOW_DATAPOINTHINT_H

#include <variant>
#include <string>
#include <vector>
#include <cstdint>

namespace o2::dcs::test
{
/*
 * A compact representation a group of alias to be generated
 */
template <typename T>
struct DataPointHint {
  std::string aliasPattern; // alias pattern e.g. DET/HV/Crate[0..2]/Channel[000..012]/vMon
  T minValue;               // minimum value to generate
  T maxValue;               // maximum value to generate
};

using HintType = std::variant<DataPointHint<double>,
                              DataPointHint<float>,
                              DataPointHint<uint32_t>,
                              DataPointHint<int32_t>,
                              DataPointHint<short>,
                              DataPointHint<bool>,
                              DataPointHint<std::string>>;

} // namespace o2::dcs::test

#endif
