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

///
/// @file   RawDataTypes.h
/// @author Jens Wiechula
///

#ifndef AliceO2_TPC_RawDataTypes_H
#define AliceO2_TPC_RawDataTypes_H

#include <unordered_map>
#include <string_view>

namespace o2::tpc::raw_data_types
{
enum Type : char {
  RAWDATA = 0, ///< GBT raw data
  LinkZS = 1,  ///< Link-based Zero Suppression
  ZS = 2,      ///< final Zero Suppression
  IDC = 3,     ///< Integrated Digitial Currents, with priority bit to end up in separate buffer
  SAC = 4,     ///< Sampled Analogue Currents from the current monitor
};

const std::unordered_map<Type, std::string_view> TypeNameMap{
  {Type::RAWDATA, "RAWDATA"},
  {Type::LinkZS, "LinkZS"},
  {Type::ZS, "ZS"},
  {Type::IDC, "IDC"},
  {Type::SAC, "SAC"},
};

} // namespace o2::tpc::raw_data_types

#endif
