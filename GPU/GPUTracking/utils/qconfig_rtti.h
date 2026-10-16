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

/// \file qconfig_rtti.h
/// \author David Rohr

#ifndef QCONFIG_RTTI_H
#define QCONFIG_RTTI_H

#include "qconfig.h"
#include "qconfig_helpers.h"
#include <string>

namespace qConfig
{
struct qConfigRTTIentry {
  std::string name = "";          // Name of option
  std::string type = "";          // Data type of option
  uint8_t size = 0;               // sizeof(type)
  bool isRTC = false;             // enable RTC constexpr optimization
  bool isVariable = false;        // is a variable, not settable at command line
  bool isVector = false;          // is std::vector
  uint16_t arraySize = 0;         // size of array if std::array, 0 = no array
  uint16_t configStructIndex = 0; // Index in configStructVector
  size_t offset = 0;              // Offset address in configStruct
  std::string configStruct = "";  // Name of config struct that stores the option
  std::string max = "";           // Maximum Value
  std::string min = "";           // Minimum Value
  std::string defaultValue = "";  // Default Value
  std::string description = "";   // Description of option
  std::string message = "";       // printf format string, printed with the value when the option is set
};
} // namespace qConfig

#endif // QCONFIG_RTTI_H
