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

/// \file qconfig_general.h
/// \author David Rohr

#ifndef QCONFIG_H_GENERAL
#define QCONFIG_H_GENERAL
#ifndef GPUCA_GPUCODE_DEVICE
#include <cstdint>
#include <string>
#endif

namespace qConfig
{
#ifndef GPUCA_GPUCODE_DEVICE
int32_t qConfigParse(int argc, const char** argv, const char* filename = nullptr);
void qConfigPrint(); // Print all settings
template <class T>
void qConfigPrint(const T& qconfig_tmp_object, const std::string& blockName = ""); // Print one sub config, specializations created in qconfig.cxx
#endif
enum qConfigRetVal { qcrOK = 0,
                     qcrError = 1,
                     qcrMinFailure = 2,
                     qcrMaxFailure = 3,
                     qcrHelp = 4,
                     qcrCmd = 5,
                     qcrArgMissing = 6,
                     qcrArgIncomplete = 7,
                     qcrArrayOverflow = 8 };
} // namespace qConfig
#endif // QCONFIG_H_GENERAL
