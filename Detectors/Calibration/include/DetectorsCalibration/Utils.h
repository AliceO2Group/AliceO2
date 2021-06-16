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

/// @file   Utils.h
/// @brief  Utils and constants for calibration and related workflows

#ifndef O2_CALIBRATION_CONVENTIONS_H
#define O2_CALIBRATION_CONVENTIONS_H

#include <typeinfo>
#include <utility>
#include <fstream>
#include <TMemFile.h>
#include "Headers/DataHeader.h"
#include "CommonUtils/StringUtils.h"
#include "CCDB/CCDBTimeStampUtils.h"

namespace o2
{
namespace calibration
{

struct Utils {
  static constexpr o2::header::DataOrigin gDataOriginCDBPayload{"CLP"}; // generic DataOrigin for calibrations payload
  static constexpr o2::header::DataOrigin gDataOriginCDBWrapper{"CLW"}; // generic DataOrigin for calibrations wrapper
};

} // namespace calibration
} // namespace o2

#endif
