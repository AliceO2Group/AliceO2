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

#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#include <fmt/format.h>
#include <memory>

using namespace o2::base;

// Filename to store ITSMFT cluster dictionary
std::string DetectorNameConf::getAlpideClusterDictionaryFileName(DetectorNameConf::DId det, const std::string_view prefix, const std::string_view ext)
{
  return buildFileName(prefix, "", det.getName(), ALPIDECLUSDICTFILENAME, ext);
}

// Filename to store detector specific noise maps
std::string DetectorNameConf::getNoiseFileName(DetectorNameConf::DId det, const std::string_view prefix, const std::string_view ext)
{
  return buildFileName(prefix, "", det.getName(), NOISEFILENAME, ext);
}
