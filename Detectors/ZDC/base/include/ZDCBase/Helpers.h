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

/// @file   Helper.h
/// @brief  ZDC helper functions
/// @author pietro.cortese@cern.ch

#include <string>

#ifndef ALICEO2_ZDC_HELPERS_H
#define ALICEO2_ZDC_HELPERS_H

namespace o2
{
namespace zdc
{
namespace helpers
{

std::string removeNamespace(const std::string& strin);
bool endsWith(const std::string& str, const std::string& suffix);
std::string ccdbShortcuts(std::string ccdbHost, std::string cln, std::string path);

} // namespace helpers
} // namespace zdc
} // namespace o2

#endif
