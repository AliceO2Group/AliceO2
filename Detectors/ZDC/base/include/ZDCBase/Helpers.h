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

/// @file   Helpers.h
/// @brief  ZDC helpers functions
/// @author pietro.cortese@cern.ch

#include <string>
#include <algorithm>

#ifndef ALICEO2_ZDC_HELPERS_H
#define ALICEO2_ZDC_HELPERS_H

namespace o2
{
namespace zdc
{

// std::string removeNamespace(const std::string& strin);
// bool endsWith(const std::string& str, const std::string& suffix);
// std::string ccdbShortcuts(std::string ccdbHost, std::string cln, std::string path);

//______________________________________________________________________________
std::string removeNamespace(const std::string& strin)
{
  std::string str = strin;
  for (auto pos = str.find(":"); pos != std::string::npos; pos = str.find(":")) {
    str = str.substr(pos + 1);
  }
  return str;
}

//______________________________________________________________________________
bool endsWith(const std::string& str, const std::string& suffix)
{
  return str.size() >= suffix.size() && 0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

//______________________________________________________________________________
std::string ccdbShortcuts(std::string ccdbHost, std::string cln, std::string path)
{
  // Commonly used shortcuts for ccdbHost
  if (ccdbHost.size() == 0 || ccdbHost == "external") {
    ccdbHost = "http://alice-ccdb.cern.ch:8080";
  } else if (ccdbHost == "internal") {
    ccdbHost = "http://o2-ccdb.internal/";
  } else if (ccdbHost == "test") {
    ccdbHost = "http://ccdb-test.cern.ch:8080";
  } else if (ccdbHost == "local") {
    ccdbHost = "http://localhost:8080";
  } else if (ccdbHost == "root") {
    std::replace(path.begin(), path.end(), '/', '_');
    ccdbHost = path + "_" + removeNamespace(cln) + ".root";
  }
  return ccdbHost;
}

} // namespace zdc
} // namespace o2

#endif
