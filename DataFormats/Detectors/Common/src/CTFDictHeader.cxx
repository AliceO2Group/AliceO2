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

/// \file CTFDictHeader.cxx
/// \brief Header: timestamps and format version for detector CTF dictionary
/// \author ruben.shahoyan@cern.ch

#include "DetectorsCommonDataFormats/CTFDictHeader.h"
#include <ctime>
#include <sstream>
#include <iomanip>

using namespace o2::ctf;

std::string CTFDictHeader::asString() const
{
  std::time_t temp = dictTimeStamp;
  std::tm* t = std::gmtime(&temp);
  std::stringstream ss;
  ss << "CTF Dict for " << det.getName() << ", v" << int(majorVersion) << '.' << int(minorVersion) << " from " << std::put_time(t, "%Y-%m-%d %I:%M:%S %p");
  return ss.str();
}
