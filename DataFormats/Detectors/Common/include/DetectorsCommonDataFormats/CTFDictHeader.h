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

/// \file CTFDictHeader.h
/// \brief Header: timestamps and format version for detector CTF dictionary
/// \author ruben.shahoyan@cern.ch

#ifndef _ALICEO2_CTFDICTHEADER_H
#define _ALICEO2_CTFDICTHEADER_H

#include <Rtypes.h>
#include <string>
#include "DetectorsCommonDataFormats/DetID.h"

namespace o2
{
namespace ctf
{

/// Detector header base
struct CTFDictHeader {
  o2::detectors::DetID det{};
  uint32_t dictTimeStamp = 0; // dictionary creation time (seconds since epoch) / hash
  uint8_t majorVersion = 1;
  uint8_t minorVersion = 0;

  bool isValidDictTimeStamp() const { return dictTimeStamp != 0; }
  bool operator==(const CTFDictHeader& o) const
  {
    return dictTimeStamp == o.dictTimeStamp && majorVersion == o.majorVersion && minorVersion == o.minorVersion;
  }
  bool operator!=(const CTFDictHeader& o) const
  {
    return dictTimeStamp != o.dictTimeStamp || majorVersion != o.majorVersion || minorVersion != o.minorVersion;
  }
  std::string asString() const;

  ClassDefNV(CTFDictHeader, 2);
};

} // namespace ctf
} // namespace o2

#endif
