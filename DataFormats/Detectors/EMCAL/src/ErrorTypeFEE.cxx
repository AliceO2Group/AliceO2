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

#include "DataFormatsEMCAL/ErrorTypeFEE.h"
#include <iomanip>
#include <iostream>

using namespace o2::emcal;

void ErrorTypeFEE::PrintStream(std::ostream& stream) const
{
  std::string typestring;
  switch (mErrorSource) {
    case ErrorSource_t::ALTRO_ERROR:
      typestring = "decode error";
      break;
    case ErrorSource_t::MINOR_ALTRO_ERROR:
      typestring = "decode error";
      break;
    case ErrorSource_t::FIT_ERROR:
      typestring = "fit error";
      break;
    case ErrorSource_t::PAGE_ERROR:
      typestring = "page error";
      break;
    case ErrorSource_t::GEOMETRY_ERROR:
      typestring = "geometry error";
      break;
    case ErrorTypeFEE::GAIN_ERROR:
      typestring = "gain type error";
      break;
    case ErrorSource_t::UNDEFINED:
      typestring = "unknown error";
      break;
    default:
      typestring = "unknown error";
      break;
  };
  stream << "EMCAL SM: " << getFEEID() << ", " << typestring << " Type: " << getErrorCode();
  if (mSubspecification >= 0) {
    stream << ", Subspecification: " << mSubspecification;
  }
  if (mHardwareAddress >= 0) {
    stream << ", hardware address 0x" << std::hex << mHardwareAddress << std::dec;
  }
}

std::ostream& operator<<(std::ostream& stream, const ErrorTypeFEE& error)
{
  error.PrintStream(stream);
  return stream;
}
