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
    case ErrorSource_t::GAIN_ERROR:
      typestring = "gain type error";
      break;
    case ErrorSource_t::STU_ERROR:
      typestring = "STU decoder error";
      break;
    case ErrorSource_t::LINK_ERROR:
      typestring = "Link missing error";
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

const char* ErrorTypeFEE::getErrorTypeName(unsigned int errorTypeID)
{
  switch (errorTypeID) {
    case ErrorSource_t::PAGE_ERROR:
      return "Page";
    case ErrorSource_t::ALTRO_ERROR:
      return "MajorAltro";
    case ErrorSource_t::MINOR_ALTRO_ERROR:
      return "MinorAltro";
    case ErrorSource_t::FIT_ERROR:
      return "Fit";
    case ErrorSource_t::GEOMETRY_ERROR:
      return "Geometry";
    case ErrorSource_t::GAIN_ERROR:
      return "GainType";
    case ErrorSource_t::STU_ERROR:
      return "STUDecoding";
    case ErrorSource_t::LINK_ERROR:
      return "LinkMissing";
    case ErrorSource_t::UNDEFINED:
      return "Undefined";
    default:
      return "";
  };
}

const char* ErrorTypeFEE::getErrorTypeTitle(unsigned int errorTypeID)
{
  switch (errorTypeID) {
    case ErrorSource_t::PAGE_ERROR:
      return "Page";
    case ErrorSource_t::ALTRO_ERROR:
      return "Major ALTRO";
    case ErrorSource_t::MINOR_ALTRO_ERROR:
      return "Minor ALTRO";
    case ErrorSource_t::FIT_ERROR:
      return "Fit";
    case ErrorSource_t::GEOMETRY_ERROR:
      return "Geometry";
    case ErrorSource_t::GAIN_ERROR:
      return "Gain";
    case ErrorSource_t::STU_ERROR:
      return "STUDecoding";
    case ErrorSource_t::LINK_ERROR:
      return "Link missing";
    case ErrorSource_t::UNDEFINED:
      return "Unknown";
    default:
      return "";
  };
}

std::ostream& operator<<(std::ostream& stream, const ErrorTypeFEE& error)
{
  error.PrintStream(stream);
  return stream;
}
