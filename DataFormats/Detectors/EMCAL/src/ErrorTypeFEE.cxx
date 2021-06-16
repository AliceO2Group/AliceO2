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
#include <iostream>

using namespace o2::emcal;

void ErrorTypeFEE::PrintStream(std::ostream& stream) const
{
  stream << "EMCAL SM: " << getFEEID() << ", Decode error Type: " << getDecodeErrorType() << ", Raw fitter error: " << getRawFitErrorType();
}

std::ostream& operator<<(std::ostream& stream, const ErrorTypeFEE& error)
{
  error.PrintStream(stream);
  return stream;
}
