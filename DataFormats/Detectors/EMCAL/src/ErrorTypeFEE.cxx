// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
