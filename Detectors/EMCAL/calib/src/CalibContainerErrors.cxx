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

#include <iostream>
#include "EMCALCalib/CalibContainerErrors.h"

using namespace o2::emcal;

CalibContainerIndexException::CalibContainerIndexException(unsigned int index) : std::exception(), mIndex(index), mErrorMessage()
{
  mErrorMessage = "Invalid index: " + std::to_string(mIndex);
}

std::ostream& o2::emcal::operator<<(std::ostream& stream, const o2::emcal::CalibContainerIndexException& obj)
{
  stream << obj.what();
  return stream;
}