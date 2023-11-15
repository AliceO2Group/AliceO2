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
#include <EMCALReconstruction/RawDecodingError.h>

std::ostream& o2::emcal::operator<<(std::ostream& stream, const o2::emcal::RawDecodingError& error)
{
  stream << error.what();
  return stream;
}

std::ostream& o2::emcal::operator<<(std::ostream& stream, const o2::emcal::RawDecodingError::ErrorType_t& error)
{
  stream << o2::emcal::RawDecodingError::getErrorCodeNames(error);
  return stream;
}