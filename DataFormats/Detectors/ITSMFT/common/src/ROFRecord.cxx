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

#include "DataFormatsITSMFT/ROFRecord.h"
#include <iostream>
#include "fmt/format.h"

using namespace o2::itsmft;

std::string ROFRecord::asString() const
{
  return fmt::format("ROF: {} | {} entries starting from {}", mROFrame, getNEntries(), getFirstEntry());
}

void ROFRecord::print() const
{
  std::cout << this << "\n\t" << mBCData << std::endl;
}

std::ostream& operator<<(std::ostream& stream, ROFRecord const& rec)
{
  stream << rec.asString();
  return stream;
}

std::string MC2ROFRecord::asString() const
{
  return fmt::format("MCEventID: {} ROFs: {}-{} Entry in ROFRecords: {}", eventRecordID, minROF, maxROF, rofRecordID);
}

void MC2ROFRecord::print() const
{
  std::cout << this << std::endl;
}

std::ostream& operator<<(std::ostream& stream, MC2ROFRecord const& rec)
{
  stream << rec.asString();
  return stream;
}
