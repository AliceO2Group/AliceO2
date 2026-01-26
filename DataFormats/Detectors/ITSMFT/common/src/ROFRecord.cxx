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
#include <format>

#include "DataFormatsITSMFT/ROFRecord.h"
#include "Framework/Logger.h"

using namespace o2::itsmft;

std::string ROFRecord::asString() const
{
  return std::format("ROF: {} | {} entries starting from {} | IR: {}", mROFrame, getNEntries(), getFirstEntry(), mBCData.asString());
}

void ROFRecord::print() const
{
  LOG(info) << asString();
}

std::ostream& operator<<(std::ostream& stream, ROFRecord const& rec)
{
  stream << rec.asString();
  return stream;
}

std::string MC2ROFRecord::asString() const
{
  return std::format("MCEventID: {} ROFs: {}-{} Entry in ROFRecords: {}", eventRecordID, minROF, maxROF, rofRecordID);
}

void MC2ROFRecord::print() const
{
  LOG(info) << asString();
}

std::ostream& operator<<(std::ostream& stream, MC2ROFRecord const& rec)
{
  stream << rec.asString();
  return stream;
}
