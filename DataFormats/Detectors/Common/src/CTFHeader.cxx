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

#include "DetectorsCommonDataFormats/CTFHeader.h"
#include <Framework/Logger.h>

using namespace o2::ctf;
using DetID = o2::detectors::DetID;

/// describe itsel as a string
std::string CTFHeader::describe() const
{
  return fmt::format("Run:{:07d} TF:{} Orbit:{:08d} CreationTime:{} Detectors: {}", run, tfCounter, firstTForbit, creationTime, DetID::getNames(detectors));
}

void CTFHeader::print() const
{
  LOG(info) << describe();
}

std::ostream& o2::ctf::operator<<(std::ostream& stream, const CTFHeader& h)
{
  stream << h.describe();
  return stream;
}
