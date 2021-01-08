// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  return fmt::format("Run:{:07d} TF@orbit:{:08d} Detectors: {:s}", run, firstTForbit, DetID::getNames(detectors));
}

std::ostream& o2::ctf::operator<<(std::ostream& stream, const CTFHeader& h)
{
  stream << h.describe();
  return stream;
}
