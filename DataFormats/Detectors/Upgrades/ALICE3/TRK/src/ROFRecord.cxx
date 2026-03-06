// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsTRK/ROFRecord.h"
#include <sstream>

ClassImp(o2::trk::ROFRecord);
ClassImp(o2::trk::MC2ROFRecord);

namespace o2::trk
{

std::string ROFRecord::asString() const
{
  std::ostringstream stream;
  stream << "IR=" << mBCData.asString() << " ROFrame=" << mROFrame
         << " first=" << mROFEntry.getFirstEntry() << " n=" << mROFEntry.getEntries();
  return stream.str();
}

} // namespace o2::trk
