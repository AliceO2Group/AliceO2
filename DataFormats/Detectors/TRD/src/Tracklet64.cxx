// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsTRD/Tracklet64.h"

#include "fairlogger/Logger.h"
#include <iostream>

namespace o2
{

namespace trd
{

void Tracklet64::printStream(std::ostream& stream) const
{
  stream << "Tracklet64 : 0x" << std::hex << getTrackletWord();
  stream << "\t hcid : " << getHCID() << " row:" << getPadRow() << " col:" << getColumn()
         << " Position:" << getPosition() << " slope:" << getSlope()
         << " PID:0x" << getPID()
         << " Q0:" << getQ0() << " Q1:" << getQ1() << " Q2:" << getQ2();
}

std::ostream& operator<<(std::ostream& stream, const Tracklet64& trg)
{
  trg.printStream(stream);
  return stream;
}

} // namespace trd
} // namespace o2
