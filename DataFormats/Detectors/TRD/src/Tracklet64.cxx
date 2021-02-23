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
#include "DataFormatsTRD/Constants.h"

#include "fairlogger/Logger.h"
#include <iostream>

namespace o2
{

namespace trd
{

using namespace constants;

float Tracklet64::getY() const
{
  int padLocalBin = getPosition();
  int padLocal = 0;
  if (padLocalBin & (1 << (NBITSTRKLPOS - 1))) {
    padLocal = -((~(padLocalBin - 1)) & ((1 << NBITSTRKLPOS) - 1));
  } else {
    padLocal = padLocalBin & ((1 << NBITSTRKLPOS) - 1);
  }
  int mcmCol = (getMCM() % NMCMROBINCOL) + NMCMROBINCOL * (getROB() % 2);
  float offset = -63.f + ((float)NCOLMCM) * mcmCol;
  float padWidth = 0.635f + 0.03f * (getDetector() % NLAYER);
  return (offset + padLocal * GRANULARITYTRKLPOS) * padWidth;
}

float Tracklet64::getDy(float nTbDrift) const
{
  float dy;
  int dyLocalBin = getSlope();
  if (dyLocalBin & (1 << (NBITSTRKLSLOPE - 1))) {
    dy = (~(dyLocalBin - 1)) & ((1 << NBITSTRKLSLOPE) - 1);
    dy *= -1.f;
  } else {
    dy = dyLocalBin & ((1 << NBITSTRKLSLOPE) - 1);
  }
  float padWidth = 0.635f + 0.03f * (getDetector() % NLAYER);
  return dy * GRANULARITYTRKLSLOPE * padWidth * nTbDrift;
}

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
