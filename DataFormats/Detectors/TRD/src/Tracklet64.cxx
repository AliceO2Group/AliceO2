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

#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/HelperMethods.h"
#include "GPUCommonMath.h"

#include "fairlogger/Logger.h"
#include <iostream>

using namespace GPUCA_NAMESPACE::gpu;

namespace o2
{

namespace trd
{

using namespace constants;

void Tracklet64::print() const
{
  LOGF(info, "%02i_%i_%i, row(%i), col(%i), position(%i), slope(%i), pid(%i), q0(%i), q1(%i), q2(%i). Format(%i)",
       HelperMethods::getSector(getDetector()), HelperMethods::getStack(getDetector()), HelperMethods::getLayer(getDetector()), getPadRow(), getColumn(), getPosition(), getSlope(), getPID(), getQ0(), getQ1(), getQ2(), getFormat());
}

GPUd() int Tracklet64::getPadCol() const
{
  // obtain pad number relative to MCM center
  int padLocal = getPositionBinSigned() * GRANULARITYTRKLPOS;
  // MCM number in column direction (0..7)
  int mcmCol = (getMCM() % NMCMROBINCOL) + NMCMROBINCOL * (getROB() % 2);
  // FIXME: understand why the offset seems to be 6 pads and not nChannels / 2 = 10.5
  return CAMath::Nint(6.f + mcmCol * ((float)NCOLMCM) + padLocal);
}

#ifndef GPUCA_GPUCODE_DEVICE
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
#endif // GPUCA_GPUCODE_DEVICE

} // namespace trd
} // namespace o2
