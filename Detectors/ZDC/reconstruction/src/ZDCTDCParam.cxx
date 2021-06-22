// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/Logger.h"
#include "ZDCReconstruction/ZDCTDCParam.h"

using namespace o2::zdc;

void ZDCTDCParam::setShift(uint32_t ich, float val)
{
  if (ich >= 0 && ich < NTDCChannels) {
    tdc_shift[ich] = val;
  } else {
    LOG(FATAL) << __func__ << " channel " << ich << " not in allowed range";
  }
}

float ZDCTDCParam::getShift(uint32_t ich) const
{
  if (ich >= 0 && ich < NTDCChannels) {
    return tdc_shift[ich];
  } else {
    LOG(FATAL) << __func__ << " channel " << ich << " not in allowed range";
    return 0;
  }
}

void ZDCTDCParam::print()
{
  for (int itdc = 0; itdc < o2::zdc::NTDCChannels; itdc++) {
    LOG(INFO) << ChannelNames[TDCSignal[itdc]] << " shift = " << tdc_shift[itdc] << " ns";
  }
}
