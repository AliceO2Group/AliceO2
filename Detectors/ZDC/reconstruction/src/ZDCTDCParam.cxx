// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "FairLogger.h"
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

void ZDCTDCParam::print()
{
  LOG(INFO) << "TDCZNAC shift " << tdc_shift[TDCZNAC] << " ns";
  LOG(INFO) << "TDCZNAS shift " << tdc_shift[TDCZNAS] << " ns";
  LOG(INFO) << "TDCZPAC shift " << tdc_shift[TDCZPAC] << " ns";
  LOG(INFO) << "TDCZPAS shift " << tdc_shift[TDCZPAS] << " ns";
  LOG(INFO) << "TDCZEM1 shift " << tdc_shift[TDCZEM1] << " ns";
  LOG(INFO) << "TDCZEM2 shift " << tdc_shift[TDCZEM2] << " ns";
  LOG(INFO) << "TDCZNCC shift " << tdc_shift[TDCZNCC] << " ns";
  LOG(INFO) << "TDCZNCS shift " << tdc_shift[TDCZNCS] << " ns";
  LOG(INFO) << "TDCZPCC shift " << tdc_shift[TDCZPCC] << " ns";
  LOG(INFO) << "TDCZPCS shift " << tdc_shift[TDCZPCS] << " ns";
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
