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

#ifndef ZDC_WAVEFORMCALIB_QUEUE_H
#define ZDC_WAVEFORMCALIB_QUEUE_H

#include "CommonDataFormat/InteractionRecord.h"
#include "ZDCBase/Constants.h"
#include "DataFormatsZDC/BCData.h"
#include "DataFormatsZDC/ChannelData.h"
#include "DataFormatsZDC/OrbitData.h"
#include "DataFormatsZDC/RecEvent.h"
#include "DataFormatsZDC/RecEventAux.h"
#include "ZDCCalib/WaveformCalibConfig.h"
#include <queue>

/// \file WaveformCalibQueue.h
/// \brief Waveform calibration intermediate data queue
/// \author pietro.cortese@cern.ch

namespace o2
{
namespace zdc
{

struct WaveformCalibQueue {
  static constexpr int NH = WaveformCalibConfig::NH;
  std::queue<o2::InteractionRecord> mIR;
  int append(const RecEventAux &ev);
  int appendEv(const RecEventAux &ev);
};

} // namespace zdc
} // namespace o2

#endif
