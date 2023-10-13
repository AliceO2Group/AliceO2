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
#include "DataFormatsZDC/RecEventFlat.h"
#include "ZDCCalib/WaveformCalibData.h"
#include "ZDCCalib/WaveformCalibConfig.h"
#include <deque>

/// \file WaveformCalibQueue.h
/// \brief Waveform calibration intermediate data queue
/// \author pietro.cortese@cern.ch

namespace o2
{
namespace zdc
{

struct WaveformCalibQueue {
  WaveformCalibQueue() = default;
  WaveformCalibQueue(WaveformCalibConfig* cfg)
  {
    configure(cfg);
  }

  int mFirst = 0;           // First bunch of waveform w.r.t. bunch of signal peak
  int mLast = 0;            // Last bunch of waveform w.r.t. bunch of signal peak
  int mPk = 0;              // Bunch position of peak w.r.t. first bunch
  int mN = 1;               // Number of bunches in waveform
  int mPeak = 0;            // Peak position (interpolated samples) w.r.t. first point
  int mNP = 0;              // Number of interpolated points in waveform
  int mTimeLow[NChannels];  /// Cut on position difference low
  int mTimeHigh[NChannels]; /// Cut on position difference high
  int mVerbosity = 0;

  const WaveformCalibConfig* mCfg = nullptr;

  static int peak(int pk)
  {
    return NIS * pk + NIS / 2;
  }

  void configure(const WaveformCalibConfig* cfg);

  std::deque<int32_t> mEntry;               // Position of event
  std::deque<o2::InteractionRecord> mIR;    // IR of event
  std::deque<bool> mHasInfos[NChannels];    // Channel has info messages
  std::deque<uint32_t> mNTDC[NTDCChannels]; // Number of TDCs in event
  std::deque<float> mTDCA[NTDCChannels];    // Peak amplitude
  std::deque<float> mTDCP[NTDCChannels];    // Peak position
  std::deque<int32_t> mFirstW;              // Position of first waveform in event
  std::deque<int32_t> mNW;                  // Number of waveforms in event

  void clear()
  {
#ifdef O2_ZDC_WAVEFORMCALIB_DEBUG
    LOG(info) << "WaveformCalibConfig::" << __func__;
#endif
    mIR.clear();
    mEntry.clear();
    for (int isig = 0; isig < NChannels; isig++) {
      mHasInfos[isig].clear();
    }
    for (int itdc = 0; itdc < NTDCChannels; itdc++) {
      mNTDC[itdc].clear();
      mTDCA[itdc].clear();
      mTDCP[itdc].clear();
    }
    mFirstW.clear();
    mNW.clear();
  }

  void pop()
  {
    mIR.pop_front();
    mEntry.pop_front();
    for (int isig = 0; isig < NChannels; isig++) {
      mHasInfos[isig].pop_front();
    }
    for (int itdc = 0; itdc < NTDCChannels; itdc++) {
      mNTDC[itdc].pop_front();
      mTDCA[itdc].pop_front();
      mTDCP[itdc].pop_front();
    }
    mFirstW.pop_front();
    mNW.pop_front();
#ifdef O2_ZDC_WAVEFORMCALIB_DEBUG
    LOG(info) << "WaveformCalibConfig::" << __func__ << " remaining: " << mNW.size();
#endif
  }

  uint32_t append(RecEventFlat& ev);
  void appendEv(RecEventFlat& ev);
  int hasData(int isig, const gsl::span<const o2::zdc::ZDCWaveform>& wave);
  int addData(int isig, const gsl::span<const o2::zdc::ZDCWaveform>& wave, WaveformCalibData& data);
  void print();
  void printConf();
};

} // namespace zdc
} // namespace o2

#endif
