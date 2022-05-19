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
  static constexpr int NH = WaveformCalibConfig::NH;
  WaveformCalibQueue() = default;
  WaveformCalibQueue(WaveformCalibConfig *cfg)
  {
    configure(cfg);
  }

  int mFirst = 0;
  int mLast = 0;
  int mPk = 0;
  int mN = 1;
  int mPPos = 0;
  int mNP = 0;
  int mPeak = 0;

  const WaveformCalibConfig *mCfg = nullptr;

  static int peak(int pk){
    return NTimeBinsPerBC * TSN * pk + NTimeBinsPerBC / 2 * TSN;
  }

  void configure(const WaveformCalibConfig *cfg)
  {
    mCfg = cfg;
    int ifirst = mCfg->getFirst();
    int ilast = mCfg->getLast();
    if (ifirst > 0 || ilast < 0 || ilast < ifirst) {
      LOGF(fatal, "WaveformCalibQueue configure error with ifirst=%d ilast=%d", ifirst, ilast);
    }
    mFirst = ifirst;
    mLast = ilast;
    mN = ilast - ifirst + 1;
    mPk = -mFirst;
    mPPos = mPk * NIS + NIS/2;
    mNP = mN * NIS;
    mPeak = peak(mPk);
  }

  std::deque<o2::InteractionRecord> mIR;
  std::deque<int32_t> mEntry;
  std::deque<bool> mHasInfos[NH];
  std::deque<uint32_t> mNTDC[NTDCChannels];
  std::deque<float> mTDCA[NTDCChannels];
  std::deque<float> mTDCP[NTDCChannels];
  std::deque<int32_t> mFirstW;
  std::deque<int32_t> mNW;
  void clear()
  {
    mIR.clear();
    mEntry.clear();
    for (int ih = 0; ih < NH; ih++) {
      mHasInfos[ih].clear();
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
    for (int ih = 0; ih < NH; ih++) {
      mHasInfos[ih].pop_front();
    }
    for (int itdc = 0; itdc < NTDCChannels; itdc++) {
      mNTDC[itdc].pop_front();
      mTDCA[itdc].pop_front();
      mTDCP[itdc].pop_front();
    }
    mFirstW.pop_front();
    mNW.pop_front();
  }
  uint32_t append(RecEventFlat& ev);
  void appendEv(RecEventFlat& ev);
  int hasData(int isig, const gsl::span<const o2::zdc::ZDCWaveform>& wave);
  int addData(int isig, const gsl::span<const o2::zdc::ZDCWaveform>& wave, WaveformCalibData& data);
};

} // namespace zdc
} // namespace o2

#endif
