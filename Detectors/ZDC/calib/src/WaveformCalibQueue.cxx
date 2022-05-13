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

#include "Framework/Logger.h"
#include "ZDCCalib/WaveformCalibQueue.h"

namespace o2
{
namespace zdc
{

// appends an event to the queue with the request that there are at most
// mN consecutive bunches
// The TDC conditions are checked and returned in output
uint32_t WaveformCalibQueue::append(RecEventFlat& ev)
{
  auto& toadd = ev.ir;
  // If queue is empty insert event
  if (mIR.size() == 0) {
    appendEv(ev);
    return 0;
  }
  // Check last element
  auto& last = mIR.back();
  // If BC are not consecutive, clear queue
  if (toadd.differenceInBC(last) > 1) {
    clear();
  }
  // If queue is not empty and is too long remove first element
  while (mIR.size() >= mN) {
    pop();
  }
  // If BC are consecutive or cleared queue append element
  appendEv(ev);
  if (mIR.size() == mN) {
    uint32_t mask = 0;
    for (int32_t itdc = 0; itdc < NTDCChannels; itdc++) {
      // Check which channels satisfy the condition on TDC
      bool tdccond = true;
      for (int i = 0; i < mN; i++) {
        int n = mNTDC[itdc].at(i);
        if (i == mPk) {
          if (n != 1) {
            tdccond = false;
            break;
          } else {
            auto tdca = mTDCA[itdc].at(i);
            auto tdcv = mTDCP[itdc].at(i);
            if (tdca < mCfg->cutLow[itdc] || tdca > mCfg->cutHigh[itdc] || tdcv < mCfg->cutTimeLow[itdc] || tdcv > mCfg->cutTimeHigh[itdc]) {
              tdccond = false;
              break;
            }
          }
        } else {
          if (n != 0) {
            tdccond = false;
            break;
          }
        }
      }
      if (tdccond) {
        mask = mask | (0x1 << itdc);
      }
    }
    return mask;
  } else {
    return 0;
  }
}

void WaveformCalibQueue::appendEv(RecEventFlat& ev)
{
  mIR.push_back(ev.ir);
  mEntry.push_back(ev.getNextEntry() - 1);

  auto& curb = ev.getCurB();
  int firstw, nw;
  curb.getRefW(firstw, nw);
  mFirstW.push_back(firstw);
  mNW.push_back(nw);

  for (int ih = 0; ih < NH; ih++) {
    mHasInfos[ih].push_back(false);
  }
  if (ev.getNInfo() > 0) {
    // Need clean data (no messages)
    // We are sure there is no pile-up in any channel (too restrictive?)
    auto& decodedInfo = ev.getDecodedInfo();
    for (uint16_t info : decodedInfo) {
      uint8_t ch = (info >> 10) & 0x1f;
      uint16_t code = info & 0x03ff;
      auto& last = mHasInfos[SignalTDC[ch]].back();
      last = true;
    }
    // if (mVerbosity > DbgMinimal) {
    //   ev.print();
    // }
  }
  // NOTE: for the moment NH = NTDCChannels. If we want to extend it to all channels
  // we will need to have a mask of affected channels (towers contributing to sum)
  for (int32_t itdc = 0; itdc < NTDCChannels; itdc++) {
    int ich = o2::zdc::TDCSignal[itdc];
    int nhit = ev.NtdcV(itdc);
    if (ev.NtdcA(itdc) != nhit) {
      LOGF(error, "Mismatch in TDC %d data Val=%d Amp=%d\n", itdc, ev.NtdcV(itdc), ev.NtdcA(ich));
      mNTDC[itdc].push_back(0);
      mTDCA[itdc].push_back(0);
      mTDCP[itdc].push_back(0);
    } else if (nhit == 0) {
      mNTDC[itdc].push_back(0);
      mTDCA[itdc].push_back(0);
      mTDCP[itdc].push_back(0);
    } else {
      // Store single TDC entry
      mNTDC[itdc].push_back(nhit);
      mTDCA[itdc].push_back(ev.tdcA(itdc, 0));
      mTDCP[itdc].push_back(ev.tdcV(itdc, 0));
    }
  }
}

int WaveformCalibQueue::hasData(int isig, const gsl::span<const o2::zdc::ZDCWaveform>& wave)
{
  int ipk = -1;
  int ipkb = -1;
  float min = std::numeric_limits<float>::infinity();
  for (int ib = 0; ib < mN; ib++) {
    int ifound = false;
    LOG(info) << "mNW[" << ib << "] = " << mNW[ib] << " mFirstW = " << mFirstW[ib];
    for (int iw = 0; iw < mNW[ib]; iw++) {
      auto& mywave = wave[iw + mFirstW[ib]];
      if (mywave.ch() == isig) {
        ifound = true;
        for (int ip = 0; ip < NIS; ip++) {
          if (mywave.inter[ip] < min) {
            ipkb = ib;
            ipk = ip;
            min = mywave.inter[ip];
          }
        }
      }
    }
    // Need to have consecutive data for all bunches
    if (!ifound) {
      return -1;
    }
  }
  if (ipkb != mPk) {
    return -1;
  } else {
    int ipos = NTimeBinsPerBC * TSN * ipkb + ipk;
    LOG(info) << "isig = " << isig << " ipkb " << ipkb << " ipk " << ipk << " min " << min;
    return ipos;
  }
}

// Checks if waveform has available data and adds it to summary data
// a compensation of the time jitter
int WaveformCalibQueue::addData(int isig, const gsl::span<const o2::zdc::ZDCWaveform>& wave, WaveformCalibData& data)
{
  int ipkb = -1; // Bunch where peak is found
  int ipk = -1;  // peak position within bunch
  float min = std::numeric_limits<float>::infinity();
  for (int ib = 0; ib < mN; ib++) {
    int ifound = false;
    LOG(info) << "mNW[" << ib << "] = " << mNW[ib] << " mFirstW = " << mFirstW[ib];
    for (int iw = 0; iw < mNW[ib]; iw++) {
      auto& mywave = wave[iw + mFirstW[ib]];
      if (mywave.ch() == isig) {
        ifound = true;
        for (int ip = 0; ip < NIS; ip++) {
          if (mywave.inter[ip] < min) {
            ipkb = ib;
            ipk = ip;
            min = mywave.inter[ip];
          }
        }
      }
    }
    // Need to have consecutive data for all bunches
    if (!ifound) {
      return -1;
    }
  }
  if (ipkb != mPk) {
    return -1;
  } else {
    // For the moment only TDC channels are interpolated
    int ih = SignalTDC[isig];
    int ppos = NTimeBinsPerBC * TSN * ipkb + ipk;
    int ipos = mPeak - ppos;
    data.mEntries[ih]++;
    // Restrict validity range because of signal jitter
    if (ipos > data.mFirstValid[ih]) {
      data.mFirstValid[ih] = ipos;
    }
    // We know that points are consecutive
    for (int ib = 0; ib < mN; ib++) {
      for (int iw = 0; iw < mNW[ib]; iw++) {
        auto& mywave = wave[iw + mFirstW[ib]];
        if (mywave.ch() == isig) {
          for (int ip = 0; ip < NIS; ip++) {
            if (ipos >= 0 && ipos < mNP) {
              data.mWave[ih][ipos] += mywave.inter[ip];
            }
            ipos++;
          }
        }
      }
    }
    ipos--;
    // Restrict validity range because of signal jitter
    if (ipos < data.mLastValid[ih]) {
      data.mLastValid[ih] = ipos;
    }
    LOG(info) << "isig = " << isig << " ipkb " << ipkb << " ipk " << ipk << " min " << min << " range=[" << data.mFirstValid[ih] << ":" << ppos << ":" << data.mLastValid[ih] << "]";
    return ipos;
  }
}

} // namespace zdc
} // namespace o2
