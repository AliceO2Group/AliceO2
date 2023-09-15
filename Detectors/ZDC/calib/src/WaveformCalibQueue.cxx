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

void WaveformCalibQueue::configure(const WaveformCalibConfig* cfg)
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
  mPeak = peak(mPk);
  mNP = mN * NIS;
  for (int32_t isig = 0; isig < NChannels; isig++) {
    mTimeLow[isig] = std::nearbyint(cfg->cutTimeLow[SignalTDC[isig]] / FTDCVal);
    mTimeHigh[isig] = std::nearbyint(cfg->cutTimeHigh[SignalTDC[isig]] / FTDCVal);
  }
}

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
#ifdef O2_ZDC_WAVEFORMCALIB_DEBUG
    LOG(info) << "WaveformCalibQueue::" << __func__ << " gap detected. Clearing " << mIR.size() << " bc";
#endif
    clear();
  }
  // If queue is not empty and is too long remove first element
  while (mIR.size() >= mN) {
    pop();
  }
  // If BC are consecutive or cleared queue append element
  appendEv(ev);
  if (mIR.size() == mN) {
#ifdef O2_ZDC_WAVEFORMCALIB_DEBUG
    LOG(info) << "WaveformCalibQueue::" << __func__ << " processing " << mIR.size() << " bcs";
#endif
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
    // #ifdef O2_ZDC_WAVEFORMCALIB_DEBUG
    //     LOG(info) << "WaveformCalibQueue::" << __func__ << " IR size = " << mIR.size() << " != " << mN;
    // #endif
    return 0;
  }
}

void WaveformCalibQueue::appendEv(RecEventFlat& ev)
{
#ifdef O2_ZDC_WAVEFORMCALIB_DEBUG
  LOGF(info, "WaveformCalibQueue::%s %u.%04u", __func__, ev.ir.orbit, ev.ir.bc);
#endif
  mIR.push_back(ev.ir);
  mEntry.push_back(ev.getNextEntry() - 1);

  auto& curb = ev.getCurB();
  int firstw, nw;
  curb.getRefW(firstw, nw);
  mFirstW.push_back(firstw);
  mNW.push_back(nw);

  // Note: pile-up messages are computed only for the 10 TDCs
  for (int isig = 0; isig < NChannels; isig++) {
    mHasInfos[isig].push_back(false);
  }
  if (ev.getNInfo() > 0) {
    // Need clean data (no messages)
    auto& decodedInfo = ev.getDecodedInfo();
    for (uint16_t info : decodedInfo) {
      uint8_t ch = (info >> 10) & 0x1f;
      uint16_t code = info & 0x03ff;
      auto& last = mHasInfos[ch].back();
      last = true;
    }
    // if (mVerbosity > DbgMinimal) {
    //   ev.print();
    // }
  }
  // TDC channels are used to select reconstructed waveforms
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
#ifdef O2_ZDC_WAVEFORMCALIB_DEBUG
    LOG(info) << "WaveformCalibQueue::" << __func__ << " mNW[" << ib << "] = " << mNW[ib] << " mFirstW = " << mFirstW[ib];
#endif
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
#ifdef O2_ZDC_WAVEFORMCALIB_DEBUG
    LOG(info) << "WaveformCalibQueue::" << __func__ << " isig = " << isig << " ipkb " << ipkb << " ipk " << ipk << " min " << min;
#endif
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
  float max = -std::numeric_limits<float>::infinity();
  bool hasInfos = false;
  for (int ib = 0; ib < mN; ib++) {
    bool ifound = false;
    // #ifdef O2_ZDC_WAVEFORMCALIB_DEBUG
    //     LOG(info) << "mNW[" << ib << "/" << mN << "] = " << mNW[ib] << " mFirstW = " << mFirstW[ib];
    // #endif
    if (mHasInfos[isig][ib] || mHasInfos[TDCSignal[SignalTDC[isig]]][ib]) {
#ifdef O2_ZDC_WAVEFORMCALIB_DEBUG
      LOG(info) << "isig=" << isig << " ib=" << ib << " tdcid=" << SignalTDC[isig] << " tdc_sig=" << TDCSignal[SignalTDC[isig]] << " " << mHasInfos[isig][ib] << " " << mHasInfos[TDCSignal[SignalTDC[isig]]][ib];
#endif
      hasInfos = true;
    }
    for (int iw = 0; iw < mNW[ib]; iw++) {
      // Signal shouldn't have info messages. We check also corresponding TDC signal for pile-up information
      // TODO: relax this condition to avoid to check pedestal messages since pedestal is subtracted
      // when converting waveform calibration object into SimConfig object
      auto& mywave = wave[iw + mFirstW[ib]];
      if (mywave.ch() == isig) {
        ifound = true;
        for (int ip = 0; ip < NIS; ip++) {
          if (mywave.inter[ip] < min) {
            ipkb = ib;
            ipk = ip;
            min = mywave.inter[ip];
          }
          if (mywave.inter[ip] > max) {
            max = mywave.inter[ip];
          }
        }
      }
    }
#ifdef O2_ZDC_WAVEFORMCALIB_DEBUG
    LOG(info) << "WaveformCalibQueue::" << __func__ << " isig=" << isig << " mNW[" << ib << "] = " << mNW[ib] << " mFirstW = " << mFirstW[ib]
              << " ifound=" << ifound << " hasInfos=" << hasInfos;
#endif
    // Need to have consecutive data for all bunches
    if (!ifound || hasInfos) {
      return -1;
    }
  }
  if (ipkb != mPk) {
#ifdef O2_ZDC_WAVEFORMCALIB_DEBUG
    LOG(info) << "WaveformCalibQueue::" << __func__ << " isig = " << isig << " ipkb " << ipkb << " != mPk " << mPk << " SKIP";
#endif
    return -1;
  } else {
    int ppos = NIS * ipkb + ipk;
    int itdc = SignalTDC[isig];
    if (isig != TDCSignal[itdc]) {
      // Additional checks for towers
      float amp = max - min;
      if (amp < mCfg->cutLow[isig] || amp > mCfg->cutHigh[isig]) {
        // No warning messages for amplitude cuts on towers
        return -1;
      }
      if ((ppos - mPeak) < mTimeLow[itdc] || (ppos - mPeak) > mTimeHigh[itdc]) {
        if (mVerbosity > DbgMinimal) {
          // Put a warning message for a signal out of time
          LOGF(warning, "%d.%04d Signal %2d peak position %d-%d=%d is outside allowed range [%d:%d]", mIR[mPk].orbit, mIR[mPk].bc, isig, ppos, mPeak, ppos - mPeak, mTimeLow[isig], mTimeHigh[isig]);
        }
        return -1;
      }
    }
    int ipos = mPeak - ppos;
    data.addEntry(isig);
    // Restrict validity range because of signal jitter
    data.setFirstValid(isig, ipos);
    // We know that points are consecutive
    for (int ib = 0; ib < mN; ib++) {
      for (int iw = 0; iw < mNW[ib]; iw++) {
        auto& mywave = wave[iw + mFirstW[ib]];
        if (mywave.ch() == isig) {
          for (int ip = 0; ip < NIS; ip++) {
            if (ipos >= 0 && ipos < mNP) {
              // We don't use incapsulation because this section is called too many times
              data.mWave[isig].mData[ipos] += mywave.inter[ip];
            }
            ipos++;
          }
        }
      }
    }
    ipos--;
    // Restrict validity range because of signal jitter
    data.setLastValid(isig, ipos);
#ifdef O2_ZDC_WAVEFORMCALIB_DEBUG
    LOG(info) << "WaveformCalibQueue::" << __func__ << " isig = " << isig << " ipkb " << ipkb << " ipk " << ipk << " min " << min << " range=[" << data.getFirstValid(isig) << ":" << ppos << ":" << data.getLastValid(isig) << "]";
#endif
    return ipos;
  }
}

void WaveformCalibQueue::print()
{
  int n = mIR.size();
  printf("WaveformCalibQueue::print() %d consecutive bunches\n", n);
  for (int i = 0; i < n; i++) {
    printf("%d.%04d mEntry=%d mFirstW=%d mNW=%d waveforms\n", mIR[i].orbit, mIR[i].bc, mEntry[i], mFirstW[i], mNW[i]);
    bool printed = false;
    for (int j = 0; j < NChannels; j++) {
      if (mHasInfos[j][i] != 0) {
        if (!printed) {
          printf("mHasInfos:");
          printed = true;
        }
        printf(" %2d=%d", j, mHasInfos[j][i] != 0);
      }
    }
    if (printed) {
      printf("\n");
      printed = false;
    }
    for (int j = 0; j < NTDCChannels; j++) {
      if (mNTDC[j][i] > 0) {
        if (!printed) {
          printf("mNTDC:");
          printed = true;
        }
        printf(" %2d=%6u", j, mNTDC[j][i]);
      }
    }
    if (printed) {
      printf("\n");
      printed = false;
    }
    for (int j = 0; j < NTDCChannels; j++) {
      if (mNTDC[j][i] > 0) {
        if (!printed) {
          printf("mTDCA:");
          printed = true;
        }
        printf(" %2d=%6.1f", j, mTDCA[j][i]);
      }
    }
    if (printed) {
      printf("\n");
      printed = false;
    }
    for (int j = 0; j < NTDCChannels; j++) {
      if (mNTDC[j][i] > 0) {
        if (!printed) {
          printf("mTDCP:");
          printed = true;
        }
        printf(" %2d=%6.1f", j, mTDCP[j][i]);
      }
    }
    if (printed) {
      printf("\n");
    }
  }
}

void WaveformCalibQueue::printConf()
{
  LOG(info) << "WaveformCalibQueue::" << __func__;
  LOGF(info, "mFirst=%d mLast=%d mPk=%d mN=%d mPeak=%d/mNP=%d", mFirst, mLast, mPk, mN, mPeak, mNP);
  for (int isig = 0; isig < NChannels; isig++) {
    LOGF(info, "ch%02d pos [%d:%d]", isig, mTimeLow[isig], mTimeHigh[isig]);
  }
}

} // namespace zdc
} // namespace o2
