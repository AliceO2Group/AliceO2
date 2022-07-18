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

#include <TROOT.h>
#include <TFile.h>
#include <TH1.h>
#include <TString.h>
#include <TStyle.h>
#include <TDirectory.h>
#include "ZDCCalib/CalibParamZDC.h"
#include "ZDCCalib/WaveformCalibEPN.h"
#include "Framework/Logger.h"

using namespace o2::zdc;

int WaveformCalibEPN::init()
{
  if (mConfig == nullptr) {
    LOG(fatal) << "o2::zdc::WaveformCalibEPN: missing configuration object";
    return -1;
  }

  auto* cfg = mConfig;
  if (mVerbosity > DbgZero) {
    mConfig->print();
  }

  // Inspect reconstruction parameters
  o2::zdc::CalibParamZDC& opt = const_cast<o2::zdc::CalibParamZDC&>(CalibParamZDC::Instance());
  opt.print();

  if (opt.debug_output > 0) {
    setSaveDebugHistos();
  }

  mQueue.configure(cfg);
  if (mVerbosity > DbgZero) {
    mQueue.printConf();
  }

  // number of bins
  mNBin = cfg->nbun * TSN;
  mFirst = cfg->ibeg;
  mLast = cfg->iend;
  mData.setN(cfg->nbun);
  mData.mPeak = mQueue.mPeak;
  mInitDone = true;
  return 0;
}

//______________________________________________________________________________
int WaveformCalibEPN::process(const gsl::span<const o2::zdc::BCRecData>& RecBC,
                              const gsl::span<const o2::zdc::ZDCEnergy>& Energy,
                              const gsl::span<const o2::zdc::ZDCTDCData>& TDCData,
                              const gsl::span<const uint16_t>& Info,
                              const gsl::span<const o2::zdc::ZDCWaveform>& wave)
{
  if (!mInitDone) {
    init();
  }
  o2::zdc::RecEventFlat ev;
  ev.init(RecBC, Energy, TDCData, Info);
  auto nen = ev.getEntries();
  std::vector<o2::InteractionRecord> ir;
  while (int ientry = ev.next()) {
    uint32_t mask = mQueue.append(ev);
#ifdef O2_ZDC_WAVEFORMCALIB_DEBUG
    LOGF(info, "WaveformCalibEPN::%s mask=0x%04x %s %s %s %s %s %s %s %s %s %s", __func__, mask,
         (mask & 0x001) ? ChannelNames[TDCSignal[0]] : "    ",
         (mask & 0x002) ? ChannelNames[TDCSignal[1]] : "    ",
         (mask & 0x004) ? ChannelNames[TDCSignal[2]] : "    ",
         (mask & 0x008) ? ChannelNames[TDCSignal[3]] : "    ",
         (mask & 0x010) ? ChannelNames[TDCSignal[4]] : "    ",
         (mask & 0x020) ? ChannelNames[TDCSignal[5]] : "    ",
         (mask & 0x040) ? ChannelNames[TDCSignal[6]] : "    ",
         (mask & 0x080) ? ChannelNames[TDCSignal[7]] : "    ",
         (mask & 0x100) ? ChannelNames[TDCSignal[8]] : "    ",
         (mask & 0x200) ? ChannelNames[TDCSignal[9]] : "    ");
#endif
    if (mask != 0) {
#ifdef O2_ZDC_WAVEFORMCALIB_DEBUG
      ev.print();
      ev.printDecodedMessages();
      mQueue.print();
#endif
      // Analyze signals that refer to the TDC channels that satisfy condition
      for (int isig = 0; isig < NChannels; isig++) {
        int itdc = SignalTDC[isig];
        if ((mask & (0x1 << itdc)) != 0) {
          // Check which channels have consecutive data
          mQueue.addData(isig, wave, mData);
        }
      }
    }
  }
  return 0;
}

//______________________________________________________________________________
int WaveformCalibEPN::endOfRun()
{
  if (mVerbosity > DbgZero) {
    LOGF(info, "WaveformCalibEPN::endOfRun ts (%llu:%llu)", mData.mCTimeBeg, mData.mCTimeEnd);
    for (int is = 0; is < NChannels; is++) {
      if (mData.getEntries(is) > 0) {
        LOGF(info, "Waveform %2d %s with %10d events and cuts AMP:(%g:%g) TDC:(%g:%g) Valid:[%d:%d:%d]", is, ChannelNames[is].data(),
             mData.getEntries(is), mConfig->cutLow[is], mConfig->cutHigh[is],
             mConfig->cutTimeLow[is], mConfig->cutTimeHigh[is],
             mData.getFirstValid(is), mData.mPeak, mData.getLastValid(is));
      }
    }
  }
  if (mSaveDebugHistos) {
    saveDebugHistos();
  }
  return 0;
}

//______________________________________________________________________________
int WaveformCalibEPN::saveDebugHistos(const std::string fn)
{
  return mData.saveDebugHistos(fn);
}
