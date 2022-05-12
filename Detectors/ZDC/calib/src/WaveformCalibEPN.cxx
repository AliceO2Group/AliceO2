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

#include <vector>
#include <TROOT.h>
#include <TFile.h>
#include <TPad.h>
#include <TString.h>
#include <TStyle.h>
#include <TDirectory.h>
#include <TPaveStats.h>
#include <TAxis.h>
#include "ZDCCalib/WaveformCalibEPN.h"
//#include "ZDCCalib/WaveformCalib.h"
#include "Framework/Logger.h"

using namespace o2::zdc;

int WaveformCalibEPN::init()
{
  if (mWaveformCalibConfig == nullptr) {
    LOG(fatal) << "o2::zdc::WaveformCalibEPN: missing configuration object";
    return -1;
  }

  auto* cfg = mWaveformCalibConfig;

  // number of bins
  mNBin = cfg->nbun * TSN;
  mFirst = cfg->ibeg;
  mLast = cfg->iend;
  mData.setN(cfg->nbun);
  for (int ih = 0; ih < NH; ih++) {
    mH[ih] = new o2::dataformats::FlatHisto1D<float>(mNBin, 0, mNBin);
  }
  mInitDone = true;
  return 0;
}

int WaveformCalibEPN::process(const gsl::span<const o2::zdc::BCRecData>& RecBC,
                              const gsl::span<const o2::zdc::ZDCEnergy>& Energy,
                              const gsl::span<const o2::zdc::ZDCTDCData>& TDCData,
                              const gsl::span<const uint16_t>& Info,
                              const gsl::span<const o2::zdc::ZDCWaveform>& wave)
{
  if (!mInitDone) {
    init();
  }
  mQueue.configure(mWaveformCalibConfig);
  o2::zdc::RecEventFlat ev;
  ev.init(RecBC, Energy, TDCData, Info);
  auto nen = ev.getEntries();
  std::vector<o2::InteractionRecord> ir;
  while (int ientry = ev.next()) {
    uint32_t mask = mQueue.append(ev);
    if (mask != 0) {
      // Analyze signals that refer to the TDC channels that satisfy condition
      // For the moment this is the same as for the TDCs
      for (int itdc = 0; itdc < NTDCChannels; itdc++) {
        if ((mask & (0x1 << itdc)) != 0) {
          // Check which channels have consecutive data
          int isig = TDCSignal[itdc];
          mQueue.addData(isig, wave, mData);
        }
      }
    }
  }
  return 0;
}

int WaveformCalibEPN::endOfRun()
{
  //   if (mVerbosity > DbgZero) {
  //     LOGF(info, "WaveformCalibEPN::endOfRun ts (%llu:%llu)", mData.mCTimeBeg, mData.mCTimeEnd);
  //     for (int ih = 0; ih < NH; ih++) {
  //       LOGF(info, "%s %g events and cuts (%g:%g)", WaveformCalibData::DN[ih], mData.mSum[ih][5][5], mWaveformCalibConfig->cutLow[ih], mWaveformCalibConfig->cutHigh[ih]);
  //     }
  //   }
  if (mSaveDebugHistos) {
    write();
  }
  return 0;
}

int WaveformCalibEPN::write(const std::string fn)
{
  /*
    TDirectory* cwd = gDirectory;
    TFile* f = new TFile(fn.data(), "recreate");
    if (f->IsZombie()) {
      LOG(error) << "Cannot create file: " << fn;
      return 1;
    }
    for (int32_t ih = 0; ih < (2 * NH); ih++) {
      if (mH[ih]) {
        auto p = mH[ih]->createTH1F(WaveformCalib::mHUncN[ih]);
        p->SetTitle(WaveformCalib::mHUncT[ih]);
        p->Write("", TObject::kOverwrite);
      }
    }
    for (int32_t ih = 0; ih < NH; ih++) {
      if (mC[ih]) {
        auto p = mC[ih]->createTH2F(WaveformCalib::mCUncN[ih]);
        p->SetTitle(WaveformCalib::mCUncT[ih]);
        p->Write("", TObject::kOverwrite);
      }
    }
    f->Close();
    cwd->cd();
  */
  return 0;
}
