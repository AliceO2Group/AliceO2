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
#include <TPad.h>
#include <TString.h>
#include <TStyle.h>
#include <TDirectory.h>
#include <TPaveStats.h>
#include <TAxis.h>
#include "ZDCCalib/TDCCalibData.h"
#include "ZDCCalib/TDCCalibEPN.h"
#include "ZDCCalib/TDCCalib.h"
#include "ZDCReconstruction/ZDCEnergyParam.h"
#include "ZDCReconstruction/ZDCTowerParam.h"
#include "ZDCReconstruction/ZDCTDCParam.h" //added by me
#include "Framework/Logger.h"

using namespace o2::zdc;

int TDCCalibEPN::init()
{
  if (mTDCCalibConfig == nullptr) {
    LOG(fatal) << "o2::zdc::TDCCalibEPN: missing configuration object";
    return -1;
  }
  clear();
  auto* cfg = mTDCCalibConfig;
  int ih;
  // clang-format off
  for (int iTDC = 0; iTDC < NTDC; iTDC++) {
    mTDC[iTDC] =    new o2::dataformats::FlatHisto1D<float>(cfg->nb1[iTDC],cfg->amin1[iTDC],cfg->amax1[iTDC]);
  }

  // clang-format on
  mInitDone = true;
  return 0;
}

//----//

int TDCCalibEPN::process(const gsl::span<const o2::zdc::BCRecData>& RecBC,
                         const gsl::span<const o2::zdc::ZDCEnergy>& Energy,
                         const gsl::span<const o2::zdc::ZDCTDCData>& TDCData,
                         const gsl::span<const uint16_t>& Info)
{
  if (!mInitDone) {
    init();
  }
  LOG(info) << "o2::zdc::TDCCalibEPN processing " << RecBC.size() << " b.c. @ TS " << mData.mCTimeBeg << " : " << mData.mCTimeEnd;
  o2::zdc::RecEventFlat ev;
  ev.init(RecBC, Energy, TDCData, Info);
  while (ev.next()) {
    if (ev.getNInfo() > 0) {
      auto& decodedInfo = ev.getDecodedInfo();
      for (uint16_t info : decodedInfo) {
        uint8_t ch = (info >> 10) & 0x1f;
        uint16_t code = info & 0x03ff;
        // hmsg->Fill(ch, code);
      }
      if (mVerbosity > DbgMinimal) {
        ev.print();
      }
      // Need clean data (no messages)
      // We are sure there is no pile-up in any channel (too restrictive?)
      continue;
    }
    if (ev.getNEnergy() > 0 && ev.mCurB.triggers == 0) {
      LOGF(info, "%9u.%04u Untriggered bunch", ev.mCurB.ir.orbit, ev.mCurB.ir.bc);
      // Skip!
      continue;
    }

    //Fill 1d histograms with tdc values. Check if channel is acquired or not
    for (int itdc = 0; itdc < NTDC; itdc++) { //loop over all TDCs
      int nhits = ev.NtdcV(itdc);

      if (nhits > 0) {
        //call fill function to fill histo
        fill1D(itdc, nhits, ev);
      }
    }
  }
  return 0;
}

//----//

int TDCCalibEPN::endOfRun()
{
  if (mVerbosity > DbgZero) {
    LOGF(info, "TDCCalibEPN::endOfRun ts (%llu:%llu)", mData.mCTimeBeg, mData.mCTimeEnd);
    std::cout << "End of run here" << std::endl;
    for (int ih = 0; ih < NTDC; ih++) {
      LOGF(info, "%s %i events and cuts (%g:%g)", TDCCalibData::CTDC[ih], mData.entries[ih], mTDCCalibConfig->cutLow[ih], mTDCCalibConfig->cutHigh[ih]);
    }
  }
  if (mSaveDebugHistos) {
    write();
  }
  return 0;
}

//----//

void TDCCalibEPN::clear(int ih)
{
  int ihstart = 0;
  int ihstop = NTDC;

  for (int32_t ii = ihstart; ii < ihstop; ii++) {
    if (mTDC[ii]) {
      mTDC[ii]->clear();
    }
  }
}

//----//

void TDCCalibEPN::fill1D(int iTDC, int nHits, o2::zdc::RecEventFlat ev)
{
  //Get TDC values
  float tdcVal[nHits];
  for (int i = 0; i < nHits; i++) {
    tdcVal[i] = ev.tdcV(iTDC, i);
  }

  //Fill histo
  for (int hit = 0; hit < nHits; hit++) {
    mTDC[iTDC]->fill(tdcVal[hit]);
  }
  mData.entries[iTDC] += nHits;
}

//----//

int TDCCalibEPN::write(const std::string fn)
{
  TDirectory* cwd = gDirectory;
  TFile* f = new TFile(fn.data(), "recreate");
  if (f->IsZombie()) {
    LOG(error) << "Cannot create file: " << fn;
    return 1;
  }
  for (int32_t ih = 0; ih < NTDC; ih++) {
    if (mTDC[ih]) {
      auto p = mTDC[ih]->createTH1F(TDCCalibData::CTDC[ih]);
      p->SetTitle(TDCCalibData::CTDC[ih]);
      p->Write("", TObject::kOverwrite);
    }
  }
  f->Close();
  cwd->cd();
  return 0;
}