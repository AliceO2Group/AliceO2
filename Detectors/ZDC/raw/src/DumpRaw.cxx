// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <TROOT.h>
#include <TPad.h>
#include <TString.h>
#include <TStyle.h>
#include <TPaveStats.h>
#include "ZDCRaw/DumpRaw.h"
#include "CommonConstants/LHCConstants.h"
#include "ZDCSimulation/Digits2Raw.h"
#include "FairLogger.h"

using namespace o2::zdc;

void DumpRaw::setStat(TH1* h)
{
  TString hn = h->GetName();
  h->Draw();
  gPad->Update();
  TPaveStats* st = (TPaveStats*)h->GetListOfFunctions()->FindObject("stats");
  st->SetFillStyle(1001);
  st->SetBorderSize(1);
  if (hn.BeginsWith("hp")) {
    st->SetOptStat(111111);
    st->SetX1NDC(0.1);
    st->SetX2NDC(0.3);
    st->SetY1NDC(0.640);
    st->SetY2NDC(0.9);
  } else if (hn.BeginsWith("hc")) {
    st->SetOptStat(1111);
    st->SetX1NDC(0.799);
    st->SetX2NDC(0.999);
    st->SetY1NDC(0.829);
    st->SetY2NDC(0.999);
  } else if (hn.BeginsWith("hs") || hn.BeginsWith("hb")) {
    st->SetOptStat(11);
    st->SetX1NDC(0.799);
    st->SetX2NDC(0.9995);
    st->SetY1NDC(0.904);
    st->SetY2NDC(0.999);
  }
}

void DumpRaw::init()
{
  gROOT->SetBatch();
  auto& sopt = ZDCSimParam::Instance();
  int nbx = (sopt.nBCAheadTrig + 1) * NTimeBinsPerBC;
  Double_t xmin = -sopt.nBCAheadTrig * NTimeBinsPerBC - 0.5;
  Double_t xmax = NTimeBinsPerBC - 0.5;
  for (UInt_t i = 0; i < NDigiChannels; i++) {
    uint32_t imod = i / NChPerModule;
    uint32_t ich = i % NChPerModule;
    if (mBaseline[i]) {
      mBaseline[i]->Reset();
    } else {
      TString hname = TString::Format("hp%d%d", imod, ich);
      TString htit = TString::Format("Baseline mod. %d ch. %d;Average orbit baseline", imod, ich);
      //mBaseline[i]=new TH1F(hname,htit,ADCRange,ADCMin-0.5,ADCMax+0.5);
      mBaseline[i] = new TH1F(hname, htit, 16378, -0.125, ADCMax + 0.125);
    }
    if (mCounts[i]) {
      mCounts[i]->Reset();
    } else {
      TString hname = TString::Format("hc%d%d", imod, ich);
      TString htit = TString::Format("Counts mod. %d ch. %d; Orbit hits", imod, ich);
      mCounts[i] = new TH1F(hname, htit, o2::constants::lhc::LHCMaxBunches + 1, -0.5, o2::constants::lhc::LHCMaxBunches + 0.5);
    }
    if (mSignal[i]) {
      mSignal[i]->Reset();
    } else {
      TString hname = TString::Format("hs%d%d", imod, ich);
      TString htit = TString::Format("Signal mod. %d ch. %d; Sample; ADC", imod, ich);
      mSignal[i] = new TH2F(hname, htit, nbx, xmin, xmax, ADCRange, ADCMin - 0.5, ADCMax + 0.5);
    }
    if (mBunch[i]) {
      mBunch[i]->Reset();
    } else {
      TString hname = TString::Format("hb%d%d", imod, ich);
      TString htit = TString::Format("Bunch mod. %d ch. %d; Sample; ADC", imod, ich);
      mBunch[i] = new TH2F(hname, htit, 100, -0.5, 99.5, 36, -35.5, 0.5);
    }
  }
  // Word id not present in payload
  mCh.f.fixed_0 = Id_wn;
  mCh.f.fixed_1 = Id_wn;
  mCh.f.fixed_2 = Id_wn;
}

void DumpRaw::write()
{
  TFile* f = new TFile("ZDCDumpRaw.root", "recreate");
  if (f->IsZombie()) {
    LOG(FATAL) << "Cannot write to file " << f->GetName();
    return;
  }
  for (UInt_t i = 0; i < NDigiChannels; i++) {
    if (mBunch[i] && mBunch[i]->GetEntries() > 0) {
      setStat(mBunch[i]);
      mBunch[i]->Write();
    }
  }
  for (UInt_t i = 0; i < NDigiChannels; i++) {
    if (mBaseline[i] && mBaseline[i]->GetEntries() > 0) {
      setStat(mBaseline[i]);
      mBaseline[i]->Write();
    }
  }
  for (UInt_t i = 0; i < NDigiChannels; i++) {
    if (mCounts[i] && mCounts[i]->GetEntries() > 0) {
      setStat(mCounts[i]);
      mCounts[i]->Write();
    }
  }
  for (UInt_t i = 0; i < NDigiChannels; i++) {
    if (mSignal[i] && mSignal[i]->GetEntries() > 0) {
      setStat(mSignal[i]);
      mSignal[i]->Write();
    }
  }
  f->Close();
}

inline int DumpRaw::getHPos(uint32_t board, uint32_t ch)
{
  int ih = board * 4 + ch;
  if (ih < NDigiChannels) {
    return ih;
  } else {
    LOG(ERROR) << "Wrong ih " << ih << " board " << board << " ch " << ch;
    return -1;
  }
}

int DumpRaw::processWord(const UInt_t* word)
{
  if (word == nullptr) {
    printf("NULL\n");
    return 1;
  }
  if ((word[0] & 0x3) == Id_w0) {
    for (Int_t iw = 0; iw < NWPerGBTW; iw++) {
      mCh.w[0][iw] = word[iw];
    }
  } else if ((word[0] & 0x3) == Id_w1) {
    if (mCh.f.fixed_0 == Id_w0) {
      for (Int_t iw = 0; iw < NWPerGBTW; iw++) {
        mCh.w[1][iw] = word[iw];
      }
    } else {
      LOG(ERROR) << "Wrong word sequence";
      mCh.f.fixed_0 = Id_wn;
      mCh.f.fixed_1 = Id_wn;
      mCh.f.fixed_2 = Id_wn;
    }
  } else if ((word[0] & 0x3) == Id_w2) {
    if (mCh.f.fixed_0 == Id_w0 && mCh.f.fixed_1 == Id_w1) {
      for (Int_t iw = 0; iw < NWPerGBTW; iw++) {
        mCh.w[2][iw] = word[iw];
      }
      process(mCh);
    } else {
      LOG(ERROR) << "Wrong word sequence";
    }
    mCh.f.fixed_0 = Id_wn;
    mCh.f.fixed_1 = Id_wn;
    mCh.f.fixed_2 = Id_wn;
  } else {
    // Word not present in payload
    LOG(FATAL) << "Event format error";
    return 1;
  }
  return 0;
}

int DumpRaw::process(const EventChData& ch)
{
  static constexpr int last_bc = o2::constants::lhc::LHCMaxBunches - 1;
  // Not empty event
  auto f = ch.f;
  int ih = getHPos(f.board, f.ch);
  if (mVerbosity > 0) {
    for (Int_t iw = 0; iw < NWPerBc; iw++) {
      Digits2Raw::print_gbt_word(ch.w[iw]);
    }
  }
  UShort_t us[12];
  Short_t s[12];
  us[0] = f.s00;
  us[1] = f.s01;
  us[2] = f.s02;
  us[3] = f.s03;
  us[4] = f.s04;
  us[5] = f.s05;
  us[6] = f.s06;
  us[7] = f.s07;
  us[8] = f.s08;
  us[9] = f.s09;
  us[10] = f.s10;
  us[11] = f.s11;
  for (Int_t i = 0; i < 12; i++) {
    if (us[i] > ADCMax) {
      s[i] = us[i] - ADCRange;
    } else {
      s[i] = us[i];
    }
    //printf("%d %u %d\n",i,us[i],s[i]);
  }
  if (f.Alice_3) {
    for (Int_t i = 0; i < 12; i++) {
      mSignal[ih]->Fill(i - 36., Double_t(s[i]));
    }
  }
  if (f.Alice_2) {
    for (Int_t i = 0; i < 12; i++) {
      mSignal[ih]->Fill(i - 24., Double_t(s[i]));
    }
  }
  if (f.Alice_1 || f.Auto_1) {
    for (Int_t i = 0; i < 12; i++) {
      mSignal[ih]->Fill(i - 12., Double_t(s[i]));
    }
  }
  if (f.Alice_0 || f.Auto_0) {
    for (Int_t i = 0; i < 12; i++) {
      mSignal[ih]->Fill(i + 0., Double_t(s[i]));
    }
    Double_t bc_d = UInt_t(f.bc / 100);
    Double_t bc_m = UInt_t(f.bc % 100);
    mBunch[ih]->Fill(bc_m, -bc_d);
  }
  if (f.bc == last_bc) {
    Int_t offset = f.offset - 32768;
    Double_t foffset = offset / 8.;
    mBaseline[ih]->Fill(foffset);
    mCounts[ih]->Fill(f.hits);
  }
  return 0;
}

int DumpRaw::process(const EventData& ev)
{
  for (Int_t im = 0; im < NModules; im++) {
    for (Int_t ic = 0; ic < NChPerModule; ic++) {
      if (ev.data[im][ic].f.fixed_0 == Id_w0 && ev.data[im][ic].f.fixed_1 == Id_w1 && ev.data[im][ic].f.fixed_2 == Id_w2) {
        process(ev.data[im][ic]);
      } else if (ev.data[im][ic].f.fixed_0 == 0 && ev.data[im][ic].f.fixed_1 == 0 && ev.data[im][ic].f.fixed_2 == 0) {
        // Empty channel
      } else {
        LOG(ERROR) << "Data format error";
      }
    }
  }
  return 0;
}
