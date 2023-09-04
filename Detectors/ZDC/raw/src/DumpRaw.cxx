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
#include <TPad.h>
#include <TString.h>
#include <TAxis.h>
#include <TStyle.h>
#include <TPaveStats.h>
#include "ZDCRaw/DumpRaw.h"
#include "CommonConstants/LHCConstants.h"
#include "ZDCSimulation/Digits2Raw.h"
#include "Framework/Logger.h"

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

void DumpRaw::setModuleLabel(TH1* h)
{
  for (int im = 0; im < NModules; im++) {
    for (int ic = 0; ic < NChPerModule; ic++) {
      h->GetXaxis()->SetBinLabel(im * NChPerModule + ic + 1, TString::Format("%d%d", im, ic));
    }
  }
}

void DumpRaw::setTriggerYLabel(TH2* h)
{
  h->GetYaxis()->SetBinLabel(10, "Alice_3");
  h->GetYaxis()->SetBinLabel(9, "Alice_2");
  h->GetYaxis()->SetBinLabel(8, "Alice_1");
  h->GetYaxis()->SetBinLabel(7, "Alice_0");
  h->GetYaxis()->SetBinLabel(6, "Auto_3");
  h->GetYaxis()->SetBinLabel(5, "Auto_2");
  h->GetYaxis()->SetBinLabel(4, "Auto_1");
  h->GetYaxis()->SetBinLabel(3, "Auto_0");
  h->GetYaxis()->SetBinLabel(2, "Auto_m");
  h->GetYaxis()->SetBinLabel(1, "None");
}

void DumpRaw::init()
{
  gROOT->SetBatch();
  auto& sopt = ZDCSimParam::Instance();
  double xmin = -sopt.nBCAheadTrig * NTimeBinsPerBC - 0.5;
  double xmax = 2 * NTimeBinsPerBC - 0.5;
  int nbx = std::round(xmax - xmin);
  if (mTransmitted == nullptr) {
    mTransmitted = std::make_unique<TH2F>("ht", "Transmitted channels", NModules, -0.5, NModules - 0.5, NChPerModule, -0.5, NChPerModule - 0.5);
  }
  if (mFired == nullptr) {
    mFired = std::make_unique<TH2F>("hfired", "Fired channels", NModules, -0.5, NModules - 0.5, NChPerModule, -0.5, NChPerModule - 0.5);
  }
  if (mLoss == nullptr) {
    mLoss = std::make_unique<TH1F>("hloss", "Data loss", NModules * NChPerModule, -0.5, NModules * NChPerModule - 0.5);
    setModuleLabel(mLoss.get());
  }
  if (mError == nullptr) {
    mError = std::make_unique<TH1F>("hError", "Error bit", NModules * NChPerModule, -0.5, NModules * NChPerModule - 0.5);
    setModuleLabel(mError.get());
  }
  if (mOve == nullptr) {
    mOve = std::make_unique<TH1F>("hove", "BC overflow", NModules * NChPerModule, -0.5, NModules * NChPerModule - 0.5);
    setModuleLabel(mOve.get());
  }
  if (mBits == nullptr) {
    mBits = std::make_unique<TH2F>("hb", "Trigger bits", NModules * NChPerModule, -0.5, NModules * NChPerModule - 0.5, 10, -0.5, 9.5);
    setTriggerYLabel(mBits.get());
    setModuleLabel(mBits.get());
  }
  if (mBitsH == nullptr) {
    mBitsH = std::make_unique<TH2F>("hbh", "Trigger bits HIT", NModules * NChPerModule, -0.5, NModules * NChPerModule - 0.5, 10, -0.5, 9.5);
    setTriggerYLabel(mBitsH.get());
    setModuleLabel(mBitsH.get());
  }
  for (uint32_t i = 0; i < NDigiChannels; i++) {
    uint32_t imod = i / NChPerModule;
    uint32_t ich = i % NChPerModule;
    if (mBaseline[i] == nullptr) {
      TString hname = TString::Format("hp%d%d", imod, ich);
      TString htit = TString::Format("Baseline mod. %d ch. %d;Average orbit baseline", imod, ich);
      // mBaseline[i]=std::make_unique<TH1F>(hname,htit,ADCRange,ADCMin-0.5,ADCMax+0.5);
      mBaseline[i] = std::make_unique<TH1F>(hname, htit, 65536, -32768.5, 32767.5);
    }
    if (mCounts[i] == nullptr) {
      TString hname = TString::Format("hc%d%d", imod, ich);
      TString htit = TString::Format("Counts mod. %d ch. %d; Orbit hits", imod, ich);
      mCounts[i] = std::make_unique<TH1F>(hname, htit, o2::constants::lhc::LHCMaxBunches + 1, -0.5, o2::constants::lhc::LHCMaxBunches + 0.5);
    }
    if (mSignalA[i] == nullptr) {
      TString hname = TString::Format("hsa%d%d", imod, ich);
      TString htit = TString::Format("Signal mod. %d ch. %d ALICET; Sample; ADC", imod, ich);
      mSignalA[i] = std::make_unique<TH2F>(hname, htit, nbx, xmin, xmax, ADCRange, ADCMin - 0.5, ADCMax + 0.5);
    }
    if (mSignalT[i] == nullptr) {
      TString hname = TString::Format("hst%d%d", imod, ich);
      TString htit = TString::Format("Signal mod. %d ch. %d AUTOT; Sample; ADC", imod, ich);
      mSignalT[i] = std::make_unique<TH2F>(hname, htit, nbx, xmin, xmax, ADCRange, ADCMin - 0.5, ADCMax + 0.5);
    }
    if (mBunchA[i] == nullptr) {
      TString hname = TString::Format("hba%d%d", imod, ich);
      TString htit = TString::Format("Bunch mod. %d ch. %d ALICET; Sample; ADC", imod, ich);
      mBunchA[i] = std::make_unique<TH2F>(hname, htit, 100, -0.5, 99.5, 36, -35.5, 0.5);
    }
    if (mBunchT[i] == nullptr) {
      TString hname = TString::Format("hbt%d%d", imod, ich);
      TString htit = TString::Format("Bunch mod. %d ch. %d AUTOT; Sample; ADC", imod, ich);
      mBunchT[i] = std::make_unique<TH2F>(hname, htit, 100, -0.5, 99.5, 36, -35.5, 0.5);
    }
    if (mBunchH[i] == nullptr) {
      TString hname = TString::Format("hbh%d%d", imod, ich);
      TString htit = TString::Format("Bunch mod. %d ch. %d AUTOT Hit; Sample; ADC", imod, ich);
      mBunchH[i] = std::make_unique<TH2F>(hname, htit, 100, -0.5, 99.5, 36, -35.5, 0.5);
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
    LOG(fatal) << "Cannot write to file " << f->GetName();
    return;
  }
  for (uint32_t i = 0; i < NDigiChannels; i++) {
    if (mBunchA[i] && mBunchA[i]->GetEntries() > 0) {
      setStat(mBunchA[i].get());
      mBunchA[i]->Write();
    }
  }
  for (uint32_t i = 0; i < NDigiChannels; i++) {
    if (mBunchT[i] && mBunchT[i]->GetEntries() > 0) {
      setStat(mBunchT[i].get());
      mBunchT[i]->Write();
    }
  }
  for (uint32_t i = 0; i < NDigiChannels; i++) {
    if (mBunchH[i] && mBunchH[i]->GetEntries() > 0) {
      setStat(mBunchH[i].get());
      mBunchH[i]->Write();
    }
  }
  for (uint32_t i = 0; i < NDigiChannels; i++) {
    if (mBaseline[i] && mBaseline[i]->GetEntries() > 0) {
      setStat(mBaseline[i].get());
      mBaseline[i]->Write();
    }
  }
  for (uint32_t i = 0; i < NDigiChannels; i++) {
    if (mCounts[i] && mCounts[i]->GetEntries() > 0) {
      setStat(mCounts[i].get());
      mCounts[i]->Write();
    }
  }
  for (uint32_t i = 0; i < NDigiChannels; i++) {
    if (mSignalA[i] && mSignalA[i]->GetEntries() > 0) {
      setStat(mSignalA[i].get());
      mSignalA[i]->Write();
    }
  }
  for (uint32_t i = 0; i < NDigiChannels; i++) {
    if (mSignalT[i] && mSignalT[i]->GetEntries() > 0) {
      setStat(mSignalT[i].get());
      mSignalT[i]->Write();
    }
  }
  mTransmitted->Write();
  mFired->Write();
  mBits->Write();
  mBitsH->Write();
  mLoss->Write();
  mError->Write();
  mOve->Write();
  f->Close();
}

inline int DumpRaw::getHPos(uint32_t board, uint32_t ch)
{
  int ih = board * 4 + ch;
  if (ih < NDigiChannels) {
    return ih;
  } else {
    LOG(error) << "Wrong ih " << ih << " board " << board << " ch " << ch;
    return -1;
  }
}

int DumpRaw::processWord(const uint32_t* word)
{
  if (word == nullptr) {
    printf("NULL\n");
    return 1;
  }
  // LOGF(info, "GBT word %04x %08x %08x id=%u", *((uint16_t*)&word[2]), word[1], word[0], word[0] & 0x3);
  if ((word[0] & 0x3) == Id_w0) {
    mCh.w[0][NWPerGBTW - 1] = 0;
    mCh.w[0][NWPerGBTW - 2] = 0;
    memcpy((void*)&mCh.w[0][0], (const void*)word, PayloadPerGBTW);
  } else if ((word[0] & 0x3) == Id_w1) {
    if (mCh.f.fixed_0 == Id_w0) {
      mCh.w[1][NWPerGBTW - 1] = 0;
      mCh.w[1][NWPerGBTW - 2] = 0;
      memcpy((void*)&mCh.w[1][0], (const void*)word, PayloadPerGBTW);
    } else {
      LOGF(error, "Wrong word sequence: %04x %08x %08x id=%u *%u*", *((uint16_t*)&word[2]), word[1], word[0], mCh.f.fixed_0, word[0] & 0x3);
      mCh.f.fixed_0 = Id_wn;
      mCh.f.fixed_1 = Id_wn;
      mCh.f.fixed_2 = Id_wn;
    }
  } else if ((word[0] & 0x3) == Id_w2) {
    if (mCh.f.fixed_0 == Id_w0 && mCh.f.fixed_1 == Id_w1) {
      mCh.w[2][NWPerGBTW - 1] = 0;
      mCh.w[2][NWPerGBTW - 2] = 0;
      memcpy((void*)&mCh.w[2][0], (const void*)word, PayloadPerGBTW);
      process(mCh);
    } else {
      LOGF(error, "Wrong word sequence: %04x %08x %08x id=%u %u *%u*", *((uint16_t*)&word[2]), word[1], word[0], mCh.f.fixed_0, mCh.f.fixed_1, word[0] & 0x3);
    }
    mCh.f.fixed_0 = Id_wn;
    mCh.f.fixed_1 = Id_wn;
    mCh.f.fixed_2 = Id_wn;
  } else {
    // Word id not foreseen in payload
    LOGF(error, "Event format error on word %04x %08x %08x id=%u", *((uint16_t*)&word[2]), word[1], word[0], word[0] & 0x3);
    return 1;
  }
  return 0;
}

int DumpRaw::process(const EventChData& ch)
{
  static constexpr int last_bc = o2::constants::lhc::LHCMaxBunches - 1;

  union {
    uint16_t uns;
    int16_t sig;
  } word16;

  // Not empty event
  auto f = ch.f;
  int ih = getHPos(f.board, f.ch);
  if (ih < 0) {
    return -1;
  }

  if (mVerbosity > 0) {
    for (int32_t iw = 0; iw < NWPerBc; iw++) {
      Digits2Raw::print_gbt_word(ch.w[iw]);
    }
  }

  mTransmitted->Fill(f.board, f.ch);
  if (f.Hit) {
    mFired->Fill(f.board, f.ch);
  }

  uint16_t us[12];
  int16_t s[12];
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
  for (int32_t i = 0; i < 12; i++) {
    if (us[i] > ADCMax) {
      s[i] = us[i] - ADCRange;
    } else {
      s[i] = us[i];
    }
    // printf("%d %u %d\n",i,us[i],s[i]);
  }
  if (f.Alice_3) {
    mBits->Fill(ih, 9);
    if (f.Hit) {
      mBitsH->Fill(ih, 9);
    }
    for (int32_t i = 0; i < 12; i++) {
      mSignalA[ih]->Fill(i - 36., double(s[i]));
    }
  }
  if (f.Alice_2) {
    mBits->Fill(ih, 8);
    if (f.Hit) {
      mBitsH->Fill(ih, 8);
    }
    for (int32_t i = 0; i < 12; i++) {
      mSignalA[ih]->Fill(i - 24., double(s[i]));
    }
  }
  if (f.Alice_1) {
    mBits->Fill(ih, 7);
    if (f.Hit) {
      mBitsH->Fill(ih, 7);
    }
    for (int32_t i = 0; i < 12; i++) {
      mSignalA[ih]->Fill(i - 12., double(s[i]));
    }
  }
  if (f.Alice_0) {
    mBits->Fill(ih, 6);
    if (f.Hit) {
      mBitsH->Fill(ih, 6);
    }
    for (int32_t i = 0; i < 12; i++) {
      mSignalA[ih]->Fill(i + 0., double(s[i]));
    }
    double bc_d = uint32_t(f.bc / 100);
    double bc_m = uint32_t(f.bc % 100);
    mBunchA[ih]->Fill(bc_m, -bc_d);
  }
  if (f.Auto_3) {
    mBits->Fill(ih, 5);
    if (f.Hit) {
      mBitsH->Fill(ih, 5);
    }
    for (int32_t i = 0; i < 12; i++) {
      mSignalT[ih]->Fill(i - 36., double(s[i]));
    }
  }
  if (f.Auto_2) {
    mBits->Fill(ih, 4);
    if (f.Hit) {
      mBitsH->Fill(ih, 4);
    }
    for (int32_t i = 0; i < 12; i++) {
      mSignalT[ih]->Fill(i - 24., double(s[i]));
    }
  }
  if (f.Auto_1) {
    mBits->Fill(ih, 3);
    if (f.Hit) {
      mBitsH->Fill(ih, 3);
    }
    for (int32_t i = 0; i < 12; i++) {
      mSignalT[ih]->Fill(i - 12., double(s[i]));
    }
  }
  if (f.Auto_0) {
    mBits->Fill(ih, 2);
    if (f.Hit) {
      mBitsH->Fill(ih, 2);
    }
    for (int32_t i = 0; i < 12; i++) {
      mSignalT[ih]->Fill(i + 0., double(s[i]));
    }
    double bc_d = uint32_t(f.bc / 100);
    double bc_m = uint32_t(f.bc % 100);
    mBunchT[ih]->Fill(bc_m, -bc_d);
  }
  if (f.Auto_m) {
    mBits->Fill(ih, 1);
    if (f.Hit) {
      mBitsH->Fill(ih, 1);
    }
    for (int32_t i = 0; i < 12; i++) {
      mSignalT[ih]->Fill(i + 12., double(s[i]));
    }
  }
  if (!(f.Alice_3 || f.Alice_2 || f.Alice_1 || f.Alice_0 || f.Alice_1 || f.Auto_3 || f.Auto_2 || f.Auto_1 || f.Auto_0 || f.Auto_m)) {
    mBits->Fill(ih, 0);
    if (f.Hit) {
      mBitsH->Fill(ih, 0);
    }
  }
  if (f.Hit) {
    double bc_d = uint32_t(f.bc / 100);
    double bc_m = uint32_t(f.bc % 100);
    mBunchH[ih]->Fill(bc_m, -bc_d);
  }
  if (f.bc >= o2::constants::lhc::LHCMaxBunches) {
    mOve->Fill(ih);
  }

  if (f.bc == last_bc) {
    word16.uns = f.offset;
    mBaseline[ih]->Fill(word16.sig);
    mCounts[ih]->Fill(f.hits & 0xfff);
    if (f.dLoss) {
      mLoss->Fill(ih);
    }
    if (f.error) {
      mError->Fill(ih);
    }
  }
  return 0;
}

int DumpRaw::process(const EventData& ev)
{
  for (int32_t im = 0; im < NModules; im++) {
    for (int32_t ic = 0; ic < NChPerModule; ic++) {
      if (ev.data[im][ic].f.fixed_0 == Id_w0 && ev.data[im][ic].f.fixed_1 == Id_w1 && ev.data[im][ic].f.fixed_2 == Id_w2) {
        process(ev.data[im][ic]);
      } else if (ev.data[im][ic].f.fixed_0 == 0 && ev.data[im][ic].f.fixed_1 == 0 && ev.data[im][ic].f.fixed_2 == 0) {
        // Empty channel
      } else {
        LOG(error) << "Data format error";
      }
    }
  }
  return 0;
}
