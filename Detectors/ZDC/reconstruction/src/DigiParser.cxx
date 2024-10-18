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

#include <TMath.h>
#include <TROOT.h>
#include <TPad.h>
#include <TString.h>
#include <TAxis.h>
#include <TStyle.h>
#include <TPaveStats.h>
#include "Framework/Logger.h"
#include "CommonConstants/LHCConstants.h"
#include "ZDCReconstruction/DigiParser.h"
#include "ZDCReconstruction/RecoParamZDC.h"

namespace o2
{
namespace zdc
{

void DigiParser::init()
{
  LOG(info) << "Initialization of ZDC DigiParser";
  if (!mModuleConfig) {
    LOG(fatal) << "Missing ModuleConfig configuration object";
    return;
  }

  mTriggerMask = mModuleConfig->getTriggerMask();

  // Update reconstruction parameters
  o2::zdc::RecoParamZDC& ropt = const_cast<o2::zdc::RecoParamZDC&>(RecoParamZDC::Instance());
  ropt.print();
  mRopt = (o2::zdc::RecoParamZDC*)&ropt;

  // Fill maps channel maps for integration
  for (int ich = 0; ich < NChannels; ich++) {
    // If the reconstruction parameters were not manually set
    if (ropt.amod[ich] < 0 || ropt.ach[ich] < 0) {
      for (int im = 0; im < NModules; im++) {
        for (uint32_t ic = 0; ic < NChPerModule; ic++) {
          if (mModuleConfig->modules[im].channelID[ic] == ich && mModuleConfig->modules[im].readChannel[ic]) {
            ropt.amod[ich] = im;
            ropt.ach[ich] = ic;
            // Fill mask to identify all channels
            mChMask[ich] = (0x1 << (4 * im + ic));
            goto next_ich;
          }
        }
      }
    } else {
      // Fill mask to identify all channels
      mChMask[ich] = (0x1 << (4 * ropt.amod[ich] + ropt.ach[ich]));
    }
  next_ich:;
    if (mVerbosity > DbgZero) {
      LOG(info) << "Channel " << ich << "(" << ChannelNames[ich] << ") mod " << ropt.amod[ich] << " ch " << ropt.ach[ich];
    }
  }

  // Fill maps to decode the pattern of channels with hit
  for (int itdc = 0; itdc < NTDCChannels; itdc++) {
    // If the reconstruction parameters were not manually set
    if (ropt.tmod[itdc] < 0 || ropt.tch[itdc] < 0) {
      int isig = TDCSignal[itdc];
      for (int im = 0; im < NModules; im++) {
        for (uint32_t ic = 0; ic < NChPerModule; ic++) {
          if (mModuleConfig->modules[im].channelID[ic] == isig && mModuleConfig->modules[im].readChannel[ic]) {
            // ropt.updateFromString(TString::Format("RecoParamZDC.tmod[%d]=%d;",itdc,im));
            // ropt.updateFromString(TString::Format("RecoParamZDC.tch[%d]=%d;",itdc,ic));
            ropt.tmod[itdc] = im;
            ropt.tch[itdc] = ic;
            // Fill mask to identify TDC channels
            mTDCMask[itdc] = (0x1 << (4 * im + ic));
            goto next_itdc;
          }
        }
      }
    } else {
      mTDCMask[itdc] = (0x1 << (4 * ropt.tmod[itdc] + ropt.tch[itdc]));
    }
  next_itdc:;
    if (mVerbosity > DbgZero) {
      LOG(info) << "TDC " << itdc << "(" << ChannelNames[TDCSignal[itdc]] << ")"
                << " mod " << ropt.tmod[itdc] << " ch " << ropt.tch[itdc];
    }
  }

  double xmin = -3 * NTimeBinsPerBC - 0.5;
  double xmax = 2 * NTimeBinsPerBC - 0.5;
  int nbx = std::round(xmax - xmin);

  if (mTransmitted == nullptr) {
    mTransmitted = std::make_unique<TH1F>("ht", "Transmitted channels", NChannels, -0.5, NChannels - 0.5);
  }
  if (mFired == nullptr) {
    mFired = std::make_unique<TH1F>("hfired", "Fired channels", NChannels, -0.5, NChannels - 0.5);
  }
  for (uint32_t ich = 0; ich < NChannels; ich++) {
    if (mBaseline[ich] == nullptr) {
      TString hname = TString::Format("hp_%s", ChannelNames[ich].data());
      TString htit = TString::Format("Baseline %s;Average orbit baseline", ChannelNames[ich].data());
      mBaseline[ich] = std::make_unique<TH1F>(hname, htit, 65536, -32768.5, 32767.5);
    }
    if (mCounts[ich] == nullptr) {
      TString hname = TString::Format("hc_%s", ChannelNames[ich].data());
      TString htit = TString::Format("Counts %s; Orbit hits", ChannelNames[ich].data());
      mCounts[ich] = std::make_unique<TH1F>(hname, htit, o2::constants::lhc::LHCMaxBunches + 1, -0.5, o2::constants::lhc::LHCMaxBunches + 0.5);
    }
    if (mSignalTH[ich] == nullptr) {
      TString hname = TString::Format("hsth_%s", ChannelNames[ich].data());
      TString htit = TString::Format("Signal %s AUTOT & Hit; Sample; ADC", ChannelNames[ich].data());
      if(mRejectPileUp){
        mSignalTH[ich] = std::make_unique<TH2F>(hname, htit, 3 * NTimeBinsPerBC, -0.5 - 1 * NTimeBinsPerBC, 2 * NTimeBinsPerBC - 0.5, ADCRange, ADCMin - 0.5, ADCMax + 0.5);
      }else{
        mSignalTH[ich] = std::make_unique<TH2F>(hname, htit, 5 * NTimeBinsPerBC, -0.5 - 3 * NTimeBinsPerBC, 2 * NTimeBinsPerBC - 0.5, ADCRange, ADCMin - 0.5, ADCMax + 0.5);
      }
    }
    if (mBunchH[ich] == nullptr) {
      TString hname = TString::Format("hbh_%s", ChannelNames[ich].data());
      TString htit = TString::Format("Bunch %s AUTOT Hit; BC units; - BC hundreds", ChannelNames[ich].data());
      mBunchH[ich] = std::make_unique<TH2F>(hname, htit, 100, -0.5, 99.5, 36, -35.5, 0.5);
    }
  }
} // init

void DigiParser::eor()
{
  TFile* f = new TFile(mOutput.data(), "recreate");
  if (f->IsZombie()) {
    LOG(fatal) << "Cannot write to file " << f->GetName();
    return;
  }
  for (uint32_t i = 0; i < NChannels; i++) {
    setStat(mBunchH[i].get());
    mBunchH[i]->Write();
  }
  for (uint32_t i = 0; i < NChannels; i++) {
    setStat(mBaseline[i].get());
    mBaseline[i]->Write();
  }
  for (uint32_t i = 0; i < NChannels; i++) {
    setStat(mCounts[i].get());
    mCounts[i]->Write();
  }
  for (uint32_t i = 0; i < NChannels; i++) {
    setStat(mSignalTH[i].get());
    mSignalTH[i]->Write();
  }
  setModuleLabel(mTransmitted.get());
  mTransmitted->Write();
  setModuleLabel(mFired.get());
  mFired->Write();
  f->Close();
}

int DigiParser::process(const gsl::span<const o2::zdc::OrbitData>& orbitdata, const gsl::span<const o2::zdc::BCData>& bcdata, const gsl::span<const o2::zdc::ChannelData>& chdata)
{
  // We assume that vectors contain data from a full time frame
  int norb = orbitdata.size();

  uint32_t scaler[NChannels] = {0};
  for (int iorb = 0; iorb < norb; iorb++) {
    for (int ich = 0; ich < NChannels; ich++) {
      if (orbitdata[iorb].scaler[ich] <= o2::constants::lhc::LHCMaxBunches) {
        scaler[ich] += orbitdata[iorb].scaler[ich];
        mCounts[ich]->Fill(orbitdata[iorb].scaler[ich]);
        auto myped = float(orbitdata[iorb].data[ich]) * mModuleConfig->baselineFactor;
        if (myped >= ADCMin && myped <= ADCMax) {
          // Pedestal information is present for this channel
          mBaseline[ich]->Fill(myped);
        }
      } else {
        LOG(warn) << "Corrupted scaler data for orbit " << orbitdata[iorb].ir.orbit;
      }
    }
  }

  mNBC = bcdata.size();
  std::vector<std::array<uint32_t, NChannels>> chRef; /// Cache of references
  chRef.resize(mNBC);

  // Assign data references
  for (int ibc = 0; ibc < mNBC; ibc++) {
    auto& bcd = bcdata[ibc];
    int chEnt = bcd.ref.getFirstEntry();
    for (int ich = 0; ich < NChannels; ich++) {
      chRef[ibc][ich] = ZDCRefInitVal;
    }
    for (int ic = 0; ic < bcd.ref.getEntries(); ic++) {
      auto& chd = chdata[chEnt];
      if (chd.id > IdDummy && chd.id < NChannels) {
        chRef[ibc][chd.id] = chEnt;
        mTransmitted->Fill(chd.id);
        if(bcdata[ibc].triggers & mChMask[chd.id] != 0){
          mFired->Fill(chd.id);
        }
      }
      chEnt++;
    }
  }

  for (uint32_t isig = 0; isig < NChannels; isig++) {
    for (int ibc = 0; ibc < mNBC; ibc++) {
      auto& ir = bcdata[ibc].ir;
      // Identify pile-up
      if (mRejectPileUp) {
        int nsig = 0;
        // Check previous bunches
        for (int ibn = -4; ibn < 5; ibn++) {
          int ibt = ibc + ibn;
          if (ibt >= 0) { // Check backward and current bunch
            if(ibt < mNBC){
            auto bcd = bcdata[ibt].ir.differenceInBC(ir);
            if (bcd == ibn) {
              if (bcdata[ibt].triggers & mChMask[isig] != 0) {
                nsig++;
              }
            }
          }else{
            break;
          }
          }
        }
        if (nsig>1) {
          continue;
        }
      }
      // Check previous, current and next bunch crossings
      for (int ibn = -1; ibn < 4; ibn++) {
        int ibt = ibc + ibn;
        if (ibt >= 0) {     // Check backward and current bunch
          if (ibt < mNBC) { // Check forward bunches
            auto bcd = bcdata[ibt].ir.differenceInBC(ir);
            if (bcd == 0) {
              // Fill bunch map
              if (bcdata[ibc].triggers & mChMask[isig] != 0) {
                double bc_d = uint32_t(ir.bc / 100);
                double bc_m = uint32_t(ir.bc % 100);
                mBunchH[isig]->Fill(bc_m, -bc_d);
                mFired->Fill(isig);
              }
            }
            if (bcd == ibn) {
              if (bcdata[ibt].triggers & mChMask[isig] != 0) {
                // Fill waveform
                auto ref = chRef[ibc][isig];
                if (ref != ZDCRefInitVal) {
                  for (int is = 0; is < NTimeBinsPerBC; is++) {
                    mSignalTH[isig]->Fill(-ibn * NTimeBinsPerBC + is, chdata[ref].data[is]);
                  }
                }
              }
            }
          } else {
            break;
          }
        }
      }
    }
  }
  return 0;
} // process

void DigiParser::setStat(TH1* h)
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

void DigiParser::setModuleLabel(TH1* h)
{
  for (uint32_t isig = 0; isig < NChannels; isig++) {
    h->GetXaxis()->SetBinLabel(isig + 1, ChannelNames[isig].data());
  }
}

} // namespace zdc
} // namespace o2
