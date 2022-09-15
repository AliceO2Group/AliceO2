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
#include "ZDCCalib/NoiseCalibEPN.h"
#include "Framework/Logger.h"

using namespace o2::zdc;

int NoiseCalibEPN::init()
{
  if (mVerbosity > DbgZero) {
    mModuleConfig->print();
  }

  // Inspect reconstruction parameters
  o2::zdc::RecoParamZDC& ropt = const_cast<o2::zdc::RecoParamZDC&>(RecoParamZDC::Instance());
  ropt.print();
  mRopt = (o2::zdc::RecoParamZDC*)&ropt;

  // Inspect calibration parameters
  o2::zdc::CalibParamZDC& opt = const_cast<o2::zdc::CalibParamZDC&>(CalibParamZDC::Instance());
  opt.print();
  if (opt.rootOutput == true) {
    setSaveDebugHistos();
  }

  for (int isig = 0; isig < NChannels; isig++) {
    mH[isig] = new o2::dataformats::FlatHisto1D<double>(4096, -2048.7, 2047.5);
  }

  // Fill maps to decode the pattern of channels with hit
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
    }
  next_ich:;
    if (mVerbosity > DbgZero) {
      LOG(info) << "Channel " << ich << "(" << ChannelNames[ich] << ") mod " << ropt.amod[ich] << " ch " << ropt.ach[ich];
    }
  }

  mInitDone = true;
  return 0;
}

//______________________________________________________________________________
int NoiseCalibEPN::process(const gsl::span<const o2::zdc::BCData>& bcdata, const gsl::span<const o2::zdc::ChannelData>& chdata)
{
  if (!mInitDone) {
    init();
  }
  auto nbc = bcdata.size();
  for (int ibc = 1; ibc < nbc; ibc++) {
    auto& bcp = bcdata[ibc - 1];
    auto& bcc = bcdata[ibc];
    if (bcc.ir.bc != 0 || bcp.ir.bc != 3563 || (bcp.ir.orbit + 1) != bcc.ir.orbit) {
      continue;
    }
    auto chEnt = bcc.ref.getFirstEntry();
    auto nch = bcc.ref.getEntries();
    for (int ich = 0; ich < nch; ich++) {
      const auto& chd = chdata[chEnt++];
      if (chd.id < NChannels) {
        // Check trigger flags
        ModuleTriggerMapData mtc, mtp;
        mtp.w = bcp.moduleTriggers[mRopt->amod[chd.id]];
        mtc.w = bcc.moduleTriggers[mRopt->amod[chd.id]];
        if (mtp.f.Auto_m                                                         // Auto trigger in bunch -2
            || mtp.f.Auto_0 || mtp.f.Alice_0 || (bcp.triggers & mChMask[chd.id]) // Trigger or hit in bunch -1
            || mtc.f.Auto_0 || mtc.f.Alice_0 || (bcc.triggers & mChMask[chd.id]) // Trigger or hit in bunch -2
            || mtc.f.Auto_1 || mtc.f.Alice_1                                     // Trigger in bunch +1
        ) {
#ifdef O2_ZDC_DEBUG
          printf("%u.%04u SKIP %s%s%s%s%s%s%s%s%s\n",
                 mtp.f.Auto_m ? "p.Auto_m" : "",
                 mtp.f.Auto_0 ? "p.Auto_0" : "",
                 mtp.f.Alice_0 ? "p.Alice_0" : "",
                 (bcp.triggers & mChMask[chd.id]) ? "p.HIT" : "",
                 mtc.f.Auto_0 ? "c.Auto_0" : "",
                 mtc.f.Alice_0 ? "c.Alice_0" : "",
                 (bcc.triggers & mChMask[chd.id]) ? "c.HIT" : "",
                 mtc.f.Auto_1 ? "c.Auto_1" : "",
                 mtc.f.Alice_1 ? "c.Alice_1" : "");
#endif
          continue;
        }
        int ss = 0;
        int sq = 0;
        for (int is = 0; is < NTimeBinsPerBC; is++) {
          auto s = chd.data[is];
          mH[chd.id]->fill(s);
          ss += s;
          sq += s * s;
        }
        int v = NTimeBinsPerBC * sq - ss * ss;
        if (v > 0) {
          // This should always be the case
          mData.addEntry(chd.id, v);
        }
      }
    }
  }
  return 0;
}

//______________________________________________________________________________
int NoiseCalibEPN::endOfRun()
{
  if (mVerbosity > DbgZero) {
    mData.print();
  }
  if (mSaveDebugHistos) {
    saveDebugHistos();
  }
  return 0;
}

//______________________________________________________________________________
int NoiseCalibEPN::saveDebugHistos(const std::string fn)
{
  int ierr = mData.saveDebugHistos(fn);
  if (ierr != 0) {
    return ierr;
  }
  TDirectory* cwd = gDirectory;
  TFile* f = new TFile(fn.data(), "update");
  if (f->IsZombie()) {
    LOG(error) << "Cannot update file: " << fn;
    return 1;
  }
  for (int32_t is = 0; is < NChannels; is++) {
    auto p = mH[is]->createTH1F(TString::Format("hs%d", is).Data());
    p->SetTitle(TString::Format("Baseline samples %s", ChannelNames[is].data()));
    p->Write("", TObject::kOverwrite);
  }
  f->Close();
  cwd->cd();
  return 0;
}
