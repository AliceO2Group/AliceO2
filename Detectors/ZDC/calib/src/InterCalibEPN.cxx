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
#include "ZDCCalib/InterCalibEPN.h"
#include "ZDCReconstruction/ZDCEnergyParam.h"
#include "ZDCReconstruction/ZDCTowerParam.h"
#include "DataFormatsZDC/InterCalibData.h"
#include "Framework/Logger.h"

using namespace o2::zdc;

int InterCalibEPN::init()
{
  if (mInterCalibConfig == nullptr) {
    LOG(fatal) << "o2::zdc::InterCalibEPN: missing configuration object";
    return -1;
  }
  clear();
  auto* cfg = mInterCalibConfig;
  int ih;
  // clang-format off
  ih = 0; mH[ih] = std::make_unique<TH1F>("hZNAS","ZNA sum",cfg->nb1[ih],cfg->amin1[ih],cfg->amax1[ih]);
  ih = 1; mH[ih] = std::make_unique<TH1F>("hZPAS","ZPA sum",cfg->nb1[ih],cfg->amin1[ih],cfg->amax1[ih]);
  ih = 2; mH[ih] = std::make_unique<TH1F>("hZNCS","ZNC sum",cfg->nb1[ih],cfg->amin1[ih],cfg->amax1[ih]);
  ih = 3; mH[ih] = std::make_unique<TH1F>("hZPCS","ZPC sum",cfg->nb1[ih],cfg->amin1[ih],cfg->amax1[ih]);
  ih = 4; mH[ih] = std::make_unique<TH1F>("hZEM2","ZEM2"   ,cfg->nb1[ih],cfg->amin1[ih],cfg->amax1[ih]);
  ih = 0; mH[NH+ih] = std::make_unique<TH1F>("hZNAC","ZNA TC",cfg->nb1[ih],cfg->amin1[ih],cfg->amax1[ih]);
  ih = 1; mH[NH+ih] = std::make_unique<TH1F>("hZPAC","ZPA TC",cfg->nb1[ih],cfg->amin1[ih],cfg->amax1[ih]);
  ih = 2; mH[NH+ih] = std::make_unique<TH1F>("hZNCC","ZNC TC",cfg->nb1[ih],cfg->amin1[ih],cfg->amax1[ih]);
  ih = 3; mH[NH+ih] = std::make_unique<TH1F>("hZPCC","ZPC TC",cfg->nb1[ih],cfg->amin1[ih],cfg->amax1[ih]);
  ih = 4; mH[NH+ih] = std::make_unique<TH1F>("hZEM1","ZEM1",  cfg->nb1[ih],cfg->amin1[ih],cfg->amax1[ih]);
  ih = 0; mC[ih] = std::make_unique<TH2F>("cZNA","ZNA;TC;SUM",cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih],cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih]);
  ih = 1; mC[ih] = std::make_unique<TH2F>("cZPA","ZPA;TC;SUM",cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih],cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih]);
  ih = 2; mC[ih] = std::make_unique<TH2F>("cZNC","ZNC;TC;SUM",cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih],cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih]);
  ih = 3; mC[ih] = std::make_unique<TH2F>("cZPC","ZPC;TC;SUM",cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih],cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih]);
  ih = 4; mC[ih] = std::make_unique<TH2F>("cZEM","ZEM;ZEM1;ZEM2",cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih],cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih]);
  // clang-format on
  mInitDone = true;
  return 0;
}

int InterCalibEPN::process(const gsl::span<const o2::zdc::BCRecData>& RecBC,
                        const gsl::span<const o2::zdc::ZDCEnergy>& Energy,
                        const gsl::span<const o2::zdc::ZDCTDCData>& TDCData,
                        const gsl::span<const uint16_t>& Info)
{
  if (!mInitDone) {
    init();
  }
  LOG(info) << "o2::zdc::InterCalibEPN processing " << RecBC.size() << " b.c.";
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
      ev.print();
      // Need clean data (no messages)
      // We are sure there is no pile-up in any channel (too restrictive?)
      continue;
    }
    if (ev.getNEnergy() > 0 && ev.mCurB.triggers == 0) {
      LOGF(info, "%9u.%04u Untriggered bunch\n", ev.mCurB.ir.orbit, ev.mCurB.ir.bc);
      // Skip!
      continue;
    }
    if ((ev.ezdcDecoded & MaskZNA) == MaskZNA) {
      cumulate(HidZNA, ev.EZDC(IdZNAC), ev.EZDC(IdZNA1), ev.EZDC(IdZNA2), ev.EZDC(IdZNA3), ev.EZDC(IdZNA4), 1.);
    }
    if ((ev.ezdcDecoded & MaskZPA) == MaskZPA) {
      cumulate(HidZPA, ev.EZDC(IdZPAC), ev.EZDC(IdZPA1), ev.EZDC(IdZPA2), ev.EZDC(IdZPA3), ev.EZDC(IdZPA4), 1.);
    }
    if ((ev.ezdcDecoded & MaskZNC) == MaskZNC) {
      cumulate(HidZNC, ev.EZDC(IdZNCC), ev.EZDC(IdZNC1), ev.EZDC(IdZNC2), ev.EZDC(IdZNC3), ev.EZDC(IdZNC4), 1.);
    }
    if ((ev.ezdcDecoded & MaskZPC) == MaskZPC) {
      cumulate(HidZPC, ev.EZDC(IdZPCC), ev.EZDC(IdZPC1), ev.EZDC(IdZPC2), ev.EZDC(IdZPC3), ev.EZDC(IdZPC4), 1.);
    }
    if ((ev.ezdcDecoded & MaskZEM) == MaskZEM) {
      cumulate(HidZEM, ev.EZDC(IdZEM1), ev.EZDC(IdZEM2), 0., 0., 0., 1.);
    }
  }
  return 0;
}

int InterCalibEPN::endOfRun()
{
  for (int ih = 0; ih < NH; ih++) {
    LOGF(info, "%g events and cuts (%g:%g)\n", data.mSum[ih][5][5], mInterCalibConfig->cutLow[ih], mInterCalibConfig->cutHigh[ih]);
  }
  write();
  return 0;
}

int InterCalibEPN::process(const char* hname, int ic)
{
  // Run 2 ZDC calibration is based on multi-dimensional histograms
  // with dimensions:
  // TC, T1, T2, T3, T4, Trigger
  // ic is the number of the selected trigger class
  THnSparse* hs = (THnSparse*)gROOT->FindObject(hname);
  if (hs == 0) {
    LOGF(error, "Not found: %s\n", hname);
    return -1;
  }
  if (!hs->InheritsFrom(THnSparse::Class())) {
    LOGF(error, "Not a THnSparse: %s\n", hname);
    hs->IsA()->Print();
    return -1;
  }
  TString hn = hname;
  int ih = -1;
  if (hn.EqualTo("hZNA")) {
    ih = HidZNA;
  } else if (hn.EqualTo("hZPA")) {
    ih = HidZPA;
  } else if (hn.EqualTo("hZNC")) {
    ih = HidZNC;
  } else if (hn.EqualTo("hZPC")) {
    ih = HidZPC;
  } else if (hn.EqualTo("hZEM")) {
    ih = HidZEM;
  } else {
    LOGF(error, "Not recognized histogram name: %s\n", hname);
    return -1;
  }
  clear(ih);
  const int32_t dim = 6;
  double x[dim];
  int32_t bins[dim];
  int64_t nb = hs->GetNbins();
  int64_t nn = 0;
  LOGF(info, "Histogram %s has %ld bins\n", hname, nb);
  double cutl = mInterCalibConfig->cutLow[ih];
  double cuth = mInterCalibConfig->cutHigh[ih];
  double contt = 0;
  for (int64_t i = 0; i < nb; i++) {
    double cont = hs->GetBinContent(i, bins);
    if (cont <= 0) {
      continue;
    }
    for (int32_t d = 0; d < dim; ++d) {
      x[d] = hs->GetAxis(d)->GetBinCenter(bins[d]);
    }
    if (TMath::Nint(x[5] - ic) == 0 && x[0] > cutl && x[0] < cuth) {
      nn++;
      contt += cont;
      cumulate(ih, x[0], x[1], x[2], x[3], x[4], cont);
    }
  }
  LOGF(info, "Trigger class selection %d and %d bins %g events and cuts (%g:%g): %ld\n", ic, nn, contt, cutl, cuth);
  return 0;
}

void InterCalibEPN::clear(int ih)
{
  int ihstart = 0;
  int ihstop = NH;
  if (ih >= 0 && ih < NH) {
    ihstart = ih;
    ihstop = ih + 1;
  }
  for (int32_t ii = ihstart; ii < ihstop; ii++) {
    for (int32_t i = 0; i < NPAR; i++) {
      for (int32_t j = 0; j < NPAR; j++) {
        data.mSum[ii][i][j] = 0;
      }
    }
    if (mH[ii]) {
      mH[ii]->Reset();
    }
    if (mC[ii]) {
      mC[ii]->Reset();
    }
  }
}

void InterCalibEPN::cumulate(int ih, double tc, double t1, double t2, double t3, double t4, double w = 1)
{
  // TODO: add histogram
  // TODO: store data to redraw histograms
  if (tc < mInterCalibConfig->cutLow[ih] || tc > mInterCalibConfig->cutHigh[ih]) {
    return;
  }
  double val[NPAR] = {0, 0, 0, 0, 0, 1};
  val[0] = tc;
  val[1] = t1;
  val[2] = t2;
  val[3] = t3;
  val[4] = t4;
  for (int32_t i = 0; i < 6; i++) {
    for (int32_t j = i; j < 6; j++) {
      data.mSum[ih][i][j] += val[i] * val[j] * w;
    }
  }
  // data.mSum[ih][5][5] contains the number of analyzed events
  double sumquad = val[1] + val[2] + val[3] + val[4];
  mH[ih]->Fill(sumquad, w);
  mH[ih + NH]->Fill(val[0]);
  mC[ih]->Fill(val[0], sumquad, w);
}

int InterCalibEPN::write(const std::string fn)
{
  TDirectory* cwd = gDirectory;
  TFile* f = new TFile(fn.data(), "recreate");
  if (f->IsZombie()) {
    LOG(error) << "Cannot create file: " << fn;
    return 1;
  }
  for (int32_t ih = 0; ih < (2 * NH); ih++) {
    if (mH[ih]) {
      mH[ih]->Write("", TObject::kOverwrite);
    }
  }
  for (int32_t ih = 0; ih < NH; ih++) {
    if (mC[ih]) {
      mC[ih]->Write("", TObject::kOverwrite);
    }
  }
  f->Close();
  cwd->cd();
  return 0;
}
