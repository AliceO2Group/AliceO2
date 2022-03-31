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
#include "ZDCCalib/TDCCalib.h"
#include "ZDCReconstruction/ZDCTDCParam.h"
#include "Framework/Logger.h"

using namespace o2::zdc;

std::mutex TDCCalib::mMtx;

int TDCCalib::init()
{
  if (mTDCCalibConfig == nullptr) {
    LOG(fatal) << "o2::zdc::TDCCalib: missing configuration object";
    return -1;
  }
  clear();
  for (int ih = 0; ih < NTDCChannels; ih++) {
    std::string hn = fmt::format("h{}", ChannelNames[TDCSignal[ih]]);
    std::string ht = fmt::format("TDC {};(ch)", ChannelNames[TDCSignal[ih]]);
    mHTDC[ih] = std::make_unique<TH1F>(hn.data(), ht.data(), mTDCCalibConfig->nb[ih], mTDCCalibConfig->amin[ih], mTDCCalibConfig->amax[ih]);
  }
  mInitDone = true;
  return 0;
}

int TDCCalib::process(const gsl::span<const o2::zdc::BCRecData>& RecBC,
                      const gsl::span<const o2::zdc::ZDCEnergy>& Energy,
                      const gsl::span<const o2::zdc::ZDCTDCData>& TDCData,
                      const gsl::span<const uint16_t>& Info)
{
  if (!mInitDone) {
    init();
  }
  LOG(info) << "o2::zdc::TDCCalib processing " << RecBC.size() << " b.c.";
  o2::zdc::RecEventFlat ev;
  ev.init(RecBC, Energy, TDCData, Info);
  while (ev.next()) {
    if (ev.getNInfo() > 0) {
      auto& decodedInfo = ev.getDecodedInfo();
      //       for (uint16_t info : decodedInfo) {
      //         uint8_t ch = (info >> 10) & 0x1f;
      //         uint16_t code = info & 0x03ff;
      //       }
      ev.print();
      // Need clean data (no messages)
      // We are sure there is no pile-up in any channel (too restrictive?)
      continue;
    }
    // TDC
    for (int32_t itdc = 0; itdc < o2::zdc::NTDCChannels; itdc++) {
      int nhit = ev.NtdcV(itdc);
      if (nhit == 1) {
        // TDC without in-bunch pile-up
        int ich = o2::zdc::TDCSignal[itdc];
        if (ev.genericE[ich] || ev.tdcPileEvC[ich] || ev.tdcPileEvE[ich] ||
            ev.tdcPileM1C[ich] || ev.tdcPileM1E[ich] || ev.tdcPileM2C[ich] || ev.tdcPileM2E[ich] || ev.tdcPileM3C[ich] || ev.tdcPileM3E[ich] ||
            ev.tdcSigE[ich]) {
          continue;
        }
        if (ev.NtdcA(itdc) != nhit) {
          LOGF(error, "Mismatch in TDC %d data Val=%d Amp=%d", itdc, ev.NtdcV(itdc), ev.NtdcA(ich));
          continue;
        }
        auto amp = ev.TDCAmp[itdc][0];
        if (amp < mTDCCalibConfig.cutLow[itdc] || mTDCCalibConfig > cutHigh[itdc]) {
          continue;
        }
        mHTDC[itdc]->Fill(ev.TDCVal[itdc][0]);
      }
    }
  }
  return 0;
}

int TDCCalib::endOfRun()
{
  for (int ih = 0; ih < NH; ih++) {
    if (mSum[ih][5][5] >= mTDCCalibConfig->min_e[ih]) {
      int ierr = mini(ih);
      if (ierr) {
        LOGF(error, "FAILED processing RUN3 data for ih = %d - ", ih);
      } else {
        LOGF(info, "Processed RUN3 data for ih = %d: ", ih);
      }
    } else {
      LOGF(info, "FAILED processing RUN3 data for ih = %d: TOO FEW EVENTS: ", ih);
    }
    LOGF(info, "%g events and cuts (%g:%g)\n", mSum[ih][5][5], mTDCCalibConfig->cutLow[ih], mTDCCalibConfig->cutHigh[ih]);
  }
  write();
  return 0;
}

int TDCCalib::process(const char* hname, int ic)
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
  double cutl = mTDCCalibConfig->cutLow[ih];
  double cuth = mTDCCalibConfig->cutHigh[ih];
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
  int ierr = mini(ih);
  if (ierr) {
    LOGF(error, "FAILED processing RUN2 data for %s ih = %d\n", hname, ih);
  } else {
    LOGF(info, "Processed RUN2 data for %s ih = %d\n", hname, ih);
    replay(ih, hs, ic);
  }
  LOGF(info, "Trigger class selection %d and %d bins %g events and cuts (%g:%g): %ld\n", ic, nn, contt, cutl, cuth);
  return 0;
}

void TDCCalib::replay(int ih, THnSparse* hs, int ic)
{
  auto* cfg = mTDCCalibConfig;
  // clang-format off
  if(ih == 0)mHCorr[ih] = std::make_unique<TH1F>("hZNASc","ZNA sum corr",cfg->nb1[ih],cfg->amin1[ih],cfg->amax1[ih]);
  if(ih == 1)mHCorr[ih] = std::make_unique<TH1F>("hZPASc","ZPA sum corr",cfg->nb1[ih],cfg->amin1[ih],cfg->amax1[ih]);
  if(ih == 2)mHCorr[ih] = std::make_unique<TH1F>("hZNCSc","ZNC sum corr",cfg->nb1[ih],cfg->amin1[ih],cfg->amax1[ih]);
  if(ih == 3)mHCorr[ih] = std::make_unique<TH1F>("hZPCSc","ZPC sum corr",cfg->nb1[ih],cfg->amin1[ih],cfg->amax1[ih]);
  if(ih == 4)mHCorr[ih] = std::make_unique<TH1F>("hZEM2c","ZEM2"   ,cfg->nb1[ih],cfg->amin1[ih],cfg->amax1[ih]);
  if(ih == 0)mCCorr[ih] = std::make_unique<TH2F>("cZNAc","ZNA;TC;SUM corr",cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih],cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih]);
  if(ih == 1)mCCorr[ih] = std::make_unique<TH2F>("cZPAc","ZPA;TC;SUM corr",cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih],cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih]);
  if(ih == 2)mCCorr[ih] = std::make_unique<TH2F>("cZNCc","ZNC;TC;SUM corr",cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih],cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih]);
  if(ih == 3)mCCorr[ih] = std::make_unique<TH2F>("cZPCc","ZPC;TC;SUM corr",cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih],cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih]);
  if(ih == 4)mCCorr[ih] = std::make_unique<TH2F>("cZEMc","ZEM;ZEM1;ZEM2 corr",cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih],cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih]);
  // clang-format on
  const int32_t dim = 6;
  double x[dim];
  int32_t bins[dim];
  int64_t nb = hs->GetNbins();
  double cutl = cfg->cutLow[ih];
  double cuth = cfg->cutHigh[ih];
  double c1 = mPar[ih][1];
  double c2 = mPar[ih][2];
  double c3 = mPar[ih][3];
  double c4 = mPar[ih][4];
  double of = mPar[ih][5];
  for (int64_t i = 0; i < nb; i++) {
    double cont = hs->GetBinContent(i, bins);
    if (cont <= 0) {
      continue;
    }
    for (int32_t d = 0; d < dim; ++d) {
      x[d] = hs->GetAxis(d)->GetBinCenter(bins[d]);
    }
    if (TMath::Nint(x[5] - ic) == 0 && x[0] > cutl && x[0] < cuth) {
      double sumquad = c1 * x[1] + c2 * x[2] + c3 * x[3] + c4 * x[4] + of;
      mHCorr[ih]->Fill(sumquad, cont);
      mCCorr[ih]->Fill(x[0], sumquad, cont);
    }
  }
}

void TDCCalib::clear(int ih)
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
        mSum[ii][i][j] = 0;
      }
    }
    if (mHUnc[ii]) {
      mHUnc[ii]->Reset();
    }
    if (mCUnc[ii]) {
      mCUnc[ii]->Reset();
    }
  }
}

void TDCCalib::cumulate(int ih, double tc, double t1, double t2, double t3, double t4, double w = 1)
{
  // TODO: add histogram
  // TODO: store data to redraw histograms
  if (tc < mTDCCalibConfig->cutLow[ih] || tc > mTDCCalibConfig->cutHigh[ih]) {
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
      mSum[ih][i][j] += val[i] * val[j] * w;
    }
  }
  // mSum[ih][5][5] contains the number of analyzed events
  double sumquad = val[1] + val[2] + val[3] + val[4];
  mHUnc[ih]->Fill(sumquad, w);
  mHUnc[ih + NH]->Fill(val[0]);
  mCUnc[ih]->Fill(val[0], sumquad, w);
}

void TDCCalib::fcn(int& npar, double* gin, double& chi, double* par, int iflag)
{
  chi = 0;
  for (int32_t i = 0; i < NPAR; i++) {
    for (int32_t j = 0; j < NPAR; j++) {
      chi += (i == 0 ? par[i] : -par[i]) * (j == 0 ? par[j] : -par[j]) * mAdd[i][j];
    }
  }
}

int TDCCalib::mini(int ih)
{
  // Copy to static object and symmetrize matrix
  // We use a static function and therefore we can do only one minimization at a time
  mMtx.lock();
  for (int32_t i = 0; i < NPAR; i++) {
    for (int32_t j = 0; j < NPAR; j++) {
      if (j < i) {
        mAdd[i][j] = mSum[ih][j][i];
      } else {
        mAdd[i][j] = mSum[ih][i][j];
      }
    }
  }
  double arglist[10];
  int ierflg = 0;
  double l_bnd = mTDCCalibConfig->l_bnd[ih];
  double u_bnd = mTDCCalibConfig->u_bnd[ih];
  double start = 1.0;
  double step = 0.1;
  mMn[ih] = std::make_unique<TMinuit>(NPAR);
  mMn[ih]->SetFCN(fcn);
  mMn[ih]->mnparm(0, "c0", 1., 0., 1., 1., ierflg);
  mMn[ih]->mnparm(1, "c1", start, step, l_bnd, u_bnd, ierflg);
  if (ih == 4) {
    // Only two ZEM calorimeters: equalize response
    l_bnd = 0;
    u_bnd = 0;
    start = 0;
    step = 0;
  }
  mMn[ih]->mnparm(2, "c2", start, step, l_bnd, u_bnd, ierflg);
  mMn[ih]->mnparm(3, "c3", start, step, l_bnd, u_bnd, ierflg);
  mMn[ih]->mnparm(4, "c4", start, step, l_bnd, u_bnd, ierflg);
  l_bnd = mTDCCalibConfig->l_bnd_o[ih];
  u_bnd = mTDCCalibConfig->u_bnd_o[ih];
  start = 0;
  step = mTDCCalibConfig->step_o[ih];
  mMn[ih]->mnparm(5, "offset", start, step, l_bnd, u_bnd, ierflg);
  mMn[ih]->mnexcm("MIGRAD", arglist, 0, ierflg);
  for (Int_t i = 0; i < NPAR; i++) {
    mMn[ih]->GetParameter(i, mPar[ih][i], mErr[ih][i]);
  }
  mMtx.unlock();
  return ierflg;
}

int TDCCalib::write(const std::string fn)
{
  TDirectory* cwd = gDirectory;
  TFile* f = new TFile(fn.data(), "recreate");
  if (f->IsZombie()) {
    LOG(error) << "Cannot create file: " << fn;
    return 1;
  }
  for (int32_t ih = 0; ih < (2 * NH); ih++) {
    if (mHUnc[ih]) {
      mHUnc[ih]->Write("", TObject::kOverwrite);
    }
  }
  for (int32_t ih = 0; ih < NH; ih++) {
    if (mCUnc[ih]) {
      mCUnc[ih]->Write("", TObject::kOverwrite);
    }
  }
  const char* mntit[NH] = {"mZNA", "mZPA", "mZNC", "mZPC", "mZEM"};
  for (int32_t ih = 0; ih < NH; ih++) {
    if (mMn[ih]) {
      mMn[ih]->Write(mntit[ih], TObject::kOverwrite);
    }
  }
  f->Close();
  cwd->cd();
  return 0;
}
