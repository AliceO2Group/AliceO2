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
#include "ZDCCalib/InterCalib.h"
#include "ZDCReconstruction/ZDCEnergyParam.h"
#include "ZDCReconstruction/ZDCTowerParam.h"
#include "Framework/Logger.h"

using namespace o2::zdc;

double InterCalib::add[InterCalib::NPAR][InterCalib::NPAR] = {0};
std::mutex InterCalib::mtx;

int InterCalib::init()
{
  if (mInterCalibConfig == nullptr) {
    LOG(fatal) << "o2::zdc::InterCalib: missing configuration object";
  }
  clear();
  for (int32_t ih = 0; ih < (2 * NH); ih++) {
    if (h[ih]) {
      delete h[ih];
      h[ih] = nullptr;
    }
  }
  for (int32_t ih = 0; ih < NH; ih++) {
    if (hc[ih]) {
      delete hc[ih];
      hc[ih] = nullptr;
    }
  }
  int ih;
  // clang-format off
  ih = 0; h[ih] = new TH1F("hZNAS","ZNA sum",mInterCalibConfig->nb1[ih],mInterCalibConfig->amin1[ih],mInterCalibConfig->amax1[ih]);
  ih = 1; h[ih] = new TH1F("hZPAS","ZPA sum",mInterCalibConfig->nb1[ih],mInterCalibConfig->amin1[ih],mInterCalibConfig->amax1[ih]);
  ih = 2; h[ih] = new TH1F("hZNCS","ZNC sum",mInterCalibConfig->nb1[ih],mInterCalibConfig->amin1[ih],mInterCalibConfig->amax1[ih]);
  ih = 3; h[ih] = new TH1F("hZPCS","ZPC sum",mInterCalibConfig->nb1[ih],mInterCalibConfig->amin1[ih],mInterCalibConfig->amax1[ih]);
  ih = 4; h[ih] = new TH1F("hZEM2","ZEM2"   ,mInterCalibConfig->nb1[ih],mInterCalibConfig->amin1[ih],mInterCalibConfig->amax1[ih]);
  ih = 0; h[NH+ih] = new TH1F("hZNAC","ZNA TC",mInterCalibConfig->nb1[ih],mInterCalibConfig->amin1[ih],mInterCalibConfig->amax1[ih]);
  ih = 1; h[NH+ih] = new TH1F("hZPAC","ZPA TC",mInterCalibConfig->nb1[ih],mInterCalibConfig->amin1[ih],mInterCalibConfig->amax1[ih]);
  ih = 2; h[NH+ih] = new TH1F("hZNCC","ZNC TC",mInterCalibConfig->nb1[ih],mInterCalibConfig->amin1[ih],mInterCalibConfig->amax1[ih]);
  ih = 3; h[NH+ih] = new TH1F("hZPCC","ZPC TC",mInterCalibConfig->nb1[ih],mInterCalibConfig->amin1[ih],mInterCalibConfig->amax1[ih]);
  ih = 4; h[NH+ih] = new TH1F("hZEM1","ZEM1",  mInterCalibConfig->nb1[ih],mInterCalibConfig->amin1[ih],mInterCalibConfig->amax1[ih]);
  ih = 0; hc[ih] = new TH2F("cZNA","ZNA;TC;SUM",mInterCalibConfig->nb2[ih],mInterCalibConfig->amin2[ih],mInterCalibConfig->amax2[ih],mInterCalibConfig->nb2[ih],mInterCalibConfig->amin2[ih],mInterCalibConfig->amax2[ih]);
  ih = 1; hc[ih] = new TH2F("cZPA","ZPA;TC;SUM",mInterCalibConfig->nb2[ih],mInterCalibConfig->amin2[ih],mInterCalibConfig->amax2[ih],mInterCalibConfig->nb2[ih],mInterCalibConfig->amin2[ih],mInterCalibConfig->amax2[ih]);
  ih = 2; hc[ih] = new TH2F("cZNC","ZNC;TC;SUM",mInterCalibConfig->nb2[ih],mInterCalibConfig->amin2[ih],mInterCalibConfig->amax2[ih],mInterCalibConfig->nb2[ih],mInterCalibConfig->amin2[ih],mInterCalibConfig->amax2[ih]);
  ih = 3; hc[ih] = new TH2F("cZPC","ZPC;TC;SUM",mInterCalibConfig->nb2[ih],mInterCalibConfig->amin2[ih],mInterCalibConfig->amax2[ih],mInterCalibConfig->nb2[ih],mInterCalibConfig->amin2[ih],mInterCalibConfig->amax2[ih]);
  ih = 4; hc[ih] = new TH2F("cZEM","ZEM;ZEM1;ZEM2",mInterCalibConfig->nb2[ih],mInterCalibConfig->amin2[ih],mInterCalibConfig->amax2[ih],mInterCalibConfig->nb2[ih],mInterCalibConfig->amin2[ih],mInterCalibConfig->amax2[ih]);
  // clang-format on
  mInitDone = true;
  return 0;
}

int InterCalib::process(const gsl::span<const o2::zdc::BCRecData>& RecBC,
                        const gsl::span<const o2::zdc::ZDCEnergy>& Energy,
                        const gsl::span<const o2::zdc::ZDCTDCData>& TDCData,
                        const gsl::span<const uint16_t>& Info)
{
  if (!mInitDone) {
    init();
  }
  LOG(info) << "o2::zdc::InterCalib processing " << RecBC.size() << " b.c.";
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
      printf("%9u.%04u Untriggered bunch\n", ev.mCurB.ir.orbit, ev.mCurB.ir.bc);
      // Skip!
      continue;
    }
    if ((ev.ezdcDecoded & MaskZNA) == MaskZNA) {
      cumulate(hidZNA, ev.EZDC(IdZNAC), ev.EZDC(IdZNA1), ev.EZDC(IdZNA2), ev.EZDC(IdZNA3), ev.EZDC(IdZNA4), 1.);
    }
    if ((ev.ezdcDecoded & MaskZPA) == MaskZPA) {
      cumulate(hidZPA, ev.EZDC(IdZPAC), ev.EZDC(IdZPA1), ev.EZDC(IdZPA2), ev.EZDC(IdZPA3), ev.EZDC(IdZPA4), 1.);
    }
    if ((ev.ezdcDecoded & MaskZNC) == MaskZNC) {
      cumulate(hidZNC, ev.EZDC(IdZNCC), ev.EZDC(IdZNC1), ev.EZDC(IdZNC2), ev.EZDC(IdZNC3), ev.EZDC(IdZNC4), 1.);
    }
    if ((ev.ezdcDecoded & MaskZPC) == MaskZPC) {
      cumulate(hidZPC, ev.EZDC(IdZPCC), ev.EZDC(IdZPC1), ev.EZDC(IdZPC2), ev.EZDC(IdZPC3), ev.EZDC(IdZPC4), 1.);
    }
    if ((ev.ezdcDecoded & MaskZEM) == MaskZEM) {
      cumulate(hidZEM, ev.EZDC(IdZEM1), ev.EZDC(IdZEM2), 0., 0., 0., 1.);
    }
  }
  for (int ih = 0; ih < NH; ih++) {
    if (sum[ih][5][5] >= mInterCalibConfig->min_e[ih]) {
      int ierr = mini(ih);
      if (ierr) {
        printf("FAILED processing RUN3 data for ih = %d - ", ih);
      } else {
        printf("Processed RUN3 data for ih = %d: ", ih);
      }
    } else {
      printf("FAILED processing RUN3 data for ih = %d: TOO FEW EVENTS: ", ih);
    }
    printf("%g events and cuts (%g:%g)\n", sum[ih][5][5], mInterCalibConfig->cutLow[ih], mInterCalibConfig->cutHigh[ih]);
    return 0;
  }
  write();
  return 0;
}

int InterCalib::process(const char* hname, int ic)
{
  // Run 2 ZDC calibration is based on multi-dimensional histograms
  // with dimensions:
  // TC, T1, T2, T3, T4, Trigger
  // ic is the number of the selected trigger class
  THnSparse* hs = (THnSparse*)gROOT->FindObject(hname);
  if (hs == 0) {
    printf("Not found: %s\n", hname);
    return -1;
  }
  if (!hs->InheritsFrom(THnSparse::Class())) {
    printf("Not a THnSparse: %s\n", hname);
    hs->IsA()->Print();
    return -1;
  }
  TString hn = hname;
  int ih = -1;
  if (hn.EqualTo("hZNA")) {
    ih = hidZNA;
  } else if (hn.EqualTo("hZPA")) {
    ih = hidZPA;
  } else if (hn.EqualTo("hZNC")) {
    ih = hidZNC;
  } else if (hn.EqualTo("hZPC")) {
    ih = hidZPC;
  } else if (hn.EqualTo("hZEM")) {
    ih = hidZEM;
  } else {
    printf("Not recognized histogram name: %s\n", hname);
    return -1;
  }
  clear(ih);
  const int32_t dim = 6;
  double x[dim];
  int32_t bins[dim];
  int64_t nb = hs->GetNbins();
  int64_t nn = 0;
  printf("Histogram %s has %ld bins\n", hname, nb);
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
  int ierr = mini(ih);
  if (h_corr[ih]) {
    delete h_corr[ih];
    h_corr[ih] = nullptr;
  }
  if (hc_corr[ih]) {
    delete hc_corr[ih];
    hc_corr[ih] = nullptr;
  }
  if (ierr) {
    printf("FAILED processing RUN2 data for %s ih = %d\n", hname, ih);
  } else {
    printf("Processed RUN2 data for %s ih = %d\n", hname, ih);
    replay(ih, hs, ic);
  }
  printf("Trigger class selection %d and %d bins %g events and cuts (%g:%g): %ld\n", ic, nn, contt, cutl, cuth);
  return 0;
}

void InterCalib::replay(int ih, THnSparse* hs, int ic)
{
  auto* cfg = mInterCalibConfig;
  // clang-format off
  if(ih == 0)h_corr[ih] = new TH1F("hZNASc","ZNA sum corr",cfg->nb1[ih],cfg->amin1[ih],cfg->amax1[ih]);
  if(ih == 1)h_corr[ih] = new TH1F("hZPASc","ZPA sum corr",cfg->nb1[ih],cfg->amin1[ih],cfg->amax1[ih]);
  if(ih == 2)h_corr[ih] = new TH1F("hZNCSc","ZNC sum corr",cfg->nb1[ih],cfg->amin1[ih],cfg->amax1[ih]);
  if(ih == 3)h_corr[ih] = new TH1F("hZPCSc","ZPC sum corr",cfg->nb1[ih],cfg->amin1[ih],cfg->amax1[ih]);
  if(ih == 4)h_corr[ih] = new TH1F("hZEM2c","ZEM2"   ,cfg->nb1[ih],cfg->amin1[ih],cfg->amax1[ih]);
  if(ih == 0)hc_corr[ih] = new TH2F("cZNAc","ZNA;TC;SUM corr",cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih],cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih]);
  if(ih == 1)hc_corr[ih] = new TH2F("cZPAc","ZPA;TC;SUM corr",cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih],cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih]);
  if(ih == 2)hc_corr[ih] = new TH2F("cZNCc","ZNC;TC;SUM corr",cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih],cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih]);
  if(ih == 3)hc_corr[ih] = new TH2F("cZPCc","ZPC;TC;SUM corr",cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih],cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih]);
  if(ih == 4)hc_corr[ih] = new TH2F("cZEMc","ZEM;ZEM1;ZEM2 corr",cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih],cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih]);
  // clang-format on
  const int32_t dim = 6;
  double x[dim];
  int32_t bins[dim];
  int64_t nb = hs->GetNbins();
  double cutl = cfg->cutLow[ih];
  double cuth = cfg->cutHigh[ih];
  double c1 = par[ih][1];
  double c2 = par[ih][2];
  double c3 = par[ih][3];
  double c4 = par[ih][4];
  double of = par[ih][5];
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
      h_corr[ih]->Fill(sumquad, cont);
      hc_corr[ih]->Fill(x[0], sumquad, cont);
    }
  }
}

void InterCalib::clear(int ih)
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
        sum[ii][i][j] = 0;
      }
    }
    if (h[ii]) {
      h[ii]->Reset();
    }
    if (hc[ii]) {
      hc[ii]->Reset();
    }
  }
}

void InterCalib::cumulate(int ih, double tc, double t1, double t2, double t3, double t4, double w = 1)
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
      sum[ih][i][j] += val[i] * val[j] * w;
    }
  }
  // sum[ih][5][5] contains the number of analyzed events
  double sumquad = val[1] + val[2] + val[3] + val[4];
  h[ih]->Fill(sumquad, w);
  h[ih + NH]->Fill(val[0]);
  hc[ih]->Fill(val[0], sumquad, w);
}

void InterCalib::fcn(int& npar, double* gin, double& chi, double* par, int iflag)
{
  chi = 0;
  for (int32_t i = 0; i < NPAR; i++) {
    for (int32_t j = 0; j < NPAR; j++) {
      chi += (i == 0 ? par[i] : -par[i]) * (j == 0 ? par[j] : -par[j]) * add[i][j];
    }
  }
  // printf("%g\n", chi);
}

int InterCalib::mini(int ih)
{
  // Copy to static object and symmetrize matrix
  // We use a static function and therefore we can do only one minimization at a time
  mtx.lock();
  for (int32_t i = 0; i < NPAR; i++) {
    for (int32_t j = 0; j < NPAR; j++) {
      if (j < i) {
        add[i][j] = sum[ih][j][i];
      } else {
        add[i][j] = sum[ih][i][j];
      }
    }
  }
  double arglist[10];
  int ierflg = 0;
  double l_bnd = mInterCalibConfig->l_bnd[ih];
  double u_bnd = mInterCalibConfig->u_bnd[ih];
  double start = 1.0;
  double step = 0.1;
  if (mn[ih] != nullptr) {
    delete mn[ih];
  }
  mn[ih] = new TMinuit(NPAR);
  mn[ih]->SetFCN(fcn);
  mn[ih]->mnparm(0, "c0", 1., 0., 1., 1., ierflg);
  mn[ih]->mnparm(1, "c1", start, step, l_bnd, u_bnd, ierflg);
  if (ih == 4) {
    // Only two ZEM calorimeters: equalize response
    l_bnd = 0;
    u_bnd = 0;
    start = 0;
    step = 0;
  }
  mn[ih]->mnparm(2, "c2", start, step, l_bnd, u_bnd, ierflg);
  mn[ih]->mnparm(3, "c3", start, step, l_bnd, u_bnd, ierflg);
  mn[ih]->mnparm(4, "c4", start, step, l_bnd, u_bnd, ierflg);
  l_bnd = mInterCalibConfig->l_bnd_o[ih];
  u_bnd = mInterCalibConfig->u_bnd_o[ih];
  start = 0;
  step = mInterCalibConfig->step_o[ih];
  mn[ih]->mnparm(5, "offset", start, step, l_bnd, u_bnd, ierflg);
  mn[ih]->mnexcm("MIGRAD", arglist, 0, ierflg);
  for (Int_t i = 0; i < NPAR; i++) {
    mn[ih]->GetParameter(i, par[ih][i], err[ih][i]);
  }
  mtx.unlock();
  return ierflg;
}

int InterCalib::write(const std::string fn)
{
  TDirectory* cwd = gDirectory;
  TFile* f = new TFile(fn.data(), "recreate");
  if (f->IsZombie()) {
    LOG(error) << "Cannot create file: " << fn;
    return 1;
  }
  for (int32_t ih = 0; ih < (2 * NH); ih++) {
    if (h[ih]) {
      h[ih]->Write("", TObject::kOverwrite);
    }
  }
  for (int32_t ih = 0; ih < NH; ih++) {
    if (hc[ih]) {
      hc[ih]->Write("", TObject::kOverwrite);
    }
  }
  const char* mntit[NH] = {"mZNA", "mZPA", "mZNC", "mZPC", "mZEM"};
  for (int32_t ih = 0; ih < NH; ih++) {
    if (mn[ih]) {
      mn[ih]->Write(mntit[ih], TObject::kOverwrite);
    }
  }
  f->Close();
  cwd->cd();
  return 0;
}
