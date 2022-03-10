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
#include <TStyle.h>
#include <TPaveStats.h>
#include <TMinuit.h>
#include <THnBase.h>
#include <THnSparse.h>
#include <TAxis.h>
#include "ZDCCalib/InterCalib.h"
#include "Framework/Logger.h"

using namespace o2::zdc;

double InterCalib::add[InterCalib::NPAR][InterCalib::NPAR] = {0};
std::mutex InterCalib::mtx;

int InterCalib::init()
{
  mCutLow = -std::numeric_limits<float>::infinity();
  mCutHigh = std::numeric_limits<float>::infinity();
  return 0;
}

int InterCalib::process(const gsl::span<const o2::zdc::BCRecData>& bcrec,
              const gsl::span<const o2::zdc::ZDCEnergy>& energy,
              const gsl::span<const o2::zdc::ZDCTDCData>& tdc,
              const gsl::span<const uint16_t>& info)
{
  /*
  o2::zdc::RecEventFlat ev;
  ev.init(RecBCPtr, EnergyPtr, TDCDataPtr, InfoPtr);
  while (ev.next()) {
    int printed = 0;
    if (ev.getNInfo() > 0) {
      auto& decodedInfo = ev.getDecodedInfo();
      for (uint16_t info : decodedInfo) {
        uint8_t ch = (info >> 10) & 0x1f;
        uint16_t code = info & 0x03ff;
        ;
        hmsg->Fill(ch, code);
      }
      ev.print();
      printed = 1;
    }
    if (ev.getNEnergy() > 0 && ev.mCurB.triggers == 0) {
      printf("%9u.%04u Untriggered bunch\n", ev.mCurB.ir.orbit, ev.mCurB.ir.bc);
      if (printed == 0) {
        ev.print();
      }
    }
    heznac->Fill(ev.EZNAC());
    auto tdcid = o2::zdc::TDCZNAC;
    auto nhit = ev.NtdcV(tdcid);
    if (ev.NtdcA(tdcid) != nhit) {
      fprintf(stderr, "Mismatch in TDC data\n");
      continue;
    }
    if (nhit > 0) {
      double bc_d = uint32_t(ev.ir.bc / 100);
      double bc_m = uint32_t(ev.ir.bc % 100);
      hbznac->Fill(bc_m, -bc_d);
      for (int ihit = 0; ihit < nhit; ihit++) {
        htznac->Fill(ev.tdcV(tdcid, ihit), ev.tdcA(tdcid, ihit));
      }
    }
  }
  */
  return 0;
}

int InterCalib::process(const char* hname, int ic)
{
  clear();
  // Run 2 ZDC calibration is based on multi-dimensional histograms
  // with dimensions:
  // TC, T1, T2, T3, T4, Trigger
  THnSparse* hs = (THnSparse*) gROOT->FindObject(hname);
  if (hs == 0) {
    return -1;
  }
  if(! hs->InheritsFrom(THnSparse::Class())){
    printf("Not a THnSparse\n");
    hs->IsA()->Print();
    return -1;
  }
  const int32_t dim = 6;
  double x[dim];
  int32_t bins[dim];
  int64_t nb = hs->GetNbins();
  int64_t nn = 0;
  printf("Histogram %s has %ld bins\n", hname, nb);
  for (int64_t i = 0; i < nb; i++) {
    double cont = hs->GetBinContent(i, bins);
    if (cont <= 0){
      continue;
    }
    for (int32_t d = 0; d < dim; ++d) {
      x[d] = hs->GetAxis(d)->GetBinCenter(bins[d]);
    }
    if (TMath::Nint(x[5] - ic) == 0 && x[0] > mCutLow && x[0] < mCutHigh) {
      nn++;
      cumulate(x[0], x[1], x[2], x[3], x[4], cont);
    }
  }
  printf("Passing trigger class selection %d and %d cuts (%g:%g): %ld\n", ic, nn);
  mini();
  return 0;
}

void InterCalib::clear()
{
  for (int32_t i = 0; i < NPAR; i++) {
    for (int32_t j = 0; j < NPAR; j++) {
      sum[i][j] = 0;
    }
  }
}

void InterCalib::cumulate(double tc, double t1, double t2, double t3, double t4, double w = 1)
{
  // TODO: add cuts
  // TODO: add histogram
  // TODO: store data to redraw histograms
  double val[NPAR] = {0, 0, 0, 0, 0, 1};
  val[0] = tc;
  val[1] = t1;
  val[2] = t2;
  val[3] = t3;
  val[4] = t4;
  for (int32_t i = 0; i < 6; i++) {
    for (int32_t j = i; j < 6; j++) {
      sum[i][j] += val[i] * val[j] * w;
    }
  }
}

void InterCalib::fcn(int& npar, double* gin, double& chi, double* par, int iflag)
{
  chi = 0;
  for (int32_t i = 0; i < NPAR; i++) {
    for (int32_t j = 0; j < NPAR; j++) {
      chi += (i == 0 ? par[i] : -par[i]) * (j == 0 ? par[j] : -par[j]) * add[i][j];
    }
  }
  printf("%g\n", chi);
}

int InterCalib::mini()
{
  // Copy to static object and symmetrize
  mtx.lock();
  for (int32_t i = 0; i < NPAR; i++) {
    for (int32_t j = 0; j < NPAR; j++) {
      if (j < i) {
        add[i][j] = sum[j][i];
      } else {
        add[i][j] = sum[i][j];
      }
    }
  }
  double arglist[10];
  int ierflg = 0;
  double start = 1.0;
  double step = 0.1;
  double l_bnd = 0.2;
  double u_bnd = 5.;
  TMinuit minuit(NPAR);
  minuit.SetFCN(fcn);
  minuit.mnparm(0, "c0", 1., 0., 1., 1., ierflg);
  minuit.mnparm(1, "c1", start, step, l_bnd, u_bnd, ierflg);
  minuit.mnparm(2, "c2", start, step, l_bnd, u_bnd, ierflg);
  minuit.mnparm(3, "c3", start, step, l_bnd, u_bnd, ierflg);
  minuit.mnparm(4, "c4", start, step, l_bnd, u_bnd, ierflg);
  start = 0;
  step = 1;
  l_bnd = -20;
  u_bnd = 20;
  step = 0;
  minuit.mnparm(5, "offset", start, step, l_bnd, u_bnd, ierflg);
  minuit.mnexcm("MIGRAD", arglist, 0, ierflg);
  for (Int_t i = 0; i < NPAR; i++) {
    minuit.GetParameter(i, par[i], err[i]);
  }
  mtx.unlock();
  return 0;
}
