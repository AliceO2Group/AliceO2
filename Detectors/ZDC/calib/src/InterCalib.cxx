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
#include "CommonUtils/MemFileHelper.h"
#include "ZDCCalib/InterCalibData.h"
#include "ZDCCalib/InterCalib.h"
#include "ZDCReconstruction/ZDCEnergyParam.h"
#include "ZDCReconstruction/ZDCTowerParam.h"
#include "Framework/Logger.h"
#include "CCDB/CcdbApi.h"

using namespace o2::zdc;

double InterCalib::mAdd[InterCalib::NPAR][InterCalib::NPAR] = {0};
std::mutex InterCalib::mMtx;

int InterCalib::init()
{
  if (mInterCalibConfig == nullptr) {
    LOG(fatal) << "o2::zdc::InterCalib: missing configuration object";
    return -1;
  }
  clear();
  auto* cfg = mInterCalibConfig;
  int ih;
  // clang-format off
  ih = 0; mHUnc[ih] =    new o2::dataformats::FlatHisto1D<float>(cfg->nb1[ih],cfg->amin1[ih],cfg->amax1[ih]);
  ih = 1; mHUnc[ih] =    new o2::dataformats::FlatHisto1D<float>(cfg->nb1[ih],cfg->amin1[ih],cfg->amax1[ih]);
  ih = 2; mHUnc[ih] =    new o2::dataformats::FlatHisto1D<float>(cfg->nb1[ih],cfg->amin1[ih],cfg->amax1[ih]);
  ih = 3; mHUnc[ih] =    new o2::dataformats::FlatHisto1D<float>(cfg->nb1[ih],cfg->amin1[ih],cfg->amax1[ih]);
  ih = 4; mHUnc[ih] =    new o2::dataformats::FlatHisto1D<float>(cfg->nb1[ih],cfg->amin1[ih],cfg->amax1[ih]);
  ih = 0; mHUnc[NH+ih] = new o2::dataformats::FlatHisto1D<float>(cfg->nb1[ih],cfg->amin1[ih],cfg->amax1[ih]);
  ih = 1; mHUnc[NH+ih] = new o2::dataformats::FlatHisto1D<float>(cfg->nb1[ih],cfg->amin1[ih],cfg->amax1[ih]);
  ih = 2; mHUnc[NH+ih] = new o2::dataformats::FlatHisto1D<float>(cfg->nb1[ih],cfg->amin1[ih],cfg->amax1[ih]);
  ih = 3; mHUnc[NH+ih] = new o2::dataformats::FlatHisto1D<float>(cfg->nb1[ih],cfg->amin1[ih],cfg->amax1[ih]);
  ih = 4; mHUnc[NH+ih] = new o2::dataformats::FlatHisto1D<float>(cfg->nb1[ih],cfg->amin1[ih],cfg->amax1[ih]);
  ih = 0; mCUnc[ih] =    new o2::dataformats::FlatHisto2D<float>(cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih],cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih]);
  ih = 1; mCUnc[ih] =    new o2::dataformats::FlatHisto2D<float>(cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih],cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih]);
  ih = 2; mCUnc[ih] =    new o2::dataformats::FlatHisto2D<float>(cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih],cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih]);
  ih = 3; mCUnc[ih] =    new o2::dataformats::FlatHisto2D<float>(cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih],cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih]);
  ih = 4; mCUnc[ih] =    new o2::dataformats::FlatHisto2D<float>(cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih],cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih]);
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

//______________________________________________________________________________
// Update calibration coefficients
int InterCalib::endOfRun()
{
  if (mVerbosity > DbgZero) {
    LOGF(info, "Computing intercalibration coefficients");
  }
  for (int ih = 0; ih < NH; ih++) {
    LOGF(info, "%s %g events and cuts (%g:%g)", InterCalibData::DN[ih], mData.mSum[ih][5][5], mInterCalibConfig->cutLow[ih], mInterCalibConfig->cutHigh[ih]);
    if (!mInterCalibConfig->enabled[ih]) {
      LOGF(info, "DISABLED processing of RUN3 data for ih = %d: %s", ih, InterCalibData::DN[ih]);
      assign(ih, false);
    } else if (mData.mSum[ih][5][5] >= mInterCalibConfig->min_e[ih]) {
      int ierr = mini(ih);
      if (ierr) {
        LOGF(error, "FAILED processing RUN3 data for ih = %d: %s", ih, InterCalibData::DN[ih]);
        assign(ih, false);
      } else {
        LOGF(info, "Processed RUN3 data for ih = %d: %s", ih, InterCalibData::DN[ih]);
        assign(ih, true);
      }
    } else {
      LOGF(info, "FAILED processing RUN3 data for ih = %d: %s: TOO FEW EVENTS: %g", ih, InterCalibData::DN[ih], mData.mSum[ih][5][5]);
      assign(ih, false);
    }
  }

  auto clName = o2::utils::MemFileHelper::getClassName(mTowerParamUpd);
  mInfo.setObjectType(clName);
  auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
  mInfo.setFileName(flName);
  mInfo.setPath(CCDBPathTowerCalib);
  std::map<std::string, std::string> md;
  md["config"] = mInterCalibConfig->desc;
  mInfo.setMetaData(md);
  uint64_t starting = mData.mCTimeBeg;
  if (starting >= 10000) {
    starting = starting - 10000; // start 10 seconds before
  }
  uint64_t stopping = mData.mCTimeEnd + 10000; // stop 10 seconds after
  mInfo.setStartValidityTimestamp(starting);
  mInfo.setEndValidityTimestamp(stopping);
  mInfo.setAdjustableEOV();

  if (mSaveDebugHistos) {
    write();
  }
  return 0;
}

//______________________________________________________________________________
// Update calibration object for the five detectors
// ismod=false if it was not possible to update the calibration coefficients
// due to low statistics or minimization error
// ismod=true if the calibration was updated
void InterCalib::assign(int ih, bool ismod)
{
  int id_0[4] = {IdZNA1, IdZNA2, IdZNA3, IdZNA4};
  int id_1[4] = {IdZPA1, IdZPA2, IdZPA3, IdZPA4};
  int id_2[4] = {IdZNC1, IdZNC2, IdZNC3, IdZNC4};
  int id_3[4] = {IdZPC1, IdZPC2, IdZPC3, IdZPC4};
  int id_4[1] = {IdZEM2};
  int nid = 0;
  int* id = nullptr;
  if (ih == 0) {
    nid = 4;
    id = id_0;
  } else if (ih == 1) {
    nid = 4;
    id = id_1;
  } else if (ih == 2) {
    nid = 4;
    id = id_2;
  } else if (ih == 3) {
    nid = 4;
    id = id_3;
  } else if (ih == 4) {
    nid = 1;
    id = id_4;
  } else {
    LOG(fatal) << "InterCalib::assign accessing not existing ih = " << ih;
  }
  for (int iid = 0; iid < nid; iid++) {
    auto ich = id[iid];
    auto oldval = mTowerParam->getTowerCalib(ich);
    if (ismod == true) {
      auto val = oldval;
      if (oldval > 0) {
        val = val * mPar[ih][iid + 1];
      }
      if (mVerbosity > DbgZero) {
        LOGF(info, "%s updated %8.6f -> %8.6f", ChannelNames[ich].data(), oldval, val);
      }
      mTowerParamUpd.setTowerCalib(ich, val, true);
    } else {
      if (mVerbosity > DbgZero) {
        LOGF(info, "%s NOT CHANGED %8.6f", ChannelNames[ich].data(), oldval);
      }
      mTowerParamUpd.setTowerCalib(ich, oldval, false);
    }
  }
}

int InterCalib::process(const char* hname, int ic)
{
  // Run 2 ZDC calibration is based on multi-dimensional histograms
  // with dimensions:
  // TC, T1, T2, T3, T4, Trigger
  // ic is the number of the selected trigger class
  THnSparse* hs = (THnSparse*)gROOT->FindObject(hname);
  if (hs == nullptr) {
    LOGF(error, "Not found: %s\n", hname);
    return -1;
  }
  if (!hs->InheritsFrom(THnSparse::Class())) {
    LOGF(error, "Not a THnSparse: %s\n", hname);
    hs->IsA()->Print();
    return -1;
  }
  if (!mInitDone) {
    init();
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

void InterCalib::replay(int ih, THnSparse* hs, int ic)
{
  auto* cfg = mInterCalibConfig;
  // clang-format off
  if(ih == 0){ mHCorr[ih] = std::make_unique<TH1F>("hZNASc","ZNA sum corr",cfg->nb1[ih],cfg->amin1[ih],cfg->amax1[ih]); }
  if(ih == 1){ mHCorr[ih] = std::make_unique<TH1F>("hZPASc","ZPA sum corr",cfg->nb1[ih],cfg->amin1[ih],cfg->amax1[ih]); }
  if(ih == 2){ mHCorr[ih] = std::make_unique<TH1F>("hZNCSc","ZNC sum corr",cfg->nb1[ih],cfg->amin1[ih],cfg->amax1[ih]); }
  if(ih == 3){ mHCorr[ih] = std::make_unique<TH1F>("hZPCSc","ZPC sum corr",cfg->nb1[ih],cfg->amin1[ih],cfg->amax1[ih]); }
  if(ih == 4){ mHCorr[ih] = std::make_unique<TH1F>("hZEM2c","ZEM2"   ,cfg->nb1[ih],cfg->amin1[ih],cfg->amax1[ih]); }
  if(ih == 0){ mCCorr[ih] = std::make_unique<TH2F>("cZNAc","ZNA;TC;SUM corr",cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih],cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih]); }
  if(ih == 1){ mCCorr[ih] = std::make_unique<TH2F>("cZPAc","ZPA;TC;SUM corr",cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih],cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih]); }
  if(ih == 2){ mCCorr[ih] = std::make_unique<TH2F>("cZNCc","ZNC;TC;SUM corr",cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih],cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih]); }
  if(ih == 3){ mCCorr[ih] = std::make_unique<TH2F>("cZPCc","ZPC;TC;SUM corr",cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih],cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih]); }
  if(ih == 4){ mCCorr[ih] = std::make_unique<TH2F>("cZEMc","ZEM;ZEM1;ZEM2 corr",cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih],cfg->nb2[ih],cfg->amin2[ih],cfg->amax2[ih]); }
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
        mData.mSum[ii][i][j] = 0;
      }
    }
    if (mHUnc[ii]) {
      mHUnc[ii]->clear();
    }
    if (mCUnc[ii]) {
      mCUnc[ii]->clear();
    }
  }
}

int InterCalib::process(const InterCalibData& data)
{
  if (!mInitDone) {
    init();
  }
  mData += data;
  return 0;
}

void InterCalib::add(int ih, o2::dataformats::FlatHisto1D<float>& h1)
{
  if (!mInitDone) {
    init();
  }
  constexpr int nh = 2 * InterCalibData::NH;
  if (ih >= 0 && ih < nh) {
    mHUnc[ih]->add(h1);
  } else {
    LOG(error) << "InterCalib::add: unsupported FlatHisto1D " << ih;
  }
}

void InterCalib::add(int ih, o2::dataformats::FlatHisto2D<float>& h2)
{
  if (!mInitDone) {
    init();
  }
  if (ih >= 0 && ih < InterCalibData::NH) {
    mCUnc[ih]->add(h2);
  } else {
    LOG(error) << "InterCalib::add: unsupported FlatHisto2D " << ih;
  }
}

void InterCalib::cumulate(int ih, double tc, double t1, double t2, double t3, double t4, double w = 1)
{
  if (tc < mInterCalibConfig->cutLow[ih] || tc > mInterCalibConfig->cutHigh[ih]) {
    return;
  }
  double val[NPAR] = {0, 0, 0, 0, 0, 1};
  val[0] = tc;
  val[1] = t1;
  val[2] = t2;
  val[3] = t3;
  val[4] = t4;
  for (int32_t i = 0; i < NPAR; i++) {
    for (int32_t j = i; j < NPAR; j++) {
      mData.mSum[ih][i][j] += val[i] * val[j] * w;
    }
  }
  // mData.mSum[ih][5][5] contains the number of analyzed events
  double sumquad = val[1] + val[2] + val[3] + val[4];
  mHUnc[ih]->fill(sumquad, w);
  mHUnc[ih + NH]->fill(val[0]);
  mCUnc[ih]->fill(val[0], sumquad, w);
}

//______________________________________________________________________________
// Compute chi2 for minimization (static function)
void InterCalib::fcn(int& npar, double* gin, double& chi, double* par, int iflag)
{
  chi = 0;
  for (int32_t i = 0; i < NPAR; i++) {
    for (int32_t j = 0; j < NPAR; j++) {
      chi += (i == 0 ? par[i] : -par[i]) * (j == 0 ? par[j] : -par[j]) * mAdd[i][j];
    }
  }
}

int InterCalib::mini(int ih)
{
  // Copy to static object and symmetrize matrix
  // We use a static function and therefore we can do only one minimization at a time
  mMtx.lock();
  for (int32_t i = 0; i < NPAR; i++) {
    for (int32_t j = 0; j < NPAR; j++) {
      if (j < i) {
        mAdd[i][j] = mData.mSum[ih][j][i];
      } else {
        mAdd[i][j] = mData.mSum[ih][i][j];
      }
    }
  }
  double arglist[10];
  int ierflg = 0;
  double l_bnd = mInterCalibConfig->l_bnd[ih];
  double u_bnd = mInterCalibConfig->u_bnd[ih];
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
  l_bnd = mInterCalibConfig->l_bnd_o[ih];
  u_bnd = mInterCalibConfig->u_bnd_o[ih];
  start = 0;
  step = mInterCalibConfig->step_o[ih];
  mMn[ih]->mnparm(5, "offset", start, step, l_bnd, u_bnd, ierflg);
  mMn[ih]->mnexcm("MIGRAD", arglist, 0, ierflg);
  for (Int_t i = 0; i < NPAR; i++) {
    mMn[ih]->GetParameter(i, mPar[ih][i], mErr[ih][i]);
  }
  mMtx.unlock();
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
    if (mHUnc[ih]) {
      auto p = mHUnc[ih]->createTH1F(InterCalib::mHUncN[ih]);
      p->SetTitle(InterCalib::mHUncT[ih]);
      p->Write("", TObject::kOverwrite);
    }
  }
  for (int32_t ih = 0; ih < NH; ih++) {
    if (mCUnc[ih]) {
      auto p = mCUnc[ih]->createTH2F(InterCalib::mCUncN[ih]);
      p->SetTitle(InterCalib::mCUncT[ih]);
      p->Write("", TObject::kOverwrite);
    }
  }
  // Only after replay of RUN2 data
  for (int32_t ih = 0; ih < NH; ih++) {
    if (mHCorr[ih]) {
      mHCorr[ih]->Write("", TObject::kOverwrite);
    }
  }
  for (int32_t ih = 0; ih < NH; ih++) {
    if (mCCorr[ih]) {
      mCCorr[ih]->Write("", TObject::kOverwrite);
    }
  }
  // Minimization output
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
