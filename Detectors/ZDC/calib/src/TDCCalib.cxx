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
#include "ZDCCalib/CalibParamZDC.h"
#include "ZDCCalib/TDCCalibData.h"
#include "ZDCCalib/TDCCalib.h"
#include "ZDCReconstruction/ZDCTDCParam.h"
#include "Framework/Logger.h"
#include "CCDB/CcdbApi.h"

using namespace o2::zdc;

int TDCCalib::init()
{
  if (mTDCCalibConfig == nullptr) {
    LOG(fatal) << "o2::zdc::TDCCalib: missing configuration object";
    return -1;
  }

  // Inspect calibration parameters
  const auto& opt = CalibParamZDC::Instance();
  opt.print();
  if (opt.debugOutput == true) {
    setSaveDebugHistos();
  }

  clear();
  auto* cfg = mTDCCalibConfig;
  int ih = 0;
  // clang-format off
  for (int i = 0; i < TDCCalibData::NTDC; i++) {
    mCTDC[i] = new o2::dataformats::FlatHisto1D<float>(cfg->nb1[ih],cfg->amin1[ih],cfg->amax1[ih]); //sum of TF histograms
    ih++;
  }
  // clang-format on
  mInitDone = true;
  return 0;
}

//______________________________________________________________________________
// Update calibration coefficients
int TDCCalib::endOfRun()
{
  if (mVerbosity > DbgZero) {
    LOGF(info, "Computing TDC Calibration coefficients");
  }
  for (int ih = 0; ih < TDCCalibData::NTDC; ih++) {
    LOGF(info, "%s %d events and cuts (%g:%g)", TDCCalibData::CTDC[ih], mData.entries[ih], mTDCCalibConfig->cutLow[ih], mTDCCalibConfig->cutHigh[ih]);

    if (!mTDCCalibConfig->enabled[ih]) {
      LOGF(info, "DISABLED processing of RUN3 data for ih = %d: %s", ih, TDCCalibData::CTDC[ih]);
      assign(ih, false);
    }

    else if (mData.entries[ih] >= mTDCCalibConfig->min_e[ih]) { //if number of events > minimum value accpeted -> process
      LOGF(info, "Processed RUN3 data for ih = %d: %s", ih, TDCCalibData::CTDC[ih]);
      assign(ih, true);
    } else {
      LOGF(info, "FAILED processing RUN3 data for ih = %d: %s: TOO FEW EVENTS: %d", ih, TDCCalibData::CTDC[ih], mData.entries[ih]);
      assign(ih, false);
    }
  }

  auto clName = o2::utils::MemFileHelper::getClassName(mTDCParamUpd);
  mInfo.setObjectType(clName);
  auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
  mInfo.setFileName(flName);
  mInfo.setPath(CCDBPathTDCCalib);
  std::map<std::string, std::string> md;
  md["config"] = mTDCCalibConfig->desc;
  mInfo.setMetaData(md);
  uint64_t starting = mData.mCTimeBeg;
  if (starting >= 10000) {
    starting = starting - 10000; // start 10 seconds before
  }
  uint64_t stopping = mData.mCTimeEnd + 10000; // stop 10 seconds after
  mInfo.setStartValidityTimestamp(starting);
  mInfo.setEndValidityTimestamp(stopping);

  if (mSaveDebugHistos) {
    write();
  }
  return 0;
}

//______________________________________________________________________________
// Update calibration object for the ten TDCs
// ismod=false if it was not possible to update the calibration coefficients
// due to low statistics
// ismod=true if the calibration was updated
void TDCCalib::assign(int ih, bool ismod)
{
  if (ih >= 0 && ih <= 9) {
    auto oldval = mTDCParam->getShift(ih); //old value from calibration object (TDCCalib)
    if (ismod == true) {                   //ismod == true
      auto val = oldval;
      auto shift = extractShift(ih);
      //Change wrt previous shift
      val = val + shift;
      if (val < 0) { //negative value or = 25ns shift is not acceptable
        LOGF(error, "Negative value of shift: %8.6f not acceptable", val);
      }

      else if (val >= 25) {
        LOGF(error, "Value of shift: %8.6f >= 25 ns not acceptable", val);
      }

      if (mVerbosity > DbgZero) {
        LOGF(info, "%s updated %8.6f -> %8.6f", TDCCalibData::CTDC[ih], oldval, val);
      }
      mTDCParamUpd.setShift(ih, val);
    }

    else { //ismod == false
      if (mVerbosity > DbgZero) {
        LOGF(info, "%s NOT CHANGED %8.6f", TDCCalibData::CTDC[ih], oldval);
      }
      mTDCParamUpd.setShift(ih, oldval);
    }
  }

  else { //TDC index out of range
    LOG(fatal) << "TDCCalib::assign accessing not existing ih = " << ih;
  }
}

void TDCCalib::clear(int ih)
{
  int ihstart = 0;
  int ihstop = TDCCalibData::NTDC;

  for (int32_t ii = ihstart; ii < ihstop; ii++) {
    if (mCTDC[ii]) {
      mCTDC[ii]->clear();
    }
  }
}

int TDCCalib::process(const TDCCalibData& data)
{
  if (!mInitDone) {
    init();
  }
  mData += data;
  return 0;
}

void TDCCalib::add(int ih, o2::dataformats::FlatHisto1D<float>& h1)
{
  if (!mInitDone) {
    init();
  }

  constexpr int nh = TDCCalibData::NTDC;

  if (ih >= 0 && ih < nh) {
    mCTDC[ih]->add(h1);
  } else {
    LOG(error) << "TDCCalib::add: unsupported FlatHisto1D " << ih;
  }
}

double TDCCalib::extractShift(int ih)
{
  // Extract the TDC shift
  auto h1 = mCTDC[ih]->createTH1F(TDCCalibData::CTDC[ih]); // createTH1F(histo_name)
  //h1->Draw("HISTO");
  int nEntries = h1->GetEntries();
  // std::cout << nEntries << std::endl;
  if ((ih >= 0 && ih <= 9) && (nEntries >= mTDCCalibConfig->min_e[ih])) { //TDC number is ok and more than minimum entries
    double avgShift = h1->GetMean();
    return avgShift;
  } else {
    LOG(error) << "TDCCalib::extractShift TDC out of range " << ih;
    return 0;
  }
}

int TDCCalib::write(const std::string fn)
{
  if (mVerbosity > DbgZero) {
    LOG(info) << "Saving aggregator histos on file " << fn;
  }
  TDirectory* cwd = gDirectory;
  TFile* f = new TFile(fn.data(), "recreate");
  if (f->IsZombie()) {
    LOG(error) << "Cannot create file: " << fn;
    return 1;
  }
  for (int32_t ih = 0; ih < TDCCalibData::NTDC; ih++) {
    if (mCTDC[ih]) {
      auto p = mCTDC[ih]->createTH1F(TDCCalibData::CTDC[ih]); // createTH1F(histo_name)
      p->SetTitle(TDCCalibData::CTDC[ih]);
      p->Write("", TObject::kOverwrite);
      if (mVerbosity > DbgMinimal) {
        LOG(info) << p->GetName() << " entries: " << p->GetEntries();
      }
    }
  }
  f->Close();
  cwd->cd();
  return 0;
}
