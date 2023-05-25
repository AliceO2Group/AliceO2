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
#include "ZDCCalib/NoiseCalib.h"
#include "Framework/Logger.h"
#include "CommonUtils/MemFileHelper.h"
#include "CCDB/CcdbApi.h"

using namespace o2::zdc;

int NoiseCalib::init()
{
  // Inspect calibration parameters
  const auto& opt = CalibParamZDC::Instance();
  opt.print();
  if (opt.debugOutput == true) {
    setSaveDebugHistos();
  }

  for (int isig = 0; isig < NChannels; isig++) {
    mH[isig] = new o2::dataformats::FlatHisto1D<double>(4096, -2048.7, 2047.5);
  }

  clear();
  mInitDone = true;
  return 0;
}

//______________________________________________________________________________
void NoiseCalib::clear()
{
  mData.clear();
}

//______________________________________________________________________________
int NoiseCalib::process(const o2::zdc::NoiseCalibSummaryData* data)
{
  if (!mInitDone) {
    init();
  }
  if (mVerbosity >= DbgFull) {
    data->print();
  }
  mData += data;
  if (mVerbosity >= DbgFull) {
    mData.print();
  }
  return 0;
}

//______________________________________________________________________________
// Add histograms
void NoiseCalib::add(int ih, o2::dataformats::FlatHisto1D<double>& h1)
{
  if (!mInitDone) {
    init();
  }
  if (ih >= 0 && ih < NChannels) {
    mH[ih]->add(h1);
  } else {
    LOG(error) << "InterCalib::add: unsupported FlatHisto1D " << ih;
  }
}

//______________________________________________________________________________
// Create calibration object
int NoiseCalib::endOfRun()
{
  if (mVerbosity > DbgZero) {
    LOGF(info, "Finalizing NoiseCalibData object");
    mData.print();
  }
  if (mSaveDebugHistos) {
    saveDebugHistos();
  }

  for (int isig = 0; isig < NChannels; isig++) {
    uint64_t en = 0;
    double mean = 0, var = 0;
    // N.B. Histogram is used to evaluate mean variance -> OK to use mean!
    mData.getStat(isig, en, mean, var);
    if (en > 0) {
      double stdev = std::sqrt(mean / double(NTimeBinsPerBC) / double(NTimeBinsPerBC - 1));
      mParam.setCalib(isig, stdev);
      mParam.entries[isig] = en;
    }
  }

  mParam.print();

  // Creating calibration object and info
  auto clName = o2::utils::MemFileHelper::getClassName(mParam);
  mInfo.setObjectType(clName);
  auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
  mInfo.setFileName(flName);
  mInfo.setPath(CCDBPathNoiseCalib);

  const auto& opt = CalibParamZDC::Instance();
  std::map<std::string, std::string> md;
  md["config"] = opt.descr;
  mInfo.setMetaData(md);
  uint64_t starting = mData.mCTimeBeg;
  if (starting >= 10000) {
    starting = starting - 10000; // start 10 seconds before
  }
  uint64_t stopping = mData.mCTimeEnd + 10000; // stop 10 seconds after
  mInfo.setStartValidityTimestamp(starting);
  mInfo.setEndValidityTimestamp(stopping);
  mInfo.setAdjustableEOV();
  LOGF(info, "Validity: %llu:%llu", starting, stopping);

  return 0;
}

//______________________________________________________________________________
int NoiseCalib::saveDebugHistos(const std::string fn)
{
  LOG(info) << "Saving debug histograms on file " << fn;
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
