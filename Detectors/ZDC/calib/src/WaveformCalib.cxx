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
#include "ZDCCalib/WaveformCalib.h"
#include "ZDCReconstruction/ZDCEnergyParam.h"
#include "ZDCReconstruction/ZDCTowerParam.h"
#include "Framework/Logger.h"
#include "CCDB/CcdbApi.h"

using namespace o2::zdc;

int WaveformCalib::init()
{
  if (mWaveformCalibConfig == nullptr) {
    LOG(fatal) << "o2::zdc::WaveformCalib: missing configuration object";
    return -1;
  }
  clear();
  mInitDone = true;
  return 0;
}

//______________________________________________________________________________
// Update calibration object
int WaveformCalib::endOfRun()
{
  if (mVerbosity > DbgZero) {
    LOGF(info, "Computing intercalibration coefficients");
  }
  auto clName = o2::utils::MemFileHelper::getClassName(mTowerParamUpd);
  mInfo.setObjectType(clName);
  auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
  mInfo.setFileName(flName);
  mInfo.setPath(CCDBPathTowerCalib);
  std::map<std::string, std::string> md;
  md["config"] = mWaveformCalibConfig->desc;
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

void WaveformCalib::clear(int ih)
{
  mData.mSum[ii][i][j] = 0;
}

int WaveformCalib::process(const WaveformCalibData& data)
{
  if (!mInitDone) {
    init();
  }
  mData += data;
  return 0;
}

int WaveformCalib::write(const std::string fn)
{
  TDirectory* cwd = gDirectory;
  TFile* f = new TFile(fn.data(), "recreate");
  if (f->IsZombie()) {
    LOG(error) << "Cannot create file: " << fn;
    return 1;
  }
  for (int32_t ih = 0; ih < (2 * NH); ih++) {
    if (mHUnc[ih]) {
      auto p = mHUnc[ih]->createTH1F(WaveformCalib::mHUncN[ih]);
      p->SetTitle(WaveformCalib::mHUncT[ih]);
      p->Write("", TObject::kOverwrite);
    }
  }
  for (int32_t ih = 0; ih < NH; ih++) {
    if (mCUnc[ih]) {
      auto p = mCUnc[ih]->createTH2F(WaveformCalib::mCUncN[ih]);
      p->SetTitle(WaveformCalib::mCUncT[ih]);
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
