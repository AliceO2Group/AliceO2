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
#include "ZDCCalib/WaveformCalib.h"
#include "ZDCCalib/WaveformCalibQueue.h"
#include "Framework/Logger.h"
#include "CommonUtils/MemFileHelper.h"
#include "CCDB/CcdbApi.h"

using namespace o2::zdc;

int WaveformCalib::init()
{
  if (mConfig == nullptr) {
    LOG(fatal) << "o2::zdc::WaveformCalib: missing configuration object";
    return -1;
  }

  auto* cfg = mConfig;
  if (mVerbosity > DbgZero) {
    mConfig->print();
  }

  // Inspect reconstruction parameters
  const auto& opt = CalibParamZDC::Instance();
  opt.print();

  if (opt.rootOutput == true) {
    setSaveDebugHistos();
  }

  clear();
  mData.setN(mConfig->nbun);
  mData.mPeak = WaveformCalibQueue::peak(-(mConfig->ibeg));
  mInitDone = true;
  return 0;
}

//______________________________________________________________________________
void WaveformCalib::clear()
{
  mData.clear();
}

//______________________________________________________________________________
int WaveformCalib::process(const WaveformCalibData& data)
{
  if (!mInitDone) {
    init();
  }
  // Add checks before addition
  if ((mData.mN != data.mN) || (mData.mPeak != data.mPeak)) {
    LOG(fatal) << "WaveformCalib::process adding inconsistent data mN cfg=" << mData.mN << " vs data=" << data.mN << " mPeak cfg=" << mData.mPeak << " vs data=" << data.mPeak;
    return -1;
  }
  mData += data;
  return 0;
}

//______________________________________________________________________________
// Create calibration object
int WaveformCalib::endOfRun()
{
  if (mVerbosity > DbgZero) {
    LOGF(info, "Finalizing WaveformCalibData object");
  }
  auto clName = o2::utils::MemFileHelper::getClassName(mData);
  mInfo.setObjectType(clName);
  auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
  mInfo.setFileName(flName);
  mInfo.setPath(CCDBPathWaveformCalib);
  std::map<std::string, std::string> md;
  md["config"] = mConfig->desc;
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
    saveDebugHistos();
  }
  return 0;
}

//______________________________________________________________________________
int WaveformCalib::saveDebugHistos(const std::string fn)
{
  return mData.saveDebugHistos(fn);
}
