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
  // Inspect reconstruction parameters
  o2::zdc::CalibParamZDC& opt = const_cast<o2::zdc::CalibParamZDC&>(CalibParamZDC::Instance());
  opt.print();

  if (opt.debug_output > 0) {
    setSaveDebugHistos();
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
// Create calibration object
int NoiseCalib::endOfRun()
{
  if (mVerbosity > DbgZero) {
    LOGF(info, "Finalizing NoiseCalibData object");
  }


  // Compute average baseline
  float factor = mModuleConfig->baselineFactor;
  for (int ic = 0; ic < NChannels; ic++) {
    double sum = 0;
    double nsum = 0;
    double bmin = mConfig->cutLow[ic];
    double bmax = mConfig->cutHigh[ic];
    for (int ib = 0; ib < NoiseRange; ib++) {
      double bval = (NoiseMin + ib) * factor;
      if (bval >= bmin && bval <= bmax) {
        nsum += mData.mHisto[ic].mData[ib];
        sum += bval * mData.mHisto[ic].mData[ib];
      }
    }
    if (nsum > 0 && mConfig->min_e[ic]) {
      float ave = sum / nsum;
      LOGF(info, "Noise %s %g events and cuts (%g:%g): %f", ChannelNames[ic].data(), nsum, bmin, bmax, ave);
      mParamUpd.setCalib(ic, ave, true);
    } else {
      if (mParam == nullptr) {
        LOGF(error, "Noise %s %g events and cuts (%g:%g): CANNOT UPDATE AND MISSING OLD VALUE", ChannelNames[ic].data(), nsum, bmin, bmax);
        mParamUpd.setCalib(ic, -std::numeric_limits<float>::infinity(), false);
      } else {
        float val = mParam->getCalib(ic);
        LOGF(info, "Noise %s %g events and cuts (%g:%g): %f NOT UPDATED", ChannelNames[ic].data(), nsum, bmin, bmax, val);
        mParamUpd.setCalib(ic, val, false);
      }
    }
  }

  // Creating calibration object and info
  auto clName = o2::utils::MemFileHelper::getClassName(mParamUpd);
  mInfo.setObjectType(clName);
  auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
  mInfo.setFileName(flName);
  mInfo.setPath(CCDBPathNoiseCalib);
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
  LOGF(info, "Validity: %llu:%llu", starting, stopping);

  if (mSaveDebugHistos) {
    LOG(info) << "Saving debug histograms";
    saveDebugHistos();
  }
  return 0;
}

//______________________________________________________________________________
int NoiseCalib::saveDebugHistos(const std::string fn)
{
  return mData.saveDebugHistos(fn, mModuleConfig->baselineFactor);
}
