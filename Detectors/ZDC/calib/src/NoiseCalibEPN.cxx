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
#include "ZDCCalib/NoiseCalibEPN.h"
#include "Framework/Logger.h"

using namespace o2::zdc;

int NoiseCalibEPN::init()
{
  // Inspect reconstruction parameters
  o2::zdc::CalibParamZDC& opt = const_cast<o2::zdc::CalibParamZDC&>(CalibParamZDC::Instance());
  opt.print();

  if (mVerbosity > DbgZero) {
    mModuleConfig->print();
  }

  if (opt.debug_output > 0) {
    setSaveDebugHistos();
  }

  mInitDone = true;
  return 0;
}

//______________________________________________________________________________
int NoiseCalibEPN::process(int process(const gsl::span<const o2::zdc::BCData>& bcdata, const gsl::span<const o2::zdc::ChannelData>& chdata)
{
  if (!mInitDone) {
    init();
  }
  return 0;
}

//______________________________________________________________________________
int NoiseCalibEPN::endOfRun()
{
  if (mVerbosity > DbgZero) {
    mData.print();
  }
  if (mSaveDebugHistos) {
    saveDebugHistos();
  }
  return 0;
}

//______________________________________________________________________________
int NoiseCalibEPN::saveDebugHistos(const std::string fn)
{
  return mData.saveDebugHistos(fn, mModuleConfig->baselineFactor);
}
