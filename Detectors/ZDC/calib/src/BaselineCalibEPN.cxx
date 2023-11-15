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
#include "ZDCCalib/BaselineCalibEPN.h"
#include "Framework/Logger.h"

using namespace o2::zdc;

int BaselineCalibEPN::init()
{
  static bool firstCall = true;
  if (firstCall) {
    // Inspect reconstruction parameters
    const auto& opt = CalibParamZDC::Instance();
    if (mVerbosity >= DbgFull) {
      opt.print();
    }
    if (opt.debugOutput == true) {
      setSaveDebugHistos();
    }
    if (mVerbosity >= DbgMedium) {
      mModuleConfig->print();
    }
    firstCall = false;
  } else {
    // Reset data structure
    mData.clear();
  }
  mInitDone = true;
  return 0;
}

//______________________________________________________________________________
int BaselineCalibEPN::process(const gsl::span<const o2::zdc::OrbitData>& orbitdata)
{
  if (!mInitDone) {
    init();
  }
  for (auto& myorbit : orbitdata) {
    for (int ich = 0; ich < NChannels; ich++) {
      // Check if orbit data is valid. N.B. the default scaler initializer has
      // 0x8fff that is in overflow. Data loss in orbit has most significant bit
      // set to 1
      // TODO: relax this condition?
      if (myorbit.scaler[ich] <= o2::constants::lhc::LHCMaxBunches) {
        auto myped = float(myorbit.data[ich]) * mModuleConfig->baselineFactor;
        if (myped >= ADCMin && myped <= ADCMax) {
          mData.addEntry(ich, myorbit.data[ich]);
          if (mSaveDebugHistos) {
            mDataSum.addEntry(ich, myorbit.data[ich]);
          }
        }
      }
    }
  }
  return 0;
}

//______________________________________________________________________________
int BaselineCalibEPN::endOfRun()
{
  if (mVerbosity > DbgZero) {
    LOG(info) << "BaselineCalibEPN::endOfRun";
  }
  if (mSaveDebugHistos) {
    if (mVerbosity >= DbgMedium) {
      mDataSum.print();
    }
    saveDebugHistos();
  }
  mInitDone = false;
  return 0;
}

//______________________________________________________________________________
int BaselineCalibEPN::saveDebugHistos(const std::string fn)
{
  // EPN debug histos are now cumulated over process life
  LOG(info) << "Saving EPN debug histos on file " << fn;
  return mDataSum.saveDebugHistos(fn, mModuleConfig->baselineFactor);
}
