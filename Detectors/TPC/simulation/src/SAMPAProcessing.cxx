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

/// \file SAMPAProcessing.cxx
/// \brief Implementation of the SAMPA response
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "TPCSimulation/SAMPAProcessing.h"
#include "TPCBase/CDBInterface.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "Framework/Logger.h"

using namespace o2::tpc;

SAMPAProcessing::SAMPAProcessing() : mRandomNoiseRing()
{
  updateParameters();
}

void SAMPAProcessing::updateParameters(float vdrift)
{
  mGasParam = &(ParameterGas::Instance());
  mDetParam = &(ParameterDetector::Instance());
  mEleParam = &(ParameterElectronics::Instance());
  auto& cdb = CDBInterface::instance();
  mPedestalMap = &(cdb.getPedestals());
  mNoiseMap = &(cdb.getNoise());
  mZeroSuppression = &(cdb.getZeroSuppressionThreshold());
  mVDrift = vdrift > 0 ? vdrift : mGasParam->DriftV;
}

void SAMPAProcessing::getShapedSignal(float ADCsignal, float driftTime, std::vector<float>& signalArray) const
{
  const float timeBinTime = getTimeBinTime(driftTime);
  const float offset = driftTime - timeBinTime;
  for (float bin = 0; bin < mEleParam->NShapedPoints; bin += Vc::float_v::Size) {
    Vc::float_v binvector;
    for (int i = 0; i < Vc::float_v::Size; ++i) {
      binvector[i] = bin + i;
    }
    Vc::float_v time = timeBinTime + binvector * mEleParam->ZbinWidth;
    Vc::float_v signal = getGamma4(time, Vc::float_v(timeBinTime + offset), Vc::float_v(ADCsignal));
    for (int i = 0; i < Vc::float_v::Size; ++i) {
      signalArray[bin + i] = signal[i];
    }
  }
}
