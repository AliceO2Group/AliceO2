// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "FairLogger.h"

using namespace o2::TPC;

SAMPAProcessing::SAMPAProcessing() : mRandomNoiseRing(RandomRing::RandomType::Gaus)
{
  updateParameters();
}

SAMPAProcessing::~SAMPAProcessing() = default;

void SAMPAProcessing::updateParameters()
{
  auto& cdb = CDBInterface::instance();
  mGasParam = &(cdb.getParameterGas());
  mDetParam = &(cdb.getParameterDetector());
  mEleParam = &(cdb.getParameterElectronics());
  mPedestalMap = &(cdb.getPedestals());
  mNoiseMap = &(cdb.getNoise());
}

void SAMPAProcessing::getShapedSignal(float ADCsignal, float driftTime, std::vector<float>& signalArray) const
{
  const float timeBinTime = getTimeBinTime(driftTime);
  const float offset = driftTime - timeBinTime;
  for (float bin = 0; bin < mEleParam->getNShapedPoints(); bin += Vc::float_v::Size) {
    Vc::float_v binvector;
    for (int i = 0; i < Vc::float_v::Size; ++i) {
      binvector[i] = bin + i;
    }
    Vc::float_v time = timeBinTime + binvector * mEleParam->getZBinWidth();
    Vc::float_v signal = getGamma4(time, Vc::float_v(timeBinTime + offset), Vc::float_v(ADCsignal));
    for (int i = 0; i < Vc::float_v::Size; ++i) {
      signalArray[bin + i] = signal[i];
    }
  }
}
