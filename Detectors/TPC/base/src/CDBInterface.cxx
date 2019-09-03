// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CDBInterface.h
/// \brief Simple interface to the CDB manager
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

// root includes
#include "TFile.h"
#include "TRandom.h"

// fairroot includes
#include "FairLogger.h"

// o2 includes
#include "TPCBase/CDBInterface.h"
#include "TPCBase/ParameterDetector.h"
#include "TPCBase/ParameterElectronics.h"
#include "TPCBase/ParameterGEM.h"
#include "TPCBase/ParameterGas.h"

using namespace o2::tpc;

//______________________________________________________________________________
const CalPad& CDBInterface::getPedestals()
{
  // ===| load noise and pedestals from file if requested |=====================
  if (mPedestalNoiseFileName.size()) {
    if (!mPedestals) {
      loadNoiseAndPedestalFromFile();
    }
  } else if (mUseDefaults) {
    if (!mPedestals)
      createDefaultPedestals();
  } else {
    // return from CDB, assume that check for object existence are done there
    return getObjectFromCDB<CalPad>("TPC/Calib/Pedestals");
  }

  if (!mPedestals) {
    LOG(FATAL) << "No valid pedestal object was loaded";
  }

  return *mPedestals;
}

//______________________________________________________________________________
const CalPad& CDBInterface::getNoise()
{
  // ===| load noise and pedestals from file if requested |=====================
  if (mPedestalNoiseFileName.size()) {
    if (!mNoise) {
      loadNoiseAndPedestalFromFile();
    }
  } else if (mUseDefaults) {
    if (!mNoise)
      createDefaultNoise();
  } else {
    // return from CDB, assume that check for object existence are done there
    return getObjectFromCDB<CalPad>("TPC/Calib/Noise");
  }

  if (!mNoise) {
    LOG(FATAL) << "No valid noise object was loaded";
  }

  return *mNoise;
}

//______________________________________________________________________________
const CalPad& CDBInterface::getGainMap()
{
  // ===| load gain map from file if requested |=====================
  if (mGainMapFileName.size()) {
    if (!mGainMap) {
      loadGainMapFromFile();
    }
  } else if (mUseDefaults) {
    if (!mGainMap)
      createDefaultGainMap();
  } else {
    // return from CDB, assume that check for object existence are done there
    return getObjectFromCDB<CalPad>("TPC/Calib/Gain");
  }

  if (!mGainMap) {
    LOG(FATAL) << "No valid gain object was loaded";
  }

  return *mGainMap;
}

//______________________________________________________________________________
const ParameterDetector& CDBInterface::getParameterDetector()
{
  if (mUseDefaults) {
    return ParameterDetector::Instance();
  }

  // return from CDB, assume that check for object existence are done there
  return getObjectFromCDB<ParameterDetector>("TPC/Parameter/Detector");
}

//______________________________________________________________________________
const ParameterElectronics& CDBInterface::getParameterElectronics()
{
  if (mUseDefaults) {
    return ParameterElectronics::Instance();
  }

  // return from CDB, assume that check for object existence are done there
  return getObjectFromCDB<ParameterElectronics>("TPC/Parameter/Electronics");
}

//______________________________________________________________________________
const ParameterGas& CDBInterface::getParameterGas()
{
  if (mUseDefaults) {
    return ParameterGas::Instance();
  }

  // return from CDB, assume that check for object existence are done there
  return getObjectFromCDB<ParameterGas>("TPC/Parameter/Gas");
}

//______________________________________________________________________________
const ParameterGEM& CDBInterface::getParameterGEM()
{
  if (mUseDefaults) {
    return ParameterGEM::Instance();
  }

  // return from CDB, assume that check for object existence are done there
  return getObjectFromCDB<ParameterGEM>("TPC/Parameter/GEM");
}

//______________________________________________________________________________
void CDBInterface::loadNoiseAndPedestalFromFile()
{
  auto file = TFile::Open(mPedestalNoiseFileName.data());
  CalPad* pedestals{nullptr};
  CalPad* noise{nullptr};
  file->GetObject("Pedestals", pedestals);
  file->GetObject("Noise", noise);
  delete file;

  if (!pedestals) {
    LOG(FATAL) << "No valid pedestal object was loaded";
  }

  if (!noise) {
    LOG(FATAL) << "No valid noise object was loaded";
  }

  mPedestals.reset(pedestals);
  mNoise.reset(noise);

  LOG(INFO) << "Loaded Noise and pedestal from file '" << mPedestalNoiseFileName << "'";
}

//______________________________________________________________________________
void CDBInterface::loadGainMapFromFile()
{
  auto file = TFile::Open(mGainMapFileName.data());
  CalPad* gain{nullptr};
  file->GetObject("Gain", gain);
  delete file;

  if (!gain) {
    LOG(FATAL) << "No valid gain map object was loaded";
  }

  mGainMap.reset(gain);

  LOG(INFO) << "Loaded gain map from file '" << mGainMapFileName << "'";
}
//______________________________________________________________________________
void CDBInterface::createDefaultPedestals()
{
  // ===| create random pedestals |=============================================
  mPedestals = std::make_unique<CalPad>("Pedestals");

  // distribution based on test beam data
  const float meanPedestal = 72.5;
  const float sigmaPedestal = 9.0;

  // set a minimum and maximum for the value
  const float minPedestal = meanPedestal - 4 * sigmaPedestal;
  const float maxPedestal = meanPedestal + 4 * sigmaPedestal;

  for (auto& calArray : mPedestals->getData()) {
    for (auto& val : calArray.getData()) {
      float random = gRandom->Gaus(meanPedestal, sigmaPedestal);
      if (random < minPedestal) {
        random = minPedestal;
      }
      if (random > maxPedestal) {
        random = maxPedestal;
      }
      val = random;
    }
  }
}

//______________________________________________________________________________
void CDBInterface::createDefaultNoise()
{
  // ===| create random noise |=============================================
  mNoise = std::make_unique<CalPad>("Noise");

  // distribution based on test beam data
  const float meanNoise = 1.0;
  const float sigmaNoise = 0.05;

  // set a minimum and maximum for the value
  const float minNoise = meanNoise - 4 * sigmaNoise;
  const float maxNoise = meanNoise + 8 * sigmaNoise;

  for (auto& calArray : mNoise->getData()) {
    for (auto& val : calArray.getData()) {
      float random = gRandom->Gaus(meanNoise, sigmaNoise);
      if (random < minNoise) {
        random = minNoise;
      }
      if (random > maxNoise) {
        random = maxNoise;
      }
      val = random;
    }
  }
}

//______________________________________________________________________________
void CDBInterface::createDefaultGainMap()
{
  // ===| create random gain map |=============================================
  mGainMap = std::make_unique<CalPad>("Gain");

  // distribution based on test beam data
  const float meanGain = 1.0;
  const float sigmaGain = 0.12;

  // set a minimum and maximum for the value
  const float minGain = meanGain - 4 * sigmaGain;
  const float maxGain = meanGain + 8 * sigmaGain;

  for (auto& calArray : mGainMap->getData()) {
    for (auto& val : calArray.getData()) {
      float random = gRandom->Gaus(meanGain, sigmaGain);
      if (random < minGain) {
        random = minGain;
      }
      if (random > maxGain) {
        random = maxGain;
      }
      val = random;
    }
  }
}
