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

/// \file CDBInterface.h
/// \brief Simple interface to the CDB manager
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

// system includes
#include <cxxabi.h>
#include <ctime>
#include <memory>
#include <fmt/format.h>
#include <fmt/chrono.h>

// root includes
#include "TFile.h"
#include "TRandom.h"

// o2 includes
#include "DataFormatsTPC/CalibdEdxCorrection.h"
#include "TPCBase/CDBInterface.h"
#include "TPCBase/ParameterDetector.h"
#include "TPCBase/ParameterElectronics.h"
#include "TPCBase/ParameterGEM.h"
#include "TPCBase/ParameterGas.h"
#include "TPCBase/Utils.h"

using namespace o2::tpc;

// anonymous namespace to prevent usage outside of this file
namespace
{
/// utility function to demangle cxx type names
std::string demangle(std::string_view name)
{
  int status = -4; // some arbitrary value to eliminate the compiler warning
  std::unique_ptr<char, void (*)(void*)> res{abi::__cxa_demangle(name.data(), nullptr, nullptr, &status), std::free};
  return (status == 0) ? res.get() : name.data();
}
} // end anonymous namespace

//______________________________________________________________________________
const CalPad& CDBInterface::getPedestals()
{
  // ===| load noise and pedestals from file if requested |=====================
  if (mPedestalNoiseFileName.size()) {
    if (!mPedestals) {
      loadNoiseAndPedestalFromFile();
    }
  } else if (mUseDefaults) {
    if (!mPedestals) {
      createDefaultPedestals();
    }
  } else {
    // return from CDB, assume that check for object existence are done there
    return getObjectFromCDB<CalPadMapType>(CDBTypeMap.at(CDBType::CalPedestalNoise)).at("Pedestals");
  }

  if (!mPedestals) {
    LOG(fatal) << "No valid pedestal object was loaded";
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
    if (!mNoise) {
      createDefaultNoise();
    }
  } else {
    // return from CDB, assume that check for object existence are done there
    return getObjectFromCDB<CalPadMapType>(CDBTypeMap.at(CDBType::CalPedestalNoise)).at("Noise");
  }

  if (!mNoise) {
    LOG(fatal) << "No valid noise object was loaded";
  }

  return *mNoise;
}

//______________________________________________________________________________
const CalPad& CDBInterface::getZeroSuppressionThreshold()
{
  // ===| load gain map from file if requested |=====================
  if (mThresholdMapFileName.size()) {
    if (!mZeroSuppression) {
      loadThresholdMapFromFile();
    }
  } else if (mUseDefaults) {
    if (!mZeroSuppression) {
      createDefaultZeroSuppression();
    }
  } else {
    // return from CDB, assume that check for object existence are done there
    return getObjectFromCDB<CalPadMapType>(CDBTypeMap.at(CDBType::ConfigFEEPad)).at("ThresholdMap");
  }
  return *mZeroSuppression;
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
    if (!mGainMap) {
      createDefaultGainMap();
    }
  } else {
    // return from CDB, assume that check for object existence are done there
    return getObjectFromCDB<CalPad>(CDBTypeMap.at(CDBType::CalPadGainFull));
  }

  if (!mGainMap) {
    LOG(fatal) << "No valid gain object was loaded";
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
  return getObjectFromCDB<ParameterDetector>(CDBTypeMap.at(CDBType::ParDetector));
}

//______________________________________________________________________________
const ParameterElectronics& CDBInterface::getParameterElectronics()
{
  if (mUseDefaults) {
    return ParameterElectronics::Instance();
  }

  // return from CDB, assume that check for object existence are done there
  return getObjectFromCDB<ParameterElectronics>(CDBTypeMap.at(CDBType::ParElectronics));
}

//______________________________________________________________________________
const ParameterGas& CDBInterface::getParameterGas()
{
  if (mUseDefaults) {
    return ParameterGas::Instance();
  }

  // return from CDB, assume that check for object existence are done there
  return getObjectFromCDB<ParameterGas>(CDBTypeMap.at(CDBType::ParGas));
}

//______________________________________________________________________________
const ParameterGEM& CDBInterface::getParameterGEM()
{
  if (mUseDefaults) {
    return ParameterGEM::Instance();
  }

  // return from CDB, assume that check for object existence are done there
  return getObjectFromCDB<ParameterGEM>(CDBTypeMap.at(CDBType::ParGEM));
}

//______________________________________________________________________________
const CalPad& CDBInterface::getCalPad(const std::string_view path)
{
  return getSpecificObjectFromCDB<CalPad>(path);
}

//______________________________________________________________________________
void CDBInterface::loadNoiseAndPedestalFromFile()
{
  std::unique_ptr<TFile> file(TFile::Open(mPedestalNoiseFileName.data()));
  CalPad* pedestals{nullptr};
  CalPad* noise{nullptr};
  file->GetObject("Pedestals", pedestals);
  file->GetObject("Noise", noise);

  if (!pedestals) {
    LOG(fatal) << "No valid pedestal object was loaded";
  }

  if (!noise) {
    LOG(fatal) << "No valid noise object was loaded";
  }

  mPedestals.reset(pedestals);
  mNoise.reset(noise);

  LOG(info) << "Loaded Noise and pedestal from file '" << mPedestalNoiseFileName << "'";
}

//______________________________________________________________________________
void CDBInterface::loadGainMapFromFile()
{
  std::unique_ptr<TFile> file(TFile::Open(mGainMapFileName.data()));
  CalPad* gain{nullptr};
  file->GetObject("GainMap", gain);

  if (!gain) {
    LOG(fatal) << "No valid gain map object was loaded";
  }

  mGainMap.reset(gain);

  LOG(info) << "Loaded gain map from file '" << mGainMapFileName << "'";
}

//______________________________________________________________________________
void CDBInterface::loadThresholdMapFromFile()
{
  if (mThresholdMapFileName.empty()) {
    return;
  }

  auto calPads = o2::tpc::utils::readCalPads(mThresholdMapFileName, "ThresholdMap");

  if (calPads.size() != 1) {
    LOGP(fatal, "Missing 'ThresholdMap' object in file {}", mThresholdMapFileName);
  }

  mZeroSuppression.reset(calPads[0]);
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
void CDBInterface::createDefaultZeroSuppression()
{
  // default map is mDefaultZSsigma * noise
  mZeroSuppression = std::unique_ptr<CalPad>(new CalPad(getNoise()));
  mZeroSuppression->setName("ThresholdMap");

  const auto zsSigma = mDefaultZSsigma;
  for (auto& calArray : mZeroSuppression->getData()) {
    auto& data = calArray.getData();
    std::transform(data.begin(), data.end(), data.begin(), [zsSigma](const auto value) { return zsSigma * value; });
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

//______________________________________________________________________________
bool CDBStorage::checkMetaData(MetaData_t metaData) const
{
  if (!metaData.size()) {
    LOGP(error, "no meta data set");
  }

  const std::array<std::string_view, 3> requirements{
    "Required",
    "Recommended",
    "Optional"};

  const std::array<std::vector<std::string_view>, 3> tests{{
    {"Responsible", "Reason", "Intervention"}, // errors
    {"JIRA"},                                  // warnings
    {"Comment"}                                // infos
  }};

  std::array<int, 3> counts{};

  for (size_t i = 0; i < requirements.size(); ++i) {
    for (const auto& test : tests[i]) {
      if (!metaData[test.data()].size()) {
        const auto message = fmt::format("{} field {} not set in meta data", requirements[i], test);
        if (i == 0) {
          LOG(error) << message;
        } else if (i == 1) {
          LOG(warning) << message;
        } else {
          LOG(info) << message;
        }
        ++counts[i];
      } else {
        LOGP(debug, "{} field '{}' set to '{}'", requirements[i], test, metaData[test.data()]);
      }
    }
  }

  return counts[0] == 0;
}

//______________________________________________________________________________
void CDBStorage::uploadNoiseAndPedestal(std::string_view fileName, long first, long last)
{
  std::unique_ptr<TFile> file(TFile::Open(fileName.data()));
  CalPad* pedestals{nullptr};
  CalPad* noise{nullptr};
  file->GetObject("Pedestals", pedestals);
  file->GetObject("Noise", noise);

  if (!pedestals) {
    LOGP(fatal, "No valid pedestal object was loaded from file {}", fileName);
  }

  if (!noise) {
    LOGP(fatal, "No valid noise object was loaded from file {}", fileName);
  }

  CDBInterface::CalPadMapType calib;

  calib["Pedestals"] = *pedestals;
  calib["Noise"] = *noise;

  storeObject(&calib, CDBType::CalPedestalNoise, first, last);
}

//______________________________________________________________________________
void CDBStorage::uploadGainMap(std::string_view fileName, bool isFull, long first, long last)
{
  std::unique_ptr<TFile> file(TFile::Open(fileName.data()));
  CalPad* gain{nullptr};
  file->GetObject("GainMap", gain);

  if (!gain) {
    LOG(fatal) << "No valid gain map object was loaded";
  }

  storeObject(gain, isFull ? CDBType::CalPadGainFull : CDBType::CalPadGainResidual, first, last);
}

//______________________________________________________________________________
void CDBStorage::uploadPulserOrCEData(CDBType type, std::string_view fileName, long first, long last)
{
  auto calPads = o2::tpc::utils::readCalPads(fileName, "T0,Width,Qtot");

  if (calPads.size() != 3) {
    LOGP(fatal, "Missing pulser object in file {}", fileName);
  }
  auto t0 = calPads[0];
  auto width = calPads[1];
  auto qtot = calPads[2];

  std::unordered_map<std::string, CalDet<float>> pulserCalib;
  pulserCalib["T0"] = *t0;
  pulserCalib["Width"] = *width;
  pulserCalib["Qtot"] = *qtot;

  storeObject(&pulserCalib, type, first, last);
}

//______________________________________________________________________________
void CDBStorage::uploadFEEConfigPad(std::string_view fileName, long first, long last)
{
  auto calPads = o2::tpc::utils::readCalPads(fileName, "ThresholdMap");

  if (calPads.size() != 1) {
    LOGP(fatal, "Missing pulser object in file {}", fileName);
  }
  auto thresholdMap = calPads[0];

  std::unordered_map<std::string, CalDet<float>> feeConfigPad;
  feeConfigPad["ThresholdMap"] = *thresholdMap;

  storeObject(&feeConfigPad, CDBType::ConfigFEEPad, first, last);
}

//______________________________________________________________________________
void CDBStorage::uploadTimeGain(std::string_view fileName, long first, long last)
{
  std::unique_ptr<TFile> file(TFile::Open(fileName.data()));
  auto timeGain = file->Get<o2::tpc::CalibdEdxCorrection>("CalibdEdxCorrection");

  if (!timeGain) {
    LOGP(fatal, "No valid timeGain object found in {}", fileName);
  }

  storeObject(timeGain, CDBType::CalTimeGain, first, last);
}

//______________________________________________________________________________
void CDBStorage::printObjectSummary(std::string_view name, CDBType const type, MetaData_t const& metadata, long start, long end) const
{
  std::time_t tstart(start / 1000);
  std::time_t tend(end / 1000);
  auto tstartms = start % 1000;
  auto tendms = end % 1000;

  std::string message = fmt::format("Writing object of type '{}'\n", demangle(name)) +
                        fmt::format("          to storage '{}'\n", mCCDB.getURL()) +
                        fmt::format("          into path '{}'\n", CDBTypeMap.at(type)) +
                        fmt::format("          with validity [{}, {}] :", start, end) +
                        fmt::format("          [{:%d.%m.%Y %H:%M:%S}.{:03d}, {:%d.%m.%Y %H:%M:%S}.{:03d}]\n", fmt::localtime(tstart), tstartms, fmt::localtime(tend), tendms) +
                        std::string("          Meta data:\n");

  for (const auto& [key, value] : metadata) {
    message += fmt::format("{:>20} = {}\n", key, value);
  }

  LOGP(info, message);
}
