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

/// \file DigitizerCalibrations.cxx
/// \brief Calibration objects of the TPC digitizer, shared read-only between digitizers

#include "TPCSimulation/DigitizerCalibrations.h"
#include "TPCBaseRecSim/CDBInterface.h"
#include "TPCBase/ParameterElectronics.h"
#include "TPCBase/IonTailSettings.h"
#include "Framework/Logger.h"

namespace o2::tpc
{

std::shared_ptr<const DigitizerCalibrations> DigitizerCalibrations::fetch()
{
  const auto& eleParam = ParameterElectronics::Instance();
  auto& cdb = CDBInterface::instance();
  auto calib = std::make_shared<DigitizerCalibrations>();

  // the same order as the digitizer used: the FEE parameter file may replace the pedestals, noise and thresholds
  calib->gainMap = &cdb.getGainMap();

  if (eleParam.doIonTailPerPad) {
    const auto& itSettings = IonTailSettings::Instance();
    if (itSettings.padITCorrFile.size()) {
      cdb.setFEEParamsFromFile(itSettings.padITCorrFile);
    }
    calib->padParams[0] = &cdb.getITFraction();
    calib->padParams[1] = &cdb.getITExpLambda();
  }
  if (eleParam.doCommonModePerPad) {
    calib->padParams[2] = &cdb.getCMkValues();
  }
  if (eleParam.applyDeadMap) {
    calib->deadMap = std::make_unique<const CalDet<bool>>(cdb.getDeadChannelMap());
  }
  calib->cmcEnabled = (eleParam.DigiMode == DigitzationMode::Auto) && cdb.getFEEConfig().isCMCEnabled();

  calib->pedestals = &cdb.getPedestals();
  calib->pedestalsCRU = &cdb.getPedestalsCRU();
  calib->noise = &cdb.getNoise();
  calib->zeroSuppression = &cdb.getZeroSuppressionThreshold();

  static bool reportedSettings = false;
  if (!reportedSettings) {
    reportedSettings = true;
    LOGP(info, "ParameterElectronics:  doIonTail={}, doIonTailPerPad={}, doCommonModePerPad={}, doSaturationTail={}, doNoiseEmptyPads={}, applyDeadMap={}, commonModeCoupling={}, DigiMode={}",
         eleParam.doIonTail, eleParam.doIonTailPerPad, eleParam.doCommonModePerPad, eleParam.doSaturationTail, eleParam.doNoiseEmptyPads, eleParam.applyDeadMap, eleParam.commonModeCoupling, (int)eleParam.DigiMode);
    cdb.getFEEConfig().printShort();
    if (calib->deadMap) {
      LOGP(info, "Using dead map with {} masked pads", calib->deadMap->getSum<int>());
    }
  }
  return calib;
}

} // namespace o2::tpc
