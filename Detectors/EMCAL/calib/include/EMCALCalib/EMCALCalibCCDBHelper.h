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

/// \class EMCALCalibCCDBHelper.h
/// \brief Helper to load the inputs for calibration objects from the ccdb
/// \author Joshua Konig <joshua.konig@cern.ch>
/// \ingroup EMCALCalib
/// \since Aug 18, 2022

#ifndef EMCAL_CALIB_CCDB_HELPER_H_
#define EMCAL_CALIB_CCDB_HELPER_H_

#include <vector>
#include <memory>
#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "CommonUtils/NameConf.h"
#include "Framework/CCDBParamSpec.h"
#include "EMCALCalib/CalibDB.h"
#include "EMCALCalib/BadChannelMap.h"
#include "EMCALCalib/TimeCalibrationParams.h"
#include "EMCALCalib/GainCalibrationFactors.h"
#include "EMCALCalib/TempCalibrationParams.h"

namespace o2::framework
{
class ProcessingContext;
class ConcreteDataMatcher;
class InputSpec;
} // namespace o2::framework

namespace o2
{

namespace emcal
{

struct EMCALCalibRequest {

  ///\brief constructor
  EMCALCalibRequest() = delete;
  EMCALCalibRequest(bool badChannel, bool timeCalib, bool gainCalib, bool tempCalib, std::vector<o2::framework::InputSpec>& inputs, bool askOnce = false);

  ///\brief
  void addInput(const o2::framework::InputSpec&& isp, std::vector<o2::framework::InputSpec>& inputs);

  bool mAskBadChannel = false;    ///< switch to ask weather bad channel calibration should be applied
  bool mAskTimeCalib = false;     ///< switch to ask weather time calibration should be applied
  bool mAskTimeCalibSlew = false; ///< switch to ask weather time calibration slope should be applied
  bool mAskGainCalib = false;     ///< switch to ask weather energy/gain calibration should be applied
  bool mAskTempCalib = false;     ///< switch to ask weather temperature calibration should be applied

  ClassDefNV(EMCALCalibRequest, 1);
};

class EMCALCalibCCDBHelper
{

 public:
  ///\brief static constructor
  static EMCALCalibCCDBHelper& instance()
  {
    static EMCALCalibCCDBHelper inst;
    return inst;
  }

  ///\brief set EMCALCalibRequest to know which calibrations have to be applied
  ///\param req EMCALCalibRequest
  void setRequest(std::shared_ptr<EMCALCalibRequest> req);

  ///\brief function to update calibration objects
  bool finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj);

  ///\brief function to check if calibration objects need to be updated
  void checkUpdates(o2::framework::ProcessingContext& pc) const;

  ///\brief check if bad channel calibration should be applied
  ///\return true if calib should be applied, false otherwise
  bool isCalibrateBadChannels() const { return mRequest->mAskBadChannel; }

  ///\brief check if time calibration should be applied
  ///\return true if calib should be applied, false otherwise
  bool isCalibrateTime() const { return mRequest->mAskTimeCalib; }

  ///\brief check if energy/gain calibration should be applied
  ///\return true if calib should be applied, false otherwise
  bool isCalibrateGain() const { return mRequest->mAskGainCalib; }

  ///\brief check if temperature calibration should be applied
  ///\return true if calib should be applied, false otherwise
  bool isCalibrateTemperature() const { return mRequest->mAskTempCalib; }

  ///\brief get current time calibration parameters
  ///\return current time calibration parameters
  auto getTimeCalibParams() const { return mTimeCalibParams; }

  ///\brief get current bad channel calibration parameters
  ///\return current bad channel calibration parameters
  auto getBadChannelMap() const { return mBadChannelMap; }

  ///\brief get current temperature calibration parameters
  ///\return current temperature calibration parameters
  auto getTempCalibParams() const { return mTemperatureCalibParams; }

  ///\brief get current gain calibration parameters
  ///\return current gain calibration parameters
  auto getGainCalibParams() const { return mGainCalibParams; }

 private:
  EMCALCalibCCDBHelper() = default;

  std::shared_ptr<EMCALCalibRequest> mRequest; ///< EMCALCalibRequest, which calibrations have to be applied

  const o2::emcal::TimeCalibrationParams* mTimeCalibParams = nullptr;        ///< pointer to time calibration parameters
  const o2::emcal::BadChannelMap* mBadChannelMap = nullptr;                  ///< pointer to bad channel calibration parameters
  const o2::emcal::TempCalibrationParams* mTemperatureCalibParams = nullptr; ///< pointer to temperature calibration parameters
  const o2::emcal::GainCalibrationFactors* mGainCalibParams = nullptr;       ///< pointer to gain calibration parameters
};

} // namespace emcal

} // namespace o2

#endif