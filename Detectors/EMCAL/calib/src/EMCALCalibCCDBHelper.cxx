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

#include "EMCALCalib/EMCALCalibCCDBHelper.h"

namespace o2
{
namespace emcal
{

EMCALCalibRequest::EMCALCalibRequest(bool badChannel, bool timeCalib, bool gainCalib, bool tempCalib, std::vector<o2::framework::InputSpec>& inputs, bool askOnce)
  : mAskBadChannel(badChannel), mAskTimeCalib(timeCalib), mAskGainCalib(gainCalib), mAskTempCalib(tempCalib)
{
  if (mAskBadChannel) {
    addInput({"EMC_BCM", "EMC", "EMC_BADCHANNELS", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec(o2::emcal::CalibDB::getCDBPathBadChannelMap())}, inputs);
  }
  if (mAskTimeCalib) {
    addInput({"EMCAL_TCP", "EMC", "EMC_TIMECALIB", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec(o2::emcal::CalibDB::getCDBPathTimeCalibrationParams())}, inputs);
  }
  if (mAskTimeCalibSlew) {
    LOG(warning) << "EMCAL time calibration slew not yet available!";
    //   addInput({"EMC_TimeCalibParams", "EMC", "EMC_TIMECALIB", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec(o2::emcal::CalibDB::getCDBPathTimeCalibrationParams())}, inputs);
  }
  if (mAskGainCalib) {
    addInput({"EMC_GCP", "EMC", "EMC_GAINCALIB", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec(o2::emcal::CalibDB::getCDBPathGainCalibrationParams())}, inputs);
  }
  if (mAskTempCalib) {
    LOG(warning) << "EMCAL temperature calibration not yet available!";
    //   addInput({"EMC_TimeCalibParams", "EMC", "EMC_TIMECALIB", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec(o2::emcal::CalibDB::getCDBPathTimeCalibrationParams())}, inputs);
  }
}

void EMCALCalibRequest::addInput(const o2::framework::InputSpec&& isp, std::vector<o2::framework::InputSpec>& inputs)
{
  if (std::find(inputs.begin(), inputs.end(), isp) == inputs.end()) {
    inputs.emplace_back(isp);
  }
}

//=====================================================================================

void EMCALCalibCCDBHelper::setRequest(std::shared_ptr<EMCALCalibRequest> req)
{
  if (mRequest) {
    LOG(fatal) << "EMCALCalibCCDBHelper CCDB request was already set";
  }
  mRequest = req;
}

bool EMCALCalibCCDBHelper::finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{

  if (mRequest->mAskBadChannel && matcher == o2::framework::ConcreteDataMatcher("EMC", "EMC_BADCHANNELS", 0)) {
    mBadChannelMap = (o2::emcal::BadChannelMap*)obj;
    LOG(info) << "EMCAL Bad Channel Map object updated";
    return true;
  }

  if (mRequest->mAskTimeCalib && matcher == o2::framework::ConcreteDataMatcher("EMC", "EMC_TIMECALIB", 0)) {
    mTimeCalibParams = (o2::emcal::TimeCalibrationParams*)obj;
    LOG(info) << "EMCAL Time Calib Params object updated";
    return true;
  }

  if (mRequest->mAskGainCalib && matcher == o2::framework::ConcreteDataMatcher("EMC", "EMC_GAINCALIB", 0)) {
    mGainCalibParams = (o2::emcal::GainCalibrationFactors*)obj;
    LOG(info) << "EMCAL Gain Calib Params object updated";
    return true;
  }

  return false;
}

void EMCALCalibCCDBHelper::checkUpdates(o2::framework::ProcessingContext& pc) const
{
  // request input just to trigger finaliseCCDB if there was an update
  if (mRequest->mAskBadChannel) {
    pc.inputs().get<o2::emcal::BadChannelMap*>("EMC_BCM");
  }
  if (mRequest->mAskTimeCalib) {
    pc.inputs().get<o2::emcal::TimeCalibrationParams*>("EMCAL_TCP");
  }
  if (mRequest->mAskGainCalib) {
    pc.inputs().get<o2::emcal::GainCalibrationFactors*>("EMC_GCP");
  }
}

} // namespace emcal

} // namespace o2