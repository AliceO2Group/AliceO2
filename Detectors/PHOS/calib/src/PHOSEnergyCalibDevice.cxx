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

#include "PHOSCalibWorkflow/PHOSEnergyCalibDevice.h"
#include "DataFormatsPHOS/TriggerRecord.h"
#include "DataFormatsPHOS/PHOSBlockHeader.h"
#include "Framework/ConfigParamRegistry.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CcdbObjectInfo.h"
#include "DetectorsCalibration/Utils.h"
#include "Framework/ControlService.h"
#include "Framework/CCDBParamSpec.h"

#include "FairLogger.h"
#include <string>

using namespace o2::phos;

void PHOSEnergyCalibDevice::init(o2::framework::InitContext& ic)
{
  mCalibrator = std::make_unique<PHOSEnergyCalibrator>();

  // read calibration and bad map objects and send them to calibrator
  if (!mHasCalib && !mUseCCDB) {
    mCalibParams = std::make_unique<CalibParams>(1); // Create test calibration coefficients
    mCalibrator->setCalibration(mCalibParams.get());
    mBadMap = std::make_unique<BadChannelsMap>(); // Create empty bad map
    mCalibrator->setBadMap(mBadMap.get());
    LOG(info) << "No reading BadMap/Calibration from ccdb requested, set default";
    mHasCalib = true;
  }
  mCalibrator->setCuts(mPtMin, mEminHGTime, mEminLGTime, mEminHGTime, mEminLGTime);

  // Create geometry instance (inclusing reading mis-alignement)
  // instance will be pick up by Calibrator
  Geometry::GetInstance("Run3");
}

void PHOSEnergyCalibDevice::run(o2::framework::ProcessingContext& pc)
{

  // Do not use ccdb if already created
  if (!mHasCalib) { // Default map and calibration was not set, use CCDB
    LOG(info) << "Getting calib from CCDB";
    // update BadMap and calibration if necessary
    auto badMapPtr = pc.inputs().get<o2::phos::BadChannelsMap*>("bdmap");
    mCalibrator->setBadMap(badMapPtr.get());

    auto calibPtr = pc.inputs().get<o2::phos::CalibParams*>("clb");
    mCalibrator->setCalibration(calibPtr.get());
    mHasCalib = true;
  }
  mOutputDigits.clear();

  auto tfcounter = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("clusters").header)->startTime; // is this the timestamp of the current TF?
  const gsl::span<const Cluster>& clusters = pc.inputs().get<gsl::span<Cluster>>("clusters");
  const gsl::span<const CluElement>& cluelements = pc.inputs().get<gsl::span<CluElement>>("cluelements");
  const gsl::span<const TriggerRecord>& cluTR = pc.inputs().get<gsl::span<TriggerRecord>>("clusterTriggerRecords");

  LOG(debug) << "[PHOSEnergyCalibDevice - run]  Received " << cluTR.size() << " TRs and " << clusters.size() << " clusters, running calibration";
  if (mRunStartTime == 0) {
    const auto ref = pc.inputs().getFirstValid(true);
    mRunStartTime = DataRefUtils::getHeader<DataProcessingHeader*>(ref)->creation; // approximate time in ms
  }
  mCalibrator->process(tfcounter, clusters, cluelements, cluTR, mOutputDigits);

  LOG(debug) << "[PHOSEnergyCalibDevice - run]  sending " << mOutputDigits.size() << " calib digits";
  pc.outputs().snapshot(o2::framework::Output{"PHS", "CALIBDIGITS", 0, o2::framework::Lifetime::Timeframe}, mOutputDigits);
}

void PHOSEnergyCalibDevice::endOfStream(o2::framework::EndOfStreamContext& ec)
{
  constexpr uint64_t INFINITE_TF = 0xffffffffffffffff;
  mCalibrator->checkSlotsToFinalize(INFINITE_TF);
  mCalibrator->endOfStream();
  if (mPostHistos) {
    postHistosCCDB(ec);
  }
}

void PHOSEnergyCalibDevice::postHistosCCDB(o2::framework::EndOfStreamContext& ec)
{
  // prepare all info to be sent to CCDB
  auto flName = o2::ccdb::CcdbApi::generateFileName("TimeEnHistos");
  std::map<std::string, std::string> md;
  o2::ccdb::CcdbObjectInfo info("PHS/Calib/TimeEnHistos", "TimeEnHistos", flName, md, mRunStartTime, 99999999999999);
  info.setMetaData(md);
  auto image = o2::ccdb::CcdbApi::createObjectImage(mCalibrator->getCollectedHistos(), &info);

  LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName()
            << " of size " << image->size()
            << " bytes, valid for " << info.getStartValidityTimestamp()
            << " : " << info.getEndValidityTimestamp();

  header::DataHeader::SubSpecificationType subSpec{(header::DataHeader::SubSpecificationType)0};
  ec.outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "PHOS_TEHistos", subSpec}, *image.get());
  ec.outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "PHOS_TEHistos", subSpec}, info);
}
o2::framework::DataProcessorSpec o2::phos::getPHOSEnergyCalibDeviceSpec(bool useCCDB, float ptMin, float eMinHGTime,
                                                                        float eMinLGTime, float edigMin, float ecluMin)
{

  std::vector<InputSpec> inputs;
  inputs.emplace_back("clusters", o2::header::gDataOriginPHS, "CLUSTERS", 0, o2::framework::Lifetime::Timeframe);
  inputs.emplace_back("cluelements", o2::header::gDataOriginPHS, "CLUELEMENTS", 0, o2::framework::Lifetime::Timeframe);
  inputs.emplace_back("clusterTriggerRecords", o2::header::gDataOriginPHS, "CLUSTERTRIGREC", 0, o2::framework::Lifetime::Timeframe);
  if (useCCDB) {
    inputs.emplace_back("bdmap", o2::header::gDataOriginPHS, "PHS_BM", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec("PHS/Calib/BadMap"));
    inputs.emplace_back("clb", o2::header::gDataOriginPHS, "PHS_Calibr", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec("PHS/Calib/CalibParams"));
  }
  using clbUtils = o2::calibration::Utils;
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{clbUtils::gDataOriginCDBPayload, "PHOS_TEHistos"}, o2::framework::Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{clbUtils::gDataOriginCDBWrapper, "PHOS_TEHistos"}, o2::framework::Lifetime::Sporadic);
  outputs.emplace_back("PHS", "CALIBDIGITS", 0, o2::framework::Lifetime::Timeframe);

  return o2::framework::DataProcessorSpec{"PHOSEnergyCalibDevice",
                                          inputs,
                                          outputs,
                                          o2::framework::adaptFromTask<PHOSEnergyCalibDevice>(useCCDB, ptMin, eMinHGTime, eMinLGTime, edigMin, ecluMin),
                                          o2::framework::Options{}};
}
