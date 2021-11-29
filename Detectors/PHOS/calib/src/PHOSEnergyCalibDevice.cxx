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
#include "Framework/ConfigParamRegistry.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "DetectorsCalibration/Utils.h"
#include "Framework/ControlService.h"

#include "FairLogger.h"
#include <string>

using namespace o2::phos;

void PHOSEnergyCalibDevice::init(o2::framework::InitContext& ic)
{
  mCalibrator.reset(new PHOSEnergyCalibrator());

  //Configure output Digits file
  mCalibrator->setOutDigitsFile(mdigitsfilename);

  //read calibration and bad map objects and send them to calibrator
  if (!mBadMap) {
    if (mUseCCDB) {
      LOG(info) << "Retrieving BadMap from CCDB";
      o2::ccdb::CcdbApi ccdb;
      ccdb.init(mCCDBPath); // or http://localhost:8080 for a local installation
      std::map<std::string, std::string> metadata;
      mBadMap.reset(ccdb.retrieveFromTFileAny<BadChannelsMap>("PHS/Calib/BadChannels", metadata, mRunStartTime));

      if (!mBadMap) { //was not read from CCDB, but expected
        LOG(fatal) << "Can not read BadMap from CCDB, you may use --not-use-ccdb option to create default bad map";
      }
      //same for calibration

      mCalibParams.reset(ccdb.retrieveFromTFileAny<CalibParams>("PHS/Calib/CalibParams", metadata, mRunStartTime));
      if (!mCalibParams) { //was not read from CCDB, but expected
        LOG(fatal) << "Can not read current CalibParams from ccdb";
      }
    } else {
      LOG(info) << "Do not use CCDB, create default BadMap and calibration";
      mBadMap.reset(new BadChannelsMap(1));
      mCalibParams.reset(new CalibParams(1));
    }
  }
  mCalibrator->setBadMap(*mBadMap);
  mCalibrator->setCalibration(*mCalibParams);
  mCalibrator->setCuts(mPtMin, mEminHGTime, mEminLGTime);

  //Create geometry instance (inclusing reading mis-alignement)
  //instance will be pick up by Calibrator
  Geometry::GetInstance("Run3");
}

void PHOSEnergyCalibDevice::run(o2::framework::ProcessingContext& pc)
{
  //TODO! extract vertex information and send to Calibrator
  auto tfcounter = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("clusters").header)->startTime; // is this the timestamp of the current TF?
  const gsl::span<const Cluster>& clusters = pc.inputs().get<gsl::span<Cluster>>("clusters");
  const gsl::span<const CluElement>& cluelements = pc.inputs().get<gsl::span<CluElement>>("cluelements");
  const gsl::span<const TriggerRecord>& cluTR = pc.inputs().get<gsl::span<TriggerRecord>>("clusterTriggerRecords");

  LOG(info) << "[PHOSEnergyCalibDevice - run]  Received " << clusters.size() << " clusters and " << clusters.size() << " clusters, running calibration";
  if (mRunStartTime == 0) {
    mRunStartTime = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("clusterTriggerRecords").header)->startTime;
  }

  mCalibrator->process(tfcounter, clusters, cluelements, cluTR);
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
o2::framework::DataProcessorSpec o2::phos::getPHOSEnergyCalibDeviceSpec(bool useCCDB, std::string path, std::string digitspath,
                                                                        float ptMin, float eMinHGTime, float eMinLGTime)
{

  std::vector<InputSpec> inputs;
  inputs.emplace_back("clusters", o2::header::gDataOriginPHS, "CLUSTERS", 0, o2::framework::Lifetime::Timeframe);
  inputs.emplace_back("cluelements", o2::header::gDataOriginPHS, "CLUELEMENTS", 0, o2::framework::Lifetime::Timeframe);
  inputs.emplace_back("clusterTriggerRecords", o2::header::gDataOriginPHS, "CLUSTERTRIGREC", 0, o2::framework::Lifetime::Timeframe);

  using clbUtils = o2::calibration::Utils;
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(
    ConcreteDataTypeMatcher{clbUtils::gDataOriginCDBPayload, "PHOS_TEHistos"}, Lifetime::Sporadic);
  outputs.emplace_back(
    ConcreteDataTypeMatcher{clbUtils::gDataOriginCDBWrapper, "PHOS_TEHistos"}, Lifetime::Sporadic);
  //stream for QC data
  //outputs.emplace_back("PHS", "TRIGGERQC", 0, o2::framework::Lifetime::Timeframe);

  return o2::framework::DataProcessorSpec{"PHOSEnergyCalibDevice",
                                          inputs,
                                          outputs,
                                          o2::framework::adaptFromTask<PHOSEnergyCalibDevice>(useCCDB, path, digitspath, ptMin, eMinHGTime, eMinLGTime),
                                          o2::framework::Options{}};
}
