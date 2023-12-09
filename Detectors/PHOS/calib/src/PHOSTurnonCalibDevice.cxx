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

#include "PHOSCalibWorkflow/PHOSTurnonCalibDevice.h"
#include "PHOSCalibWorkflow/TurnOnHistos.h"
#include "Framework/ConfigParamRegistry.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "DetectorsCalibration/Utils.h"
#include "Framework/ControlService.h"

#include "TF1.h"
#include "TH1.h"
#include "TGraphAsymmErrors.h"

#include <fairlogger/Logger.h>
#include <fstream> // std::ifstream

using namespace o2::phos;

void PHOSTurnonCalibDevice::init(o2::framework::InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
  // int slotL = ic.options().get<int>("tf-per-slot");
  // int delay = ic.options().get<int>("max-delay");
  mCalibrator.reset(new PHOSTurnonCalibrator());

  // mCalibrator->setSlotLength(slotL);
  // mCalibrator->setMaxSlotsDelay(delay);
  mCalibrator->setUpdateAtTheEndOfRunOnly();
}
void PHOSTurnonCalibDevice::run(o2::framework::ProcessingContext& pc)
{
  auto crTime = pc.services().get<o2::framework::TimingInfo>().creation;
  if (mRunStartTime == 0 || crTime < mRunStartTime) {
    mRunStartTime = crTime;
  }
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  auto tfcounter = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("clusters").header)->startTime; // is this the timestamp of the current TF?
  auto cells = pc.inputs().get<gsl::span<Cell>>("cells");
  auto cellTR = pc.inputs().get<gsl::span<TriggerRecord>>("cellTriggerRecords");
  auto clusters = pc.inputs().get<gsl::span<Cluster>>("clusters");
  auto cluTR = pc.inputs().get<gsl::span<TriggerRecord>>("clusterTriggerRecords");

  LOG(detail) << "[PHOSTurnonCalibDevice - run]  Received " << cells.size() << " cells and " << clusters.size() << " clusters, running calibration";

  mCalibrator->process(tfcounter, cells, cellTR, clusters, cluTR);
}

void PHOSTurnonCalibDevice::endOfStream(o2::framework::EndOfStreamContext& ec)
{
  mCalibrator->checkSlotsToFinalize(o2::calibration::INFINITE_TF);
  mCalibrator->endOfStream();
  mTriggerMap.reset(new TriggerMap(mCalibrator->getCalibration()));
  if (checkFitResult()) {
    // Calculate and send final object to CCDB
    auto flName = o2::ccdb::CcdbApi::generateFileName("TriggerMap");
    std::map<std::string, std::string> md;
    o2::ccdb::CcdbObjectInfo info("PHS/Calib/TriggerMap", "TriggerMap", flName, md,
                                  mRunStartTime - o2::ccdb::CcdbObjectInfo::MINUTE, mRunStartTime + o2::ccdb::CcdbObjectInfo::YEAR);
    info.setMetaData(md);
    auto image = o2::ccdb::CcdbApi::createObjectImage(mTriggerMap.get(), &info);

    LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName()
              << " of size " << image->size()
              << " bytes, valid for " << info.getStartValidityTimestamp()
              << " : " << info.getEndValidityTimestamp();

    ec.outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "PHOS_Turnon", 0}, *image.get());
    ec.outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "PHOS_Turnon", 0}, info);
  } else {
    LOG(alarm) << "Incorrect fit results";
  }
  // //Send result to QC
  // ec.outputs().snapshot(o2::framework::Output{"PHS", "TRIGMAPDIFF", 0}, mTrigMapDiff);
  // ec.outputs().snapshot(o2::framework::Output{"PHS", "TURNONDIFF", 0}, mTurnOnDiff);
}

o2::framework::DataProcessorSpec o2::phos::getPHOSTurnonCalibDeviceSpec(bool useCCDB)
{

  std::vector<InputSpec> inputs;
  inputs.emplace_back("cells", o2::header::gDataOriginPHS, "CELLS", 0, o2::framework::Lifetime::Timeframe);
  inputs.emplace_back("cellTriggerRecords", o2::header::gDataOriginPHS, "CELLTRIGREC", 0, o2::framework::Lifetime::Timeframe);
  inputs.emplace_back("clusters", o2::header::gDataOriginPHS, "CLUSTERS", 0, o2::framework::Lifetime::Timeframe);
  inputs.emplace_back("clusterTriggerRecords", o2::header::gDataOriginPHS, "CLUSTERTRIGREC", 0, o2::framework::Lifetime::Timeframe);
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                           // orbitResetTime
                                                                true,                           // GRPECS=true
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);
  using clbUtils = o2::calibration::Utils;
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(clbUtils::gDataOriginCDBPayload, "PHOS_Turnon", 0, Lifetime::Sporadic);
  outputs.emplace_back(clbUtils::gDataOriginCDBWrapper, "PHOS_Turnon", 0, Lifetime::Sporadic);
  // stream for QC data
  // outputs.emplace_back("PHS", "TRIGGERQC", 0, o2::framework::Lifetime::Timeframe);

  return o2::framework::DataProcessorSpec{"PHOSTurnonCalibDevice",
                                          inputs,
                                          outputs,
                                          o2::framework::adaptFromTask<PHOSTurnonCalibDevice>(useCCDB, ccdbRequest),
                                          o2::framework::Options{}};
}
