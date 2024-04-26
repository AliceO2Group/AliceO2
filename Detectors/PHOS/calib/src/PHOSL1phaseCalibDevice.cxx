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

#include "PHOSCalibWorkflow/PHOSL1phaseCalibDevice.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataTakingContext.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "DetectorsCalibration/Utils.h"
#include "Framework/ControlService.h"

#include "FairLogger.h"

using namespace o2::phos;

void PHOSL1phaseCalibDevice::init(o2::framework::InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
  mCalibrator.reset(new PHOSL1phaseCalibrator());
  mCalibrator->setUpdateAtTheEndOfRunOnly();
}

void PHOSL1phaseCalibDevice::run(o2::framework::ProcessingContext& pc)
{

  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  auto crTime = pc.services().get<o2::framework::TimingInfo>().creation;
  if (mRunStartTime == 0 || crTime < mRunStartTime) {
    mRunStartTime = crTime;
  }
  auto tfcounter = o2::header::get<o2::header::DataHeader*>(pc.inputs().get("cells").header)->tfCounter;
  auto cells = pc.inputs().get<gsl::span<Cell>>("cells");
  auto cellTR = pc.inputs().get<gsl::span<TriggerRecord>>("cellTR");
  LOG(detail) << "Processing TF with " << cells.size() << " cells and " << cellTR.size() << " TrigRecords";
  mCalibrator->process(tfcounter, cells, cellTR);

  ++mNprocessed;
  // If sufficient statistics was collected, try to calculate L1phases
  // If statistics reached N*thresholds try to re-calculate L1phases
  // if they changed wrt previous, replace ccdb object, otherwise do nothing
  if (mNprocessed % minStatisticsForCalib == 0) {
    minStatisticsForCalib *= 2;
    mCalibrator->checkSlotsToFinalize(o2::calibration::INFINITE_TF);
    mCalibrator->endOfStream();
    if (mPrevCalibration != mCalibrator->getCalibration()) {
      mPrevCalibration = mCalibrator->getCalibration();
      // send this version of calibration to replace old one
      sendCCDB(pc.outputs());
    }
  }
}

void PHOSL1phaseCalibDevice::endOfStream(o2::framework::EndOfStreamContext& ec)
{
  mCalibrator->checkSlotsToFinalize(o2::calibration::INFINITE_TF);
  mCalibrator->endOfStream();

  // try to re-calculate L1phases. If they was not calculated yet, or
  // they changed wrt previous, replace ccdb object, otherwise do nothing

  if (mRunStartTime == 0 || mCalibrator->getCalibration() == 0) { // run not started || calibration was not produced
    return;                                                       // do not create CCDB object
  }
  // already uploaded, no need to repeat
  if (mPrevCalibration == mCalibrator->getCalibration()) {
    return;
  }
  mPrevCalibration = mCalibrator->getCalibration();
  sendCCDB(ec.outputs());
}
void PHOSL1phaseCalibDevice::sendCCDB(DataAllocator& outputs)
{

  std::vector<int> l1phase{mPrevCalibration};
  LOG(info) << "Sending L1phase to CCDB";
  // prepare all info to be sent to CCDB
  auto flName = o2::ccdb::CcdbApi::generateFileName("L1phase");
  std::map<std::string, std::string> md;
  o2::ccdb::CcdbObjectInfo info("PHS/Calib/L1phase", "L1phase", flName, md, mRunStartTime - o2::ccdb::CcdbObjectInfo::MINUTE,
                                mRunStartTime + o2::ccdb::CcdbObjectInfo::DAY);
  info.setMetaData(md);
  auto image = o2::ccdb::CcdbApi::createObjectImage(&l1phase, &info);

  LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName()
            << " of size " << image->size()
            << " bytes, valid for " << info.getStartValidityTimestamp()
            << " : " << info.getEndValidityTimestamp();

  outputs.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "PHOS_L1phase", 0}, *image.get());
  outputs.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "PHOS_L1phase", 0}, info);
  // Send summary to QC
  LOG(info) << "Sending histos to QC ";
  outputs.snapshot(o2::framework::Output{"PHS", "L1PHASEHISTO", 0}, mCalibrator->getQcHistos());
}

o2::framework::DataProcessorSpec o2::phos::getPHOSL1phaseCalibDeviceSpec()
{

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(o2::calibration::Utils::gDataOriginCDBPayload, "PHOS_L1phase", 0, Lifetime::Sporadic);
  outputs.emplace_back(o2::calibration::Utils::gDataOriginCDBWrapper, "PHOS_L1phase", 0, Lifetime::Sporadic);
  outputs.emplace_back(o2::header::gDataOriginPHS, "L1PHASEHISTO", 0, o2::framework::Lifetime::Sporadic);

  std::vector<InputSpec> inputs;
  inputs.emplace_back("cells", "PHS", "CELLS");
  inputs.emplace_back("cellTR", "PHS", "CELLTRIGREC");
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                           // orbitResetTime
                                                                true,                           // GRPECS=true
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);
  return DataProcessorSpec{
    "calib-phos-l1phase",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<PHOSL1phaseCalibDevice>(ccdbRequest)},
    Options{}};
}
