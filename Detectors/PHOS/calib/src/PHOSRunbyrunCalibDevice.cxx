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

#include "PHOSCalibWorkflow/PHOSRunbyrunCalibDevice.h"
#include "Framework/ConfigParamRegistry.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "DetectorsCalibration/Utils.h"
#include "Framework/ControlService.h"

#include "FairLogger.h"

using namespace o2::phos;

void PHOSRunbyrunCalibDevice::init(o2::framework::InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
  // int slotL = ic.options().get<int>("tf-per-slot");
  // int delay = ic.options().get<int>("max-delay");
  mCalibrator.reset(new PHOSRunbyrunCalibrator());

  // mCalibrator->setSlotLength(slotL);
  // mCalibrator->setMaxSlotsDelay(delay);
  mCalibrator->setUpdateAtTheEndOfRunOnly();
}
void PHOSRunbyrunCalibDevice::run(o2::framework::ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  if (mRunStartTime == 0) {
    mRunStartTime = pc.services().get<o2::framework::TimingInfo>().creation; // approximate time in ms
  }
  auto tfcounter = o2::header::get<o2::header::DataHeader*>(pc.inputs().get("clusters").header)->tfCounter;
  auto clusters = pc.inputs().get<gsl::span<Cluster>>("clusters");
  auto cluTR = pc.inputs().get<gsl::span<TriggerRecord>>("cluTR");
  LOG(info) << "Processing TF with " << clusters.size() << " clusters and " << cluTR.size() << " TriggerRecords";
  mCalibrator->process(tfcounter, clusters, cluTR);
}

void PHOSRunbyrunCalibDevice::endOfStream(o2::framework::EndOfStreamContext& ec)
{
  mCalibrator->checkSlotsToFinalize(o2::calibration::INFINITE_TF);
  mCalibrator->endOfStream();
  mRunByRun = mCalibrator->getCalibration();
  if (checkFitResult()) {
    LOG(info) << "End of stream reached, sending output to CCDB";
    // prepare all info to be sent to CCDB
    auto flName = o2::ccdb::CcdbApi::generateFileName("Runbyrun");
    std::map<std::string, std::string> md;
    o2::ccdb::CcdbObjectInfo info("PHS/Calib/Runbyrun", "Runbyrun", flName, md, mRunStartTime, o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP);
    info.setMetaData(md);
    auto image = o2::ccdb::CcdbApi::createObjectImage(&mRunByRun, &info);

    LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName()
              << " of size " << image->size()
              << " bytes, valid for " << info.getStartValidityTimestamp()
              << " : " << info.getEndValidityTimestamp();

    header::DataHeader::SubSpecificationType subSpec{(header::DataHeader::SubSpecificationType)0};
    ec.outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "PHOS_Runbyrun", subSpec}, *image.get());
    ec.outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "PHOS_Runbyrun", subSpec}, info);
  } else {
    LOG(error) << "Incorrect fit results: " << mRunByRun[0] << "+-" << mRunByRun[1] << ", "
               << mRunByRun[2] << "+-" << mRunByRun[3] << ", "
               << mRunByRun[4] << "+-" << mRunByRun[5] << ", "
               << mRunByRun[6] << "+-" << mRunByRun[7];
  }
  // TODO! Send mRunByRun for QC and trending plots
  //

  // Get ready for next run
  mCalibrator->initOutput(); // reset the outputs once they are already sent
}
bool PHOSRunbyrunCalibDevice::checkFitResult()
{
  bool res = true;
  const float massmin = 0.125;
  const float massmax = 0.155;
  for (int mod = 0; mod < 4; mod++) {
    res &= mRunByRun[2 * mod] < massmax && mRunByRun[2 * mod] > massmin;
  }
  return res;
}

o2::framework::DataProcessorSpec o2::phos::getPHOSRunbyrunCalibDeviceSpec(bool useCCDB)
{

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("PHS", "RUNBYRUNHISTOS", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "PHOS_Runbyrun"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "PHOS_Runbyrun"}, Lifetime::Sporadic);

  std::vector<InputSpec> inputs;
  inputs.emplace_back("clusters", "PHS", "CLUSTERS");
  inputs.emplace_back("cluTR", "PHS", "CLUSTERTRIGREC");
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                           // orbitResetTime
                                                                true,                           // GRPECS=true
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);
  return DataProcessorSpec{
    "calib-phos-runbyrun",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<PHOSRunbyrunCalibDevice>(ccdbRequest)},
    Options{}};
}
