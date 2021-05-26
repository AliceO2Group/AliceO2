// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

#include "FairLogger.h"
#include <fstream> // std::ifstream

using namespace o2::phos;

void PHOSTurnonCalibDevice::init(o2::framework::InitContext& ic)
{
  // int slotL = ic.options().get<int>("tf-per-slot");
  // int delay = ic.options().get<int>("max-delay");
  mCalibrator.reset(new PHOSTurnonCalibrator());

  // mCalibrator->setSlotLength(slotL);
  // mCalibrator->setMaxSlotsDelay(delay);
  mCalibrator->setUpdateAtTheEndOfRunOnly();
}
void PHOSTurnonCalibDevice::run(o2::framework::ProcessingContext& pc)
{
  auto tfcounter = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("clusters").header)->startTime; // is this the timestamp of the current TF?
  auto cells = pc.inputs().get<gsl::span<Cell>>("cells");
  auto cellTR = pc.inputs().get<gsl::span<TriggerRecord>>("cellTriggerRecords");
  auto clusters = pc.inputs().get<gsl::span<Cluster>>("clusters");
  auto cluTR = pc.inputs().get<gsl::span<TriggerRecord>>("clusterTriggerRecords");

  LOG(INFO) << "[PHOSTurnonCalibDevice - run]  Received " << cells.size() << " cells and " << clusters.size() << " clusters, running calibration";

  mCalibrator->process(tfcounter, cells, cellTR, clusters, cluTR);
}

void PHOSTurnonCalibDevice::endOfStream(o2::framework::EndOfStreamContext& ec)
{
  constexpr uint64_t INFINITE_TF = 0xffffffffffffffff;
  mCalibrator->checkSlotsToFinalize(INFINITE_TF);
  mCalibrator->endOfStream();
  mTriggerMap.reset(new TriggerMap(mCalibrator->getCalibration()));
  if (checkFitResult()) {
    //Calculate and send final object to CCDB
    auto flName = o2::ccdb::CcdbApi::generateFileName("TriggerMap");
    std::map<std::string, std::string> md;
    o2::ccdb::CcdbObjectInfo info("PHS/Calib/TriggerMap", "TriggerMap", flName, md, mRunStartTime, 99999999999999);
    info.setMetaData(md);
    auto image = o2::ccdb::CcdbApi::createObjectImage(mTriggerMap.get(), &info);

    LOG(INFO) << "Sending object " << info.getPath() << "/" << info.getFileName()
              << " of size " << image->size()
              << " bytes, valid for " << info.getStartValidityTimestamp()
              << " : " << info.getEndValidityTimestamp();

    header::DataHeader::SubSpecificationType subSpec{(header::DataHeader::SubSpecificationType)0};
    ec.outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "PHOS_Tunron", subSpec}, *image.get());
    ec.outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "PHOS_Tunron", subSpec}, info);
  } else {
    LOG(ERROR) << "Incorrect fit results";
  }
  // //Send result to QC
  // ec.outputs().snapshot(o2::framework::Output{"PHS", "TRIGMAPDIFF", 0, o2::framework::Lifetime::Timeframe}, mTrigMapDiff);
  // ec.outputs().snapshot(o2::framework::Output{"PHS", "TURNONDIFF", 0, o2::framework::Lifetime::Timeframe}, mTurnOnDiff);
}

o2::framework::DataProcessorSpec o2::phos::getPHOSTurnonCalibDeviceSpec(bool useCCDB, std::string path)
{

  std::vector<InputSpec> inputs;
  inputs.emplace_back("cells", o2::header::gDataOriginPHS, "CELLS", 0, o2::framework::Lifetime::Timeframe);
  inputs.emplace_back("cellTriggerRecords", o2::header::gDataOriginPHS, "CELLTRIGREC", 0, o2::framework::Lifetime::Timeframe);
  inputs.emplace_back("clusters", o2::header::gDataOriginPHS, "CLUSTERS", 0, o2::framework::Lifetime::Timeframe);
  inputs.emplace_back("clusterTriggerRecords", o2::header::gDataOriginPHS, "CLUSTERTRIGREC", 0, o2::framework::Lifetime::Timeframe);

  using clbUtils = o2::calibration::Utils;
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(
    ConcreteDataTypeMatcher{clbUtils::gDataOriginCDBPayload, "PHOS_Tunron"});
  outputs.emplace_back(
    ConcreteDataTypeMatcher{clbUtils::gDataOriginCDBWrapper, "PHOS_Tunron"});
  //stream for QC data
  //outputs.emplace_back("PHS", "TRIGGERQC", 0, o2::framework::Lifetime::Timeframe);

  return o2::framework::DataProcessorSpec{"PHOSTurnonCalibDevice",
                                          inputs,
                                          outputs,
                                          o2::framework::adaptFromTask<PHOSTurnonCalibDevice>(useCCDB, path),
                                          o2::framework::Options{}};
}
