// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  // int slotL = ic.options().get<int>("tf-per-slot");
  // int delay = ic.options().get<int>("max-delay");
  mCalibrator.reset(new PHOSRunbyrunCalibrator());

  // mCalibrator->setSlotLength(slotL);
  // mCalibrator->setMaxSlotsDelay(delay);
  mCalibrator->setUpdateAtTheEndOfRunOnly();
}
void PHOSRunbyrunCalibDevice::run(o2::framework::ProcessingContext& pc)
{
  auto tfcounter = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("clusters").header)->startTime; // is this the timestamp of the current TF?
  auto clusters = pc.inputs().get<gsl::span<Cluster>>("clusters");
  auto cluTR = pc.inputs().get<gsl::span<TriggerRecord>>("cluTR");
  LOG(INFO) << "Processing TF with " << clusters.size() << " clusters and " << cluTR.size() << " TriggerRecords";
  mCalibrator->process(tfcounter, clusters, cluTR);
}

void PHOSRunbyrunCalibDevice::endOfStream(o2::framework::EndOfStreamContext& ec)
{
  constexpr uint64_t INFINITE_TF = 0xffffffffffffffff;
  mCalibrator->checkSlotsToFinalize(INFINITE_TF);
  mCalibrator->endOfStream();
  mRunByRun = mCalibrator->getCalibration();
  if (checkFitResult()) {
    LOG(INFO) << "End of stream reached, sending output to CCDB";
    // prepare all info to be sent to CCDB
    auto flName = o2::ccdb::CcdbApi::generateFileName("Runbyrun");
    std::map<std::string, std::string> md;
    o2::ccdb::CcdbObjectInfo info("PHS/Calib/Runbyrun", "Runbyrun", flName, md, mRunStartTime, INFINITE_TF);
    info.setMetaData(md);
    auto image = o2::ccdb::CcdbApi::createObjectImage(&mRunByRun, &info);

    LOG(INFO) << "Sending object " << info.getPath() << "/" << info.getFileName()
              << " of size " << image->size()
              << " bytes, valid for " << info.getStartValidityTimestamp()
              << " : " << info.getEndValidityTimestamp();

    header::DataHeader::SubSpecificationType subSpec{(header::DataHeader::SubSpecificationType)0};
    ec.outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "PHOS_Runbyrun", subSpec}, *image.get());
    ec.outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "PHOS_Runbyrun", subSpec}, info);
  } else {
    LOG(ERROR) << "Incorrect fit results: " << mRunByRun[0] << "+-" << mRunByRun[1] << ", "
               << mRunByRun[2] << "+-" << mRunByRun[3] << ", "
               << mRunByRun[4] << "+-" << mRunByRun[5] << ", "
               << mRunByRun[6] << "+-" << mRunByRun[7];
  }
  //TODO! Send mRunByRun for QC and trending plots
  //

  //Get ready for next run
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

o2::framework::DataProcessorSpec o2::phos::getPHOSRunbyrunCalibDeviceSpec(bool useCCDB, std::string path)
{

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("PHS", "RUNBYRUNHISTOS", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "PHOS_Runbyrun"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "PHOS_Runbyrun"});

  std::vector<InputSpec> inputs;
  inputs.emplace_back("clusters", "PHS", "CLUSTERS");
  inputs.emplace_back("cluTR", "PHS", "CLUSTERTRIGREC");

  return DataProcessorSpec{
    "calib-phos-runbyrun",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<PHOSRunbyrunCalibDevice>()},
    Options{}};
}
