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

#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/CCDBParamSpec.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "DetectorsCalibration/Utils.h"
#include "CPVCalibration/NoiseCalibrator.h"
#include "CPVBase/CPVCalibParams.h"
#include "DataFormatsCPV/Digit.h"
#include "DataFormatsCPV/TriggerRecord.h"
#include "DetectorsBase/GRPGeomHelper.h"

using namespace o2::framework;

namespace o2
{
namespace calibration
{
class CPVNoiseCalibratorSpec : public o2::framework::Task
{
 public:
  CPVNoiseCalibratorSpec(std::shared_ptr<o2::base::GRPGeomRequest> req) : mCCDBRequest(req) {}

  //_________________________________________________________________
  void init(o2::framework::InitContext& ic) final
  {
    o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    auto slotL = ic.options().get<uint32_t>("tf-per-slot");
    auto delay = ic.options().get<uint32_t>("max-delay");
    auto updateInterval = ic.options().get<uint32_t>("updateInterval");
    bool updateAtTheEndOfRunOnly = ic.options().get<bool>("updateAtTheEndOfRunOnly");
    mCalibrator = std::make_unique<o2::cpv::NoiseCalibrator>();
    mCalibrator->setSlotLength(slotL);
    mCalibrator->setMaxSlotsDelay(delay);
    if (updateAtTheEndOfRunOnly) {
      mCalibrator->setUpdateAtTheEndOfRunOnly();
    }
    mCalibrator->setCheckIntervalInfiniteSlot(updateInterval);
    LOG(info) << "CPVNoiseCalibratorSpec initialized";
    LOG(info) << "tf-per-slot = " << slotL;
    LOG(info) << "max-delay = " << delay;
    LOG(info) << "updateInterval = " << updateInterval;
    LOG(info) << "updateAtTheEndOfRunOnly = " << updateAtTheEndOfRunOnly;
  }

  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final
  {
    o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
  }

  //_________________________________________________________________
  void run(o2::framework::ProcessingContext& pc) final
  {
    o2::base::GRPGeomHelper::instance().checkUpdates(pc);
    o2::base::TFIDInfoHelper::fillTFIDInfo(pc, mCalibrator->getCurrentTFInfo());
    TFType tfcounter = mCalibrator->getCurrentTFInfo().startTime;

    // update config
    static bool isConfigFetched = false;
    if (!isConfigFetched) {
      LOG(info) << "NoiseCalibratorSpec::run() : fetching o2::cpv::CPVCalibParams from CCDB";
      pc.inputs().get<o2::cpv::CPVCalibParams*>("calibparams");
      LOG(info) << "NoiseCalibratorSpec::run() : o2::cpv::CPVCalibParams::Instance() now is following:";
      o2::cpv::CPVCalibParams::Instance().printKeyValues();
      mCalibrator->configParameters();
      isConfigFetched = true;
    }

    // read pedestal efficiencies, dead and high ped channels from pedestal run
    // do it only once as they don't change during noise scan
    if (!mCalibrator->isSettedPedEfficiencies()) {
      const auto pedEffs = pc.inputs().get<std::vector<float>*>("pedeffs");
      if (pedEffs) {
        mCalibrator->setPedEfficiencies(new std::vector<float>(pedEffs->begin(), pedEffs->end()));
        LOG(info) << "NoiseCalibratorSpec()::run() : I got pedestal efficiencies vetor of size " << pedEffs->size();
      }
    }
    if (!mCalibrator->isSettedDeadChannels()) {
      // const auto deadChs = o2::framework::DataRefUtils::as<CCDBSerialized<std::vector<int>>>(pc.inputs().get("deadchs"));
      const auto deadChs = pc.inputs().get<std::vector<int>*>("deadchs");
      if (deadChs) {
        mCalibrator->setDeadChannels(new std::vector<int>(deadChs->begin(), deadChs->end()));
        LOG(info) << "NoiseCalibratorSpec()::run() : I got dead channels vetor of size " << deadChs->size();
      }
    }
    if (!mCalibrator->isSettedHighPedChannels()) {
      // const auto highPeds = o2::framework::DataRefUtils::as<CCDBSerialized<std::vector<int>>>(pc.inputs().get("highpeds"));
      const auto highPeds = pc.inputs().get<std::vector<int>*>("highpeds");
      if (highPeds) {
        mCalibrator->setHighPedChannels(new std::vector<int>(highPeds->begin(), highPeds->end()));
        LOG(info) << "NoiseCalibratorSpec()::run() : I got high pedestal channels vetor of size " << highPeds->size();
      }
    }

    // process data
    auto&& digits = pc.inputs().get<gsl::span<o2::cpv::Digit>>("digits");
    auto&& trigrecs = pc.inputs().get<gsl::span<o2::cpv::TriggerRecord>>("trigrecs");

    LOG(info) << "Processing TF " << tfcounter << " with " << digits.size() << " digits in " << trigrecs.size() << " trigger records.";
    auto& slotTF = mCalibrator->getSlotForTF(tfcounter);

    for (auto trigrec = trigrecs.begin(); trigrec != trigrecs.end(); trigrec++) { // event loop
      // here we're filling TimeSlot event by event
      // and when last event is reached we call mCalibrator->process() to finalize the TimeSlot
      auto&& digitsInOneEvent = digits.subspan((*trigrec).getFirstEntry(), (*trigrec).getNumberOfObjects());
      if ((trigrec + 1) == trigrecs.end()) { // last event in current TF, let's process corresponding TimeSlot
        // LOG(info) << "last event, I call mCalibrator->process()";
        mCalibrator->process(digitsInOneEvent); // fill TimeSlot with digits from 1 event and check slots for finalization
      } else {
        slotTF.getContainer()->fill(digitsInOneEvent); // fill TimeSlot with digits from 1 event
      }
    }

    auto infoVecSize = mCalibrator->getCcdbInfoBadChannelMapVector().size();
    auto badMapVecSize = mCalibrator->getBadChannelMapVector().size();
    if (infoVecSize > 0) {
      LOG(info) << "Created " << infoVecSize << " ccdb infos and " << badMapVecSize << " BadChannelMap objects for TF " << tfcounter;
    }
    sendOutput(pc.outputs());
  }
  //_________________________________________________________________
 private:
  std::unique_ptr<o2::cpv::NoiseCalibrator> mCalibrator;
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;

  void sendOutput(DataAllocator& output)
  {
    // extract CCDB infos and calibration objects, convert it to TMemFile and send them to the output
    // TO NOT(!) DO in principle, this routine is non-generic(!), can be moved to Utils.h
    // if there are more than 1 calibration objects to output

    // send o2::cpv::BadChannelMap
    const auto& payloadVec = mCalibrator->getBadChannelMapVector();
    auto&& infoVec = mCalibrator->getCcdbInfoBadChannelMapVector(); // use non-const version as we update it

    assert(payloadVec.size() == infoVec.size());

    for (uint32_t i = 0; i < payloadVec.size(); i++) {
      auto& w = infoVec[i];
      auto image = o2::ccdb::CcdbApi::createObjectImage(&(payloadVec[i]), &w);
      LOG(info) << "Sending object " << w.getPath() << "/" << w.getFileName() << " of size " << image->size()
                << " bytes, valid for " << w.getStartValidityTimestamp() << " : " << w.getEndValidityTimestamp();
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "CPV_BadMap", i}, *image.get()); // vector<char>
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "CPV_BadMap", i}, w);            // root-serialized
    }
    mCalibrator->initOutput(); // reset the outputs once they are already sent
  }
}; // class CPVNoiseCalibratorSpec
} // namespace calibration

namespace framework
{
DataProcessorSpec getCPVNoiseCalibratorSpec()
{
  using device = o2::calibration::CPVNoiseCalibratorSpec;

  std::vector<InputSpec> inputs;
  inputs.emplace_back("digits", "CPV", "DIGITS", 0, Lifetime::Timeframe);
  inputs.emplace_back("trigrecs", "CPV", "DIGITTRIGREC", 0, Lifetime::Timeframe);
  inputs.emplace_back("pedeffs", "CPV", "CPV_PedEffs", 0, Lifetime::Condition, ccdbParamSpec("CPV/PedestalRun/ChannelEfficiencies"));
  inputs.emplace_back("deadchs", "CPV", "CPV_DeadChnls", 0, Lifetime::Condition, ccdbParamSpec("CPV/PedestalRun/DeadChannels"));
  inputs.emplace_back("highpeds", "CPV", "CPV_HighThrs", 0, Lifetime::Condition, ccdbParamSpec("CPV/PedestalRun/HighPedChannels"));
  inputs.emplace_back("calibparams", "CPV", "CPV_CalibPars", 0, Lifetime::Condition, ccdbParamSpec("CPV/Config/CPVCalibParams"));
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                           // orbitResetTime
                                                                true,                           // GRPECS=true
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);
  std::vector<OutputSpec> outputs;
  // Length of data description ("CPV_Pedestals") must be < 16 characters.
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "CPV_BadMap"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "CPV_BadMap"}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "cpv-noise-calibration",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<device>(ccdbRequest)},
    Options{
      {"tf-per-slot", VariantType::UInt32, o2::calibration::INFINITE_TF, {"number of TFs per calibration time slot, if 0: finalize once statistics is reached"}},
      {"max-delay", VariantType::UInt32, 1000u, {"number of slots in past to consider"}},
      {"updateAtTheEndOfRunOnly", VariantType::Bool, false, {"finalize the slots and prepare the CCDB entries only at the end of the run."}},
      {"updateInterval", VariantType::UInt32, 10u, {"try to finalize the slot (and produce calibration) when the updateInterval has passed.\n To be used together with tf-per-slot = 0"}}}};
}
} // namespace framework
} // namespace o2
