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
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "DetectorsCalibration/Utils.h"
#include "CPVCalibration/PedestalCalibrator.h"
#include "CPVBase/CPVCalibParams.h"
#include "DataFormatsCPV/Digit.h"
#include "DataFormatsCPV/TriggerRecord.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "Framework/CCDBParamSpec.h"

using namespace o2::framework;

namespace o2
{
namespace calibration
{
class CPVPedestalCalibratorSpec : public o2::framework::Task
{
 public:
  CPVPedestalCalibratorSpec(std::shared_ptr<o2::base::GRPGeomRequest> req) : mCCDBRequest(req) {}
  //_________________________________________________________________
  void init(o2::framework::InitContext& ic) final
  {
    o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    uint64_t slotL = ic.options().get<uint64_t>("tf-per-slot");
    uint64_t delay = ic.options().get<uint64_t>("max-delay");
    uint64_t updateInterval = ic.options().get<uint64_t>("updateInterval");
    bool updateAtTheEndOfRunOnly = ic.options().get<bool>("updateAtTheEndOfRunOnly");
    mCalibrator = std::make_unique<o2::cpv::PedestalCalibrator>();
    mCalibrator->setSlotLength(slotL);
    mCalibrator->setMaxSlotsDelay(delay);
    if (updateAtTheEndOfRunOnly) {
      mCalibrator->setUpdateAtTheEndOfRunOnly();
    }
    mCalibrator->setCheckIntervalInfiniteSlot(updateInterval);
    LOG(info) << "CPVPedestalCalibratorSpec initialized";
    LOG(info) << "tf-per-slot = " << slotL;
    LOG(info) << "max-delay = " << delay;
    LOG(info) << "updateInterval = " << updateInterval;
    LOG(info) << "updateAtTheEndOfRunOnly = " << updateAtTheEndOfRunOnly;
  }

  //_________________________________________________________________
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
      LOG(info) << "PedestalCalibratorSpec::run() : fetching o2::cpv::CPVCalibParams from CCDB";
      pc.inputs().get<o2::cpv::CPVCalibParams*>("calibparams");
      LOG(info) << "PedestalCalibratorSpec::run() : o2::cpv::CPVCalibParams::Instance() now is following:";
      o2::cpv::CPVCalibParams::Instance().printKeyValues();
      mCalibrator->configParameters();
      isConfigFetched = true;
    }

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

    auto infoVecSize = mCalibrator->getCcdbInfoPedestalsVector().size();
    auto pedsVecSize = mCalibrator->getPedestalsVector().size();
    if (infoVecSize > 0) {
      LOG(info) << "Created " << infoVecSize << " ccdb infos and " << pedsVecSize << " pedestal objects for TF " << tfcounter;
    }
    sendOutput(pc.outputs());
  }
  //_________________________________________________________________

 private:
  std::unique_ptr<o2::cpv::PedestalCalibrator> mCalibrator;
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;

  void sendOutput(DataAllocator& output)
  {
    // extract CCDB infos and calibration objects, convert it to TMemFile and send them to the output
    // TO NOT(!) DO in principle, this routine is non-generic(!), can be moved to Utils.h
    // if there are more than 1 calibration objects to output

    // send o2::cpv::Pedestals
    /*const auto& payloadVec = mCalibrator->getPedestalsVector();
    auto&& infoVec = mCalibrator->getCcdbInfoPedestalsVector(); // use non-const version as we update it

    assert(payloadVec.size() == infoVec.size());

    for (uint32_t i = 0; i < payloadVec.size(); i++) {
      auto& w = infoVec[i];
      auto image = o2::ccdb::CcdbApi::createObjectImage(&payloadVec[i], &w);
      LOG(info) << "Sending object " << w.getPath() << "/" << w.getFileName() << " of size " << image->size()
                << " bytes, valid for " << w.getStartValidityTimestamp() << " : " << w.getEndValidityTimestamp();
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "CPV_Pedestals", i}, *image.get()); // vector<char>
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "CPV_Pedestals", i}, w);            // root-serialized
    }*/
    bool isSomethingSent = false;
    isSomethingSent = sendOutputWhat<o2::cpv::Pedestals>(mCalibrator->getPedestalsVector(), mCalibrator->getCcdbInfoPedestalsVector(), "CPV_Pedestals", output);
    isSomethingSent += sendOutputWhat<std::vector<int>>(mCalibrator->getThresholdsFEEVector(), mCalibrator->getCcdbInfoThresholdsFEEVector(), "CPV_FEEThrs", output);
    isSomethingSent += sendOutputWhat<std::vector<int>>(mCalibrator->getDeadChannelsVector(), mCalibrator->getCcdbInfoDeadChannelsVector(), "CPV_DeadChnls", output);
    isSomethingSent += sendOutputWhat<std::vector<int>>(mCalibrator->getHighPedChannelsVector(), mCalibrator->getCcdbInfoHighPedChannelsVector(), "CPV_HighThrs", output);
    isSomethingSent += sendOutputWhat<std::vector<float>>(mCalibrator->getEfficienciesVector(), mCalibrator->getCcdbInfoEfficienciesVector(), "CPV_PedEffs", output);

    if (isSomethingSent) {
      mCalibrator->initOutput(); // reset the outputs once they are already sent
    }
  }

  template <class Payload>
  bool sendOutputWhat(const std::vector<Payload>& payloadVec, std::vector<o2::ccdb::CcdbObjectInfo>& infoVec, header::DataDescription what, DataAllocator& output)
  {
    assert(payloadVec.size() == infoVec.size());
    if (!payloadVec.size()) {
      return false;
    }

    for (uint32_t i = 0; i < payloadVec.size(); i++) {
      auto& w = infoVec[i];
      auto image = o2::ccdb::CcdbApi::createObjectImage(&payloadVec[i], &w);
      LOG(info) << "Sending object " << w.getPath() << "/" << w.getFileName() << " of size " << image->size()
                << " bytes, valid for " << w.getStartValidityTimestamp() << " : " << w.getEndValidityTimestamp();
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, what, i}, *image.get()); // vector<char>
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, what, i}, w);            // root-serialized
    }

    return true;
  }
}; // class CPVPedestalCalibratorSpec
} // namespace calibration

namespace framework
{
DataProcessorSpec getCPVPedestalCalibratorSpec()
{
  using device = o2::calibration::CPVPedestalCalibratorSpec;

  std::vector<OutputSpec> outputs;
  // Length of data description ("CPV_Pedestals") must be < 16 characters.
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "CPV_Pedestals"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "CPV_Pedestals"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "CPV_FEEThrs"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "CPV_FEEThrs"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "CPV_PedEffs"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "CPV_PedEffs"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "CPV_DeadChnls"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "CPV_DeadChnls"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "CPV_HighThrs"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "CPV_HighThrs"}, Lifetime::Sporadic);
  std::vector<InputSpec> inputs{{"digits", "CPV", "DIGITS"},
                                {"trigrecs", "CPV", "DIGITTRIGREC"}};
  inputs.emplace_back("calibparams", "CPV", "CPV_CalibPars", 0, Lifetime::Condition, ccdbParamSpec("CPV/Config/CPVCalibParams"));
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                           // orbitResetTime
                                                                true,                           // GRPECS=true
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);
  return DataProcessorSpec{
    "cpv-pedestal-calibration",
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
