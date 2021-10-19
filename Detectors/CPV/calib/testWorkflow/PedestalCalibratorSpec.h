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
#include "DataFormatsCPV/Digit.h"
#include "DataFormatsCPV/TriggerRecord.h"

using namespace o2::framework;

namespace o2
{
namespace calibration
{
class CPVPedestalCalibDevice : public o2::framework::Task
{
 public:
  //_________________________________________________________________
  void init(o2::framework::InitContext& ic) final
  {
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
    LOG(INFO) << "CPVPedestalCalibDevice initialized";
    LOG(INFO) << "tf-per-slot = " << slotL;
    LOG(INFO) << "max-delay = " << delay;
    LOG(INFO) << "updateInterval = " << updateInterval;
    LOG(INFO) << "updateAtTheEndOfRunOnly = " << updateAtTheEndOfRunOnly;
  }
  //_________________________________________________________________
  void run(o2::framework::ProcessingContext& pc) final
  {
    auto tfcounter = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("digits").header)->startTime;
    auto&& digits = pc.inputs().get<gsl::span<o2::cpv::Digit>>("digits");
    auto&& trigrecs = pc.inputs().get<gsl::span<o2::cpv::TriggerRecord>>("trigrecs");
    LOG(INFO) << "Processing TF " << tfcounter << " with " << digits.size() << " digits in " << trigrecs.size() << " trigger records.";
    auto& slotTF = mCalibrator->getSlotForTF(tfcounter);

    for (auto trigrec = trigrecs.begin(); trigrec != trigrecs.end(); trigrec++) { //event loop
      // here we're filling TimeSlot event by event
      // and when last event is reached we call mCalibrator->process() to finalize the TimeSlot
      auto&& digitsInOneEvent = digits.subspan((*trigrec).getFirstEntry(), (*trigrec).getNumberOfObjects());
      if ((trigrec + 1) == trigrecs.end()) { //last event in current TF, let's process corresponding TimeSlot
        //LOG(INFO) << "last event, I call mCalibrator->process()";
        mCalibrator->process(tfcounter, digitsInOneEvent); //fill TimeSlot with digits from 1 event and check slots for finalization
      } else {
        slotTF.getContainer()->fill(digitsInOneEvent); //fill TimeSlot with digits from 1 event
      }
    }

    auto infoVecSize = mCalibrator->getCcdbInfoPedestalsVector().size();
    auto pedsVecSize = mCalibrator->getPedestalsVector().size();
    if (infoVecSize > 0) {
      LOG(INFO) << "Created " << infoVecSize << " ccdb infos and " << pedsVecSize << " pedestal objects for TF " << tfcounter;
    }
    sendOutput(pc.outputs());
  }
  //_________________________________________________________________

 private:
  std::unique_ptr<o2::cpv::PedestalCalibrator> mCalibrator;

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
      LOG(INFO) << "Sending object " << w.getPath() << "/" << w.getFileName() << " of size " << image->size()
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
  bool sendOutputWhat(const std::vector<Payload>& payloadVec, std::vector<o2::ccdb::CcdbObjectInfo>&& infoVec, header::DataDescription what, DataAllocator& output)
  {
    assert(payloadVec.size() == infoVec.size());
    if (!payloadVec.size()) {
      return false;
    }

    for (uint32_t i = 0; i < payloadVec.size(); i++) {
      auto& w = infoVec[i];
      auto image = o2::ccdb::CcdbApi::createObjectImage(&payloadVec[i], &w);
      LOG(INFO) << "Sending object " << w.getPath() << "/" << w.getFileName() << " of size " << image->size()
                << " bytes, valid for " << w.getStartValidityTimestamp() << " : " << w.getEndValidityTimestamp();
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, what, i}, *image.get()); // vector<char>
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, what, i}, w);            // root-serialized
    }

    return true;
  }
}; // class CPVPedestalCalibDevice
} // namespace calibration
namespace framework
{
DataProcessorSpec getCPVPedestalCalibDeviceSpec()
{
  using device = o2::calibration::CPVPedestalCalibDevice;

  std::vector<OutputSpec> outputs;
  // Length of data description ("CPV_Pedestals") must be < 16 characters.
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "CPV_Pedestals"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "CPV_Pedestals"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "CPV_FEEThrs"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "CPV_FEEThrs"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "CPV_PedEffs"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "CPV_PedEffs"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "CPV_DeadChnls"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "CPV_DeadChnls"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "CPV_HighThrs"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "CPV_HighThrs"});

  return DataProcessorSpec{
    "cpv-pedestal-calibration",
    Inputs{
      {"digits", "CPV", "DIGITS"},
      {"trigrecs", "CPV", "DIGITTRIGREC"}},
    outputs,
    AlgorithmSpec{adaptFromTask<device>()},
    Options{
      {"tf-per-slot", VariantType::UInt64, (uint64_t)std::numeric_limits<long>::max(), {"number of TFs per calibration time slot"}},
      {"max-delay", VariantType::UInt64, uint64_t(1000), {"number of slots in past to consider"}},
      {"updateAtTheEndOfRunOnly", VariantType::Bool, false, {"finalize the slots and prepare the CCDB entries only at the end of the run."}},
      {"updateInterval", VariantType::UInt64, (uint64_t)10, {"try to finalize the slot (and produce calibration) when the updateInterval has passed.\n To be used together with tf-per-slot = std::numeric_limits<long>::max()"}}}};
}
} // namespace framework
} // namespace o2