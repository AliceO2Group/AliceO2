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
#include "DataFormatsCPV/Digit.h"
#include "DataFormatsCPV/TriggerRecord.h"

using namespace o2::framework;

namespace o2
{
namespace calibration
{
class CPVNoiseCalibratorSpec : public o2::framework::Task
{
 public:
  //_________________________________________________________________
  void init(o2::framework::InitContext& ic) final
  {
    uint64_t slotL = ic.options().get<uint64_t>("tf-per-slot");
    uint64_t delay = ic.options().get<uint64_t>("max-delay");
    uint64_t updateInterval = ic.options().get<uint64_t>("updateInterval");
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
  //_________________________________________________________________
  void run(o2::framework::ProcessingContext& pc) final
  {
    auto tfcounter = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("digits").header)->startTime;
    auto&& digits = pc.inputs().get<gsl::span<o2::cpv::Digit>>("digits");
    auto&& trigrecs = pc.inputs().get<gsl::span<o2::cpv::TriggerRecord>>("trigrecs");
    //LOG(info) << "NoiseCalibratorSpec::run() : I got deadchs vector with size = " << (pc.inputs().get<std::vector<int>>("deadchs")).size();

    //std::vector<int>* const deadChs = (pc.inputs().get<std::vector<int>*>("deadchs")).size;
    /*
    static bool lStart = true;
    if (deadChs.size()) {
      LOG(info) << "NoiseCalibratorSpec::run() : I got deadchs vector with following dead channels:";
      for (int i = 0; i < deadChs.size(); i++) {
        LOG(info) << "Dead channel " << deadChs.at(i);
      }
      LOG(info) << "NoiseCalibratorSpec::run() : that's all dead channels I had";
      lStart = false;
    }
*/
    LOG(info) << "Processing TF " << tfcounter << " with " << digits.size() << " digits in " << trigrecs.size() << " trigger records.";
    auto& slotTF = mCalibrator->getSlotForTF(tfcounter);

    for (auto trigrec = trigrecs.begin(); trigrec != trigrecs.end(); trigrec++) { //event loop
      // here we're filling TimeSlot event by event
      // and when last event is reached we call mCalibrator->process() to finalize the TimeSlot
      auto&& digitsInOneEvent = digits.subspan((*trigrec).getFirstEntry(), (*trigrec).getNumberOfObjects());
      if ((trigrec + 1) == trigrecs.end()) { //last event in current TF, let's process corresponding TimeSlot
        //LOG(info) << "last event, I call mCalibrator->process()";
        mCalibrator->process(tfcounter, digitsInOneEvent); //fill TimeSlot with digits from 1 event and check slots for finalization
      } else {
        slotTF.getContainer()->fill(digitsInOneEvent); //fill TimeSlot with digits from 1 event
      }
    }

    auto infoVecSize = mCalibrator->getCcdbInfoBadChannelMapVector().size();
    auto badMapVecSize = mCalibrator->getBadChannelMapVector().size();
    if (infoVecSize > 0) {
      LOG(info) << "Created " << infoVecSize << " ccdb infos and " << badMapVecSize << " pedestal objects for TF " << tfcounter;
    }
    sendOutput(pc.outputs());
  }
  //_________________________________________________________________
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
  {
    LOG(info) << "finaliseCCDB() : I've been called with ConcreteDataMatcher" << matcher.origin.str << "/" << matcher.description.str << "/" << matcher.subSpec;
    if (matcher == ConcreteDataMatcher{"CPV", "CPV_PedEffs", 0}) {
      auto* pedEffs = static_cast<std::vector<float>*>(obj);
      mCalibrator->mPedEfficiencies = pedEffs;
      LOG(info) << "finaliseCCDB() : obtained vector of pedestal efficiencies from CCDB";
    }

    if (matcher == ConcreteDataMatcher{"CPV", "CPV_DeadChnls", 0}) {
      auto* deadCh = static_cast<std::vector<int>*>(obj);
      mCalibrator->mDeadChannels = deadCh;
      LOG(info) << "finaliseCCDB() : obtained vector of dead channels from CCDB";
    }
    if (matcher == ConcreteDataMatcher{"CPV", "CPV_HighThrs", 0}) {
      auto* highPeds = static_cast<std::vector<int>*>(obj);
      mCalibrator->mHighPedChannels = highPeds;
      LOG(info) << "finaliseCCDB() : obtained vector of high pedestal channels from CCDB";
    }
  }
  //_________________________________________________________________
 private:
  std::unique_ptr<o2::cpv::NoiseCalibrator> mCalibrator;

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
      auto image = o2::ccdb::CcdbApi::createObjectImage(&payloadVec[i], &w);
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

  std::vector<OutputSpec> outputs;
  // Length of data description ("CPV_Pedestals") must be < 16 characters.
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "CPV_BadMap"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "CPV_BadMap"}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "cpv-noise-calibration",
    inputs,
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
