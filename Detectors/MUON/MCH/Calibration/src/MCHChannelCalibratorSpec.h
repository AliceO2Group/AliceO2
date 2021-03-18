// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_CALIBRATION_MCH_CHANNEL_CALIBRATOR_SPEC_H
#define O2_CALIBRATION_MCH_CHANNEL_CALIBRATOR_SPEC_H

/// @file   MCHChannelCalibratorSpec.h
/// @brief  Device to calibrate MCH channles (offsets)

#include "MCHCalibration/MCHChannelCalibrator.h"
#include "DetectorsCalibration/Utils.h"
#include "CommonUtils/MemFileHelper.h"
#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"

using namespace o2::framework;

namespace o2
{
namespace mch
{
namespace calibration
{

class MCHChannelCalibDevice : public o2::framework::Task
{
 public:
  explicit MCHChannelCalibDevice() = default;

  void init(o2::framework::InitContext& ic) final
  {
    float pedThreshold = ic.options().get<float>("pedestal-threshold");
    float noiseThreshold = ic.options().get<float>("noise-threshold");
    mCalibrator = std::make_unique<o2::mch::calibration::MCHChannelCalibrator>(pedThreshold, noiseThreshold);

    int slotL = ic.options().get<int>("tf-per-slot");
    int delay = ic.options().get<int>("max-delay");
    mCalibrator->setSlotLength(slotL);
    mCalibrator->setMaxSlotsDelay(delay);
    mCalibrator->setUpdateAtTheEndOfRunOnly();
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    auto tfcounter = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("input").header)->startTime; // is this the timestamp of the current TF?

    auto data = pc.inputs().get<gsl::span<o2::mch::calibration::PedestalDigit>>("input");
    mCalibrator->process(tfcounter, data);
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    constexpr uint64_t INFINITE_TF = 0xffffffffffffffff;
    mCalibrator->checkSlotsToFinalize(INFINITE_TF);
    mCalibrator->endOfStream();
    LOG(INFO) << "End of stream reached, sending output to CCDB";
    sendOutput(ec.outputs());
  }

 private:
  std::unique_ptr<o2::mch::calibration::MCHChannelCalibrator> mCalibrator;

  //________________________________________________________________
  void sendOutput(DataAllocator& output)
  {
    auto createBuffer = [&](auto& vec, size_t& size) {
      size = vec.empty() ? 0 : sizeof(*(vec.begin())) * vec.size();
      char* buf = nullptr;
      if (size > 0) {
        buf = (char*)malloc(size);
        if (buf) {
          char* p = buf;
          size_t sizeofElement = sizeof(*(vec.begin()));
          for (auto& element : vec) {
            memcpy(p, &element, sizeofElement);
            p += sizeofElement;
          }
        }
      }
      return buf;
    };

    // extract CCDB infos and calibration objects, convert it to TMemFile and send them to the output
    // TODO in principle, this routine is generic, can be moved to Utils.h
    using clbUtils = o2::calibration::Utils;
    const auto& payload = mCalibrator->getBadChannelsVector();
    auto& info = mCalibrator->getBadChannelsInfo(); // use non-const version as we update it
    auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
    LOG(INFO) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
              << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
    output.snapshot(Output{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBPayload, 0}, *image.get());
    output.snapshot(Output{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBInfo, 0}, info); // root-serialized

    size_t pedestalsSize;
    char* pedestalsBuffer = createBuffer(mCalibrator->getPedestalsVector(), pedestalsSize);
    auto freefct = [](void* data, void*) { free(data); };
    output.adoptChunk(Output{"MCH", "PEDESTALS", 0}, pedestalsBuffer, pedestalsSize, freefct, nullptr);

    mCalibrator->initOutput(); // reset the outputs once they are already sent
  }
};

} // namespace calibration
} // namespace mch

namespace framework
{

DataProcessorSpec getMCHChannelCalibDeviceSpec()
{
  constexpr int64_t INFINITE_TF = 0xffffffffffffffff;
  using device = o2::mch::calibration::MCHChannelCalibDevice;
  using clbUtils = o2::calibration::Utils;

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBPayload});
  outputs.emplace_back(ConcreteDataTypeMatcher{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBInfo});
  outputs.emplace_back(OutputSpec{"MCH", "PEDESTALS", 0, Lifetime::Timeframe});

  std::vector<InputSpec> inputs;
  inputs.emplace_back("input", "MCH", "PDIGITS");

  return DataProcessorSpec{
    "calib-mchchannel-calibration",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<device>()},
    Options{
      {"tf-per-slot", VariantType::Int64, INFINITE_TF - 10, {"number of TFs per calibration time slot"}},
      {"max-delay", VariantType::Int, 1, {"number of slots in past to consider"}},
      {"pedestal-threshold", VariantType::Float, 200.0f, {"maximum allowed pedestal value"}},
      {"noise-threshold", VariantType::Float, 2.0f, {"maximum allowed noise value"}},
      {"ccdb-path", VariantType::String, "http://ccdb-test.cern.ch:8080", {"Path to CCDB"}}}};
}

} // namespace framework
} // namespace o2

#endif
