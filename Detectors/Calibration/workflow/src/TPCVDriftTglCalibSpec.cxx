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

/// @file  TPCVDriftTglCalibratorSpec.cxx

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "DetectorsCalibration/TPCVDriftTglCalibration.h"
#include "DetectorsCalibrationWorkflow/TPCVDriftTglCalibSpec.h"
#include "DetectorsCalibration/Utils.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "Framework/Task.h"

using namespace o2::framework;

namespace o2
{
namespace calibration
{
class TPCVDriftTglCalibSpec : public Task
{
 public:
  TPCVDriftTglCalibSpec(int ntgl, float tglMax, int ndtgl, float dtglMax, size_t slotL, size_t minEnt)
  {
    mCalibrator = std::make_unique<o2::calibration::TPCVDriftTglCalibration>(ntgl, tglMax, ndtgl, dtglMax, slotL, minEnt);
  }

  void init(InitContext& ic) final{};

  void run(ProcessingContext& pc) final
  {
    auto tfcounter = DataRefUtils::getHeader<DataProcessingHeader*>(pc.inputs().getFirstValid(true))->startTime;
    auto data = pc.inputs().get<gsl::span<o2::dataformats::Pair<float, float>>>("input");
    LOG(INFO) << "Processing TF " << tfcounter << " with " << data.size() << " tracks";
    mCalibrator->process(tfcounter, data);
    sendOutput(pc.outputs());
  }

  void endOfStream(EndOfStreamContext& ec) final
  {
    LOG(INFO) << "Finalizing calibration";
    constexpr uint64_t INFINITE_TF = 0xffffffffffffffff;
    mCalibrator->checkSlotsToFinalize(INFINITE_TF);
    sendOutput(ec.outputs());
  }

 private:
  void sendOutput(DataAllocator& output);
  std::unique_ptr<o2::calibration::TPCVDriftTglCalibration> mCalibrator;
};

//_____________________________________________________________
void TPCVDriftTglCalibSpec::sendOutput(DataAllocator& output)
{
  // extract CCDB infos and calibration objects, convert it to TMemFile and send them to the output
  // TODO in principle, this routine is generic, can be moved to Utils.h
  using clbUtils = o2::calibration::Utils;
  const auto& payloadVec = mCalibrator->getVDPerSlot();
  auto& infoVec = mCalibrator->getCCDBInfoPerSlot(); // use non-const version as we update it
  assert(payloadVec.size() == infoVec.size());

  for (uint32_t i = 0; i < payloadVec.size(); i++) {
    auto& w = infoVec[i];
    auto image = o2::ccdb::CcdbApi::createObjectImage(&payloadVec[i], &w);
    LOG(INFO) << "Sending object " << w.getPath() << "/" << w.getFileName() << " of size " << image->size()
              << " bytes, valid for " << w.getStartValidityTimestamp() << " : " << w.getEndValidityTimestamp();
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "TPCVDTGL", i}, *image.get()); // vector<char>
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "TPCVDTGL", i}, w);            // root-serialized
  }
  if (payloadVec.size()) {
    mCalibrator->initOutput(); // reset the outputs once they are already sent
  }
}

//_____________________________________________________________
DataProcessorSpec getTPCVDriftTglCalibSpec(int ntgl, float tglMax, int ndtgl, float dtglMax, size_t slotL, size_t minEnt)
{

  using device = o2::calibration::TPCVDriftTglCalibSpec;
  using clbUtils = o2::calibration::Utils;

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "TPCVDTGL"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "TPCVDTGL"});

  return DataProcessorSpec{
    "tpc-vd-tgl-calib",
    Inputs{{"input", "GLO", "TPCITS_VDTGL", 0, Lifetime::Timeframe}},
    outputs,
    AlgorithmSpec{adaptFromTask<o2::calibration::TPCVDriftTglCalibSpec>(ntgl, tglMax, ndtgl, dtglMax, slotL, minEnt)},
    Options{}};
}

} // namespace calibration
} // namespace o2
