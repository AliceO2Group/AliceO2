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
#include "TPCCalibration/TPCVDriftTglCalibration.h"
#include "TPCWorkflow/TPCVDriftTglCalibSpec.h"
#include "DetectorsCalibration/Utils.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "Framework/Task.h"

using namespace o2::framework;

namespace o2
{
namespace tpc
{
class TPCVDriftTglCalibSpec : public Task
{
 public:
  TPCVDriftTglCalibSpec(int ntgl, float tglMax, int ndtgl, float dtglMax, size_t slotL, size_t minEnt, std::shared_ptr<o2::base::GRPGeomRequest> req) : mCCDBRequest(req)
  {
    mCalibrator = std::make_unique<o2::tpc::TPCVDriftTglCalibration>(ntgl, tglMax, ndtgl, dtglMax, slotL, minEnt);
  }

  void init(InitContext& ic) final
  {
    o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    mCalibrator->setSaveHistosFile(ic.options().get<std::string>("vdtgl-histos-file-name"));
  };

  void run(ProcessingContext& pc) final
  {
    o2::base::GRPGeomHelper::instance().checkUpdates(pc);
    auto data = pc.inputs().get<gsl::span<o2::dataformats::Pair<float, float>>>("input");
    o2::base::TFIDInfoHelper::fillTFIDInfo(pc, mCalibrator->getCurrentTFInfo());
    LOG(info) << "Processing TF " << mCalibrator->getCurrentTFInfo().tfCounter << " with " << data.size() << " tracks";
    mCalibrator->process(data);
    sendOutput(pc.outputs());
  }

  void endOfStream(EndOfStreamContext& ec) final
  {
    LOG(info) << "Finalizing calibration";
    mCalibrator->checkSlotsToFinalize(o2::calibration::INFINITE_TF);
    sendOutput(ec.outputs());
  }

  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final
  {
    o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
  }

 private:
  void sendOutput(DataAllocator& output);
  std::unique_ptr<o2::tpc::TPCVDriftTglCalibration> mCalibrator;
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;
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
    LOG(info) << "Sending object " << w.getPath() << "/" << w.getFileName() << " of size " << image->size()
              << " bytes, valid for " << w.getStartValidityTimestamp() << " : " << w.getEndValidityTimestamp();
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "TPCVDTGL", i, Lifetime::Sporadic}, *image.get()); // vector<char>
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "TPCVDTGL", i, Lifetime::Sporadic}, w);            // root-serialized
  }
  if (payloadVec.size()) {
    mCalibrator->initOutput(); // reset the outputs once they are already sent
  }
}

//_____________________________________________________________
DataProcessorSpec getTPCVDriftTglCalibSpec(int ntgl, float tglMax, int ndtgl, float dtglMax, size_t slotL, size_t minEnt)
{

  using device = o2::tpc::TPCVDriftTglCalibSpec;
  using clbUtils = o2::calibration::Utils;

  std::vector<InputSpec> inputs;
  inputs.emplace_back("input", "GLO", "TPCITS_VDTGL", 0, Lifetime::Timeframe);
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                           // orbitResetTime
                                                                true,                           // GRPECS=true
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "TPCVDTGL"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "TPCVDTGL"}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "tpc-vd-tgl-calib",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<o2::tpc::TPCVDriftTglCalibSpec>(ntgl, tglMax, ndtgl, dtglMax, slotL, minEnt, ccdbRequest)},
    Options{{"vdtgl-histos-file-name", VariantType::String, "", {"file to save histos (if name provided)"}}}};
}

} // namespace tpc
} // namespace o2
