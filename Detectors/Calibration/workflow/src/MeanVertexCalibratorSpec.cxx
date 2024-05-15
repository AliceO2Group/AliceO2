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

/// @file   MeanVertexCalibratorSpec.cxx

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "DetectorsCalibration/MeanVertexParams.h"
#include "DetectorsCalibrationWorkflow/MeanVertexCalibratorSpec.h"
#include "DetectorsCalibration/Utils.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"

using namespace o2::framework;

namespace o2
{
namespace calibration
{
void MeanVertexCalibDevice::init(InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
  const auto& params = MeanVertexParams::Instance();
  mCalibrator = std::make_unique<o2::calibration::MeanVertexCalibrator>();
  mCalibrator->setSlotLength(params.tfPerSlot);
  mCalibrator->setMaxSlotsDelay(float(params.maxTFdelay) / params.tfPerSlot);
  bool useVerboseMode = ic.options().get<bool>("use-verbose-mode");
  LOG(info) << " ************************* Verbose? " << useVerboseMode;
  if (useVerboseMode) {
    mCalibrator->useVerboseMode(true);
  }
}

//_____________________________________________________________

void MeanVertexCalibDevice::run(o2::framework::ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  const auto& tinfo = pc.services().get<o2::framework::TimingInfo>();
  if (tinfo.globalRunNumberChanged) { // new run is starting
    mRunNumber = (tinfo.runNumber != -1 && tinfo.runNumber > 0) ? tinfo.runNumber : o2::base::GRPGeomHelper::instance().getGRPECS()->getRun();
    mFillNumber = o2::base::GRPGeomHelper::instance().getGRPLHCIF()->getFillNumber();
  }

  auto data = pc.inputs().get<gsl::span<o2::dataformats::PrimaryVertex>>("input");
  o2::base::TFIDInfoHelper::fillTFIDInfo(pc, mCalibrator->getCurrentTFInfo());
  LOG(debug) << "Processing TF " << mCalibrator->getCurrentTFInfo().tfCounter << " with " << data.size() << " vertices";
  mCalibrator->process(data);
  sendOutput(pc.outputs());
  const auto& infoVec = mCalibrator->getMeanVertexObjectInfoVector();
  LOG(detail) << "Processed TF " << mCalibrator->getCurrentTFInfo().tfCounter << " with " << data.size() << " vertices, for which we created " << infoVec.size() << " objects for TF " << mCalibrator->getCurrentTFInfo().tfCounter;
}

//_________________________________________________________________
void MeanVertexCalibDevice::finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
  o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
}

//_____________________________________________________________

void MeanVertexCalibDevice::endOfStream(o2::framework::EndOfStreamContext& ec)
{

  LOG(info) << "Finalizing calibration";
  mCalibrator->checkSlotsToFinalize(o2::calibration::INFINITE_TF);
  sendOutput(ec.outputs());
}

//_____________________________________________________________

void MeanVertexCalibDevice::sendOutput(DataAllocator& output)
{

  // extract CCDB infos and calibration objects, convert it to TMemFile and send them to the output
  // TODO in principle, this routine is generic, can be moved to Utils.h
  using clbUtils = o2::calibration::Utils;
  const auto& payloadVec = mCalibrator->getMeanVertexObjectVector();
  auto& infoVec = mCalibrator->getMeanVertexObjectInfoVector(); // use non-const version as we update it
  assert(payloadVec.size() == infoVec.size());

  if (mDCSSubSpec && mDCSSubSpec < payloadVec.size()) {
    LOGP(alarm, "Minimum subspec {} of messages for DCS CCDB is below the maximum subspec {} for production CCDB, increase the former", mDCSSubSpec, payloadVec.size());
  }
  static std::vector<char> dcsMVObj;
  for (uint32_t i = 0; i < payloadVec.size(); i++) {
    auto w = infoVec[i];
    auto& mv = payloadVec[i];
    auto image = o2::ccdb::CcdbApi::createObjectImage(&mv, &w);
    LOG(info) << (MeanVertexParams::Instance().skipObjectSending ? "Skip " : "") << "sending object "
              << w.getPath() << "/" << w.getFileName() << " of size " << image->size()
              << " bytes, valid for " << w.getStartValidityTimestamp() << " : " << w.getEndValidityTimestamp();
    if (!MeanVertexParams::Instance().skipObjectSending) {
      if (mDCSSubSpec) { // create message for DCS CCDB
        auto ts = (w.getStartValidityTimestamp() + w.getEndValidityTimestamp()) / 2;
        o2::ccdb::CcdbObjectInfo dcsw("GLO/Calib/MeanVertexCSV", "csv", fmt::format("meanvertex_{}.csv", ts), {}, w.getStartValidityTimestamp(), w.getEndValidityTimestamp());

        std::string csvMeanVertex = fmt::format("timestamp={},fillNumber={},runNumber={},x={:+.4e},y={:+.4e},z={:+.4e},sigmax={:+.4e},sigmay={:+.4e},sigmaz={:+.4e}",
                                                ts, mFillNumber, mRunNumber, mv.getX(), mv.getY(), mv.getZ(), mv.getSigmaX(), mv.getSigmaY(), mv.getSigmaZ());
        dcsMVObj.clear();
        std::copy(csvMeanVertex.begin(), csvMeanVertex.end(), std::back_inserter(dcsMVObj));
        output.snapshot(Output{clbUtils::gDataOriginCDBPayload, "MEANVERTEX_DCS", mDCSSubSpec + i}, dcsMVObj);
        output.snapshot(Output{clbUtils::gDataOriginCDBWrapper, "MEANVERTEX_DCS", mDCSSubSpec + i}, dcsw);
      }
      w.setEndValidityTimestamp(w.getEndValidityTimestamp() + o2::ccdb::CcdbObjectInfo::MONTH);
      output.snapshot(Output{clbUtils::gDataOriginCDBPayload, "MEANVERTEX", i}, *image.get()); // vector<char>
      output.snapshot(Output{clbUtils::gDataOriginCDBWrapper, "MEANVERTEX", i}, w);            // root-serialized
    }
  }
  if (payloadVec.size()) {
    mCalibrator->initOutput(); // reset the outputs once they are already sent
  }
}
} // namespace calibration

namespace framework
{

DataProcessorSpec getMeanVertexCalibDeviceSpec(uint32_t dcsMVsubspec)
{

  using device = o2::calibration::MeanVertexCalibDevice;
  using clbUtils = o2::calibration::Utils;
  std::vector<InputSpec> inputs;
  inputs.emplace_back("input", "GLO", "PVTX");
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                           // orbitResetTime
                                                                true,                           // GRPECS=true
                                                                true,                           // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "MEANVERTEX"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "MEANVERTEX"}, Lifetime::Sporadic);
  if (dcsMVsubspec) {
    outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "MEANVERTEX_DCS"}, Lifetime::Sporadic);
    outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "MEANVERTEX_DCS"}, Lifetime::Sporadic);
  }

  return DataProcessorSpec{
    "mean-vertex-calibration",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<device>(ccdbRequest, dcsMVsubspec)},
    Options{{"use-verbose-mode", VariantType::Bool, false, {"Use verbose mode"}}}};
}

} // namespace framework
} // namespace o2
