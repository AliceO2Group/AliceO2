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
#include "DetectorsCalibrationWorkflow/MeanVertexCalibratorSpec.h"
#include "DetectorsCalibration/Utils.h"
#include "DetectorsCalibration/MeanVertexParams.h"
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
  const o2::calibration::MeanVertexParams* params = &o2::calibration::MeanVertexParams::Instance();
  int minEnt = params->minEntries;
  int nbX = params->nbinsX;
  float rangeX = params->rangeX;
  int nbY = params->nbinsY;
  float rangeY = params->rangeY;
  int nbZ = params->nbinsZ;
  float rangeZ = params->rangeZ;
  int nSlots4SMA = params->nSlots4SMA;
  auto slotL = params->tfPerSlot;
  auto delay = params->maxTFdelay;
  auto nPointsForSlope = params->nPointsForSlope;
  mCalibrator = std::make_unique<o2::calibration::MeanVertexCalibrator>(minEnt, nbX, rangeX, nbY, rangeY, nbZ, rangeZ, nSlots4SMA, nPointsForSlope);
  mCalibrator->setSlotLength(slotL);
  mCalibrator->setMaxSlotsDelay(delay);
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
  auto data = pc.inputs().get<gsl::span<o2::dataformats::PrimaryVertex>>("input");
  o2::base::TFIDInfoHelper::fillTFIDInfo(pc, mCalibrator->getCurrentTFInfo());
  LOG(info) << "Processing TF " << mCalibrator->getCurrentTFInfo().tfCounter << " with " << data.size() << " vertices";
  mCalibrator->process(data);
  sendOutput(pc.outputs());
  const auto& infoVec = mCalibrator->getMeanVertexObjectInfoVector();
  LOG(info) << "Created " << infoVec.size() << " objects for TF " << mCalibrator->getCurrentTFInfo().tfCounter;
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

  for (uint32_t i = 0; i < payloadVec.size(); i++) {
    auto& w = infoVec[i];
    auto image = o2::ccdb::CcdbApi::createObjectImage(&payloadVec[i], &w);
    LOG(info) << "Sending object " << w.getPath() << "/" << w.getFileName() << " of size " << image->size()
              << " bytes, valid for " << w.getStartValidityTimestamp() << " : " << w.getEndValidityTimestamp();

    output.snapshot(Output{clbUtils::gDataOriginCDBPayload, "MEANVERTEX", i}, *image.get()); // vector<char>
    output.snapshot(Output{clbUtils::gDataOriginCDBWrapper, "MEANVERTEX", i}, w);            // root-serialized
  }
  if (payloadVec.size()) {
    mCalibrator->initOutput(); // reset the outputs once they are already sent
  }
}
} // namespace calibration

namespace framework
{

DataProcessorSpec getMeanVertexCalibDeviceSpec()
{

  using device = o2::calibration::MeanVertexCalibDevice;
  using clbUtils = o2::calibration::Utils;
  std::vector<InputSpec> inputs;
  inputs.emplace_back("input", "GLO", "PVTX");
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                           // orbitResetTime
                                                                true,                           // GRPECS=true
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "MEANVERTEX"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "MEANVERTEX"}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "mean-vertex-calibration",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<device>(ccdbRequest)},
    Options{{"use-verbose-mode", VariantType::Bool, false, {"Use verbose mode"}}}};
}

} // namespace framework
} // namespace o2
