// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

  const o2::calibration::MeanVertexParams* params = &o2::calibration::MeanVertexParams::Instance();
  int minEnt = params->minEntries;
  int nbX = params->nbinsX;
  float rangeX = params->rangeX;
  int nbY = params->nbinsY;
  float rangeY = params->rangeY;
  int nbZ = params->nbinsZ;
  float rangeZ = params->rangeZ;
  int nSlots4SMA = params->nSlots4SMA;
  bool useFit = params->useFit;
  int slotL = params->tfPerSlot;
  int delay = params->maxTFdelay;
  mCalibrator = std::make_unique<o2::calibration::MeanVertexCalibrator>(minEnt, useFit, nbX, rangeX, nbY, rangeY, nbZ, rangeZ, nSlots4SMA);
  mCalibrator->setSlotLength(slotL);
  mCalibrator->setMaxSlotsDelay(delay);
}

//_____________________________________________________________

void MeanVertexCalibDevice::run(o2::framework::ProcessingContext& pc)
{

  auto tfcounter = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("input").header)->startTime;
  auto data = pc.inputs().get<gsl::span<o2::dataformats::PrimaryVertex>>("input");
  LOG(INFO) << "Processing TF " << tfcounter << " with " << data.size() << " tracks";
  mCalibrator->process(tfcounter, data);
  sendOutput(pc.outputs());
  const auto& infoVec = mCalibrator->getMeanVertexObjectInfoVector();
  LOG(INFO) << "Created " << infoVec.size() << " objects for TF " << tfcounter;
}

//_____________________________________________________________

void MeanVertexCalibDevice::endOfStream(o2::framework::EndOfStreamContext& ec)
{

  LOG(INFO) << "Finalizing calibration";
  constexpr uint64_t INFINITE_TF = 0xffffffffffffffff;
  mCalibrator->checkSlotsToFinalize(INFINITE_TF);
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
    LOG(INFO) << "Sending object " << w.getPath() << "/" << w.getFileName() << " of size " << image->size()
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

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "MEANVERTEX"});
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "MEANVERTEX"});

  return DataProcessorSpec{
    "mean-vertex-calibration",
    Inputs{{"input", "GLO", "PVTX"}},
    outputs,
    AlgorithmSpec{adaptFromTask<device>()},
    Options{
      {}}};
}

} // namespace framework
} // namespace o2
