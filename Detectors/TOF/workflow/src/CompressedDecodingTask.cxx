// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   CompressedDecodingTask.cxx
/// @author Francesco Noferini
/// @since  2020-02-25
/// @brief  TOF compressed data decoding task

#include "TOFWorkflow/CompressedDecodingTask.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"

#include "Headers/RAWDataHeader.h"
#include "DataFormatsTOF/CompressedDataFormat.h"
#include "DetectorsRaw/HBFUtils.h"
#include "DataFormatsParameters/GRPObject.h"
#include "Framework/WorkflowSpec.h"

using namespace o2::framework;

namespace o2
{
namespace tof
{

void CompressedDecodingTask::init(InitContext& ic)
{
  LOG(INFO) << "CompressedDecoding init";

  auto finishFunction = [this]() {
    LOG(INFO) << "CompressedDecoding finish";
  };
  ic.services().get<CallbackService>().set(CallbackService::Id::Stop, finishFunction);
}

void CompressedDecodingTask::postData(ProcessingContext& pc)
{
  mHasToBePosted = false;
  mDecoder.FillWindows();

  // send output message
  std::vector<o2::tof::Digit>* alldigits = mDecoder.getDigitPerTimeFrame();
  std::vector<o2::tof::ReadoutWindowData>* row = mDecoder.getReadoutWindowData();

  int n_tof_window = row->size();
  int n_orbits = n_tof_window / 3;
  int digit_size = alldigits->size();

  LOG(INFO) << "TOF: N tof window decoded = " << n_tof_window << "(orbits = " << n_orbits << ") with " << digit_size << " digits";

  // add digits in the output snapshot
  pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "DIGITS", 0, Lifetime::Timeframe}, *alldigits);
  pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "READOUTWINDOW", 0, Lifetime::Timeframe}, *row);

  static o2::parameters::GRPObject::ROMode roMode = o2::parameters::GRPObject::CONTINUOUS;

  LOG(INFO) << "TOF: Sending ROMode= " << roMode << " to GRPUpdater";
  pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "ROMode", 0, Lifetime::Timeframe}, roMode);

  mDecoder.clear();

  LOG(INFO) << "TOF: TF = " << mNTF << " - Crate in " << mNCrateOpenTF;

  mNTF++;
  mNCrateOpenTF = 0;
  mNCrateCloseTF = 0;
}

void CompressedDecodingTask::run(ProcessingContext& pc)
{
  LOG(INFO) << "CompressedDecoding run";

  /** check status **/
  if (mStatus) {
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    return;
  }

  /** receive input **/
  for (auto& input : pc.inputs()) {

    /** input **/
    const auto* headerIn = DataRefUtils::getHeader<o2::header::DataHeader*>(input);
    auto payloadIn = const_cast<char*>(input.payload);
    auto payloadInSize = headerIn->payloadSize;

    DecoderBase::setDecoderBuffer(payloadIn);
    DecoderBase::setDecoderBufferSize(payloadInSize);
    DecoderBase::run();
  }

  if (mNCrateOpenTF == 72 && mNCrateOpenTF == mNCrateCloseTF)
    mHasToBePosted = true;

  if (mHasToBePosted) {
    postData(pc);
  }
}

void CompressedDecodingTask::rdhHandler(const o2::header::RAWDataHeader* rdh)
{

  // rdh close
  if (rdh->stop && rdh->heartbeatOrbit == 255 + mInitOrbit) {
    mNCrateCloseTF++;
    printf("New TF close RDH %d\n", rdh->feeId);
    return;
  }

  // rdh open
  if ((rdh->pageCnt == 0) && (rdh->triggerType & o2::trigger::TF)) {
    mNCrateOpenTF++;
    mInitOrbit = rdh->heartbeatOrbit;
    printf("New TF open RDH %d\n", rdh->feeId);
  }
};

void CompressedDecodingTask::frameHandler(const CrateHeader_t* crateHeader, const CrateOrbit_t* crateOrbit,
                                          const FrameHeader_t* frameHeader, const PackedHit_t* packedHits)
{
  for (int i = 0; i < frameHeader->numberOfHits; ++i) {
    auto packedHit = packedHits + i;
    mDecoder.InsertDigit(crateHeader->drmID, frameHeader->trmID, packedHit->tdcID, packedHit->chain, packedHit->channel, crateOrbit->orbitID, crateHeader->bunchID, frameHeader->frameID << 13, packedHit->time, packedHit->tot);
  }
};

DataProcessorSpec getCompressedDecodingSpec(std::string inputDesc)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(o2::header::gDataOriginTOF, "DIGITS", 0, Lifetime::Timeframe);
  outputs.emplace_back(o2::header::gDataOriginTOF, "READOUTWINDOW", 0, Lifetime::Timeframe);
  outputs.emplace_back(o2::header::gDataOriginTOF, "ROMode", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "tof-compressed-decoder",
    select(std::string("x:TOF/" + inputDesc).c_str()),
    outputs,
    AlgorithmSpec{adaptFromTask<CompressedDecodingTask>()},
    Options{}};
}

} // namespace tof
} // namespace o2
