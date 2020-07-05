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
#include "Framework/Logger.h"
#include "DetectorsRaw/RDHUtils.h"

using namespace o2::framework;

namespace o2
{
namespace tof
{

using RDHUtils = o2::raw::RDHUtils;

void CompressedDecodingTask::init(InitContext& ic)
{
  LOG(INFO) << "CompressedDecoding init";

  auto finishFunction = [this]() {
    LOG(INFO) << "CompressedDecoding finish";
  };
  ic.services().get<CallbackService>().set(CallbackService::Id::Stop, finishFunction);
  mTimer.Stop();
  mTimer.Reset();
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
  mTimer.Start(false);

  /** loop over inputs routes **/
  for (auto iit = pc.inputs().begin(), iend = pc.inputs().end(); iit != iend; ++iit) {
    if (!iit.isValid())
      continue;

    /** loop over input parts **/
    for (auto const& ref : iit) {

      const auto* headerIn = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      auto payloadIn = ref.payload;
      auto payloadInSize = headerIn->payloadSize;

      DecoderBase::setDecoderBuffer(payloadIn);
      DecoderBase::setDecoderBufferSize(payloadInSize);
      DecoderBase::run();
    }
  }

  if (mNCrateOpenTF == 72 && mNCrateOpenTF == mNCrateCloseTF)
    mHasToBePosted = true;

  if (mHasToBePosted) {
    postData(pc);
  }
  mTimer.Stop();
}

void CompressedDecodingTask::endOfStream(EndOfStreamContext& ec)
{
  LOGF(INFO, "TOF CompressedDecoding total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

void CompressedDecodingTask::rdhHandler(const o2::header::RAWDataHeader* rdh)
{

  // rdh close
  const auto& rdhr = *rdh;
  if (RDHUtils::getStop(rdhr) && RDHUtils::getHeartBeatOrbit(rdhr) == o2::raw::HBFUtils::Instance().getNOrbitsPerTF() - 1 + mInitOrbit) {
    mNCrateCloseTF++;
    //    printf("New TF close RDH %d\n", int(rdh->feeId));
    return;
  }

  // rdh open
  if ((RDHUtils::getPageCounter(rdhr) == 0) && (RDHUtils::getTriggerType(rdhr) & o2::trigger::TF)) {
    mNCrateOpenTF++;
    mInitOrbit = RDHUtils::getHeartBeatOrbit(rdhr);
    //    printf("New TF open RDH %d\n", int(rdh->feeId));
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

DataProcessorSpec getCompressedDecodingSpec(const std::string& inputDesc)
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
