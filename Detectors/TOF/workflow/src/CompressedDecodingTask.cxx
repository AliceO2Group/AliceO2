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

    /** process input **/
    auto pointer = payloadIn;
    while (pointer < (payloadIn + payloadInSize)) {
      auto rdh = reinterpret_cast<o2::header::RAWDataHeader*>(pointer);

      /** RDH close detected **/
      if (rdh->stop) {
#ifdef VERBOSE
        std::cout << "--- RDH close detected" << std::endl;
        o2::raw::HBFUtils::printRDH(*rdh);
#endif
        if (rdh->heartbeatOrbit == 255 + mInitOrbit) {
          mNCrateCloseTF++;
          printf("New TF close RDH %d\n", rdh->feeId);
        }
        pointer += rdh->offsetToNext;
        continue;
      }

#ifdef VERBOSE
      std::cout << "--- RDH open detected" << std::endl;
      o2::raw::HBFUtils::printRDH(*rdh);
#endif

      if ((rdh->pageCnt == 0) && (rdh->triggerType & o2::trigger::TF)) {
        mNCrateOpenTF++;
        mInitOrbit = rdh->heartbeatOrbit;
        printf("New TF open RDH %d\n", rdh->feeId);
      }

      pointer += rdh->headerSize;

      while (pointer < (reinterpret_cast<char*>(rdh) + rdh->memorySize)) {

        auto word = reinterpret_cast<uint32_t*>(pointer);
        if ((*word & 0x80000000) != 0x80000000) {
          printf(" %08x [ERROR] \n ", *(uint32_t*)pointer);
          return;
        }

        /** crate header detected **/
        auto crateHeader = reinterpret_cast<compressed::CrateHeader_t*>(pointer);
#ifdef VERBOSE
        printf(" %08x CrateHeader          (drmID=%d) \n ", *(uint32_t*)pointer, crateHeader->drmID);
#endif
        pointer += 4;

        /** crate orbit expected **/
        auto crateOrbit = reinterpret_cast<compressed::CrateOrbit_t*>(pointer);
#ifdef VERBOSE
        printf(" %08x CrateOrbit           (orbit=0x%08x) \n ", *(uint32_t*)pointer, crateOrbit->orbitID);
#endif
        pointer += 4;

        while (true) {
          word = reinterpret_cast<uint32_t*>(pointer);

          /** crate trailer detected **/
          if (*word & 0x80000000) {
            auto crateTrailer = reinterpret_cast<compressed::CrateTrailer_t*>(pointer);
#ifdef VERBOSE
            printf(" %08x CrateTrailer         (numberOfDiagnostics=%d) \n ", *(uint32_t*)pointer, crateTrailer->numberOfDiagnostics);
#endif
            pointer += 4;

            /** loop over diagnostics **/
            for (int i = 0; i < crateTrailer->numberOfDiagnostics; ++i) {
              auto diagnostic = reinterpret_cast<compressed::Diagnostic_t*>(pointer);
#ifdef VERBOSE
              printf(" %08x Diagnostic           (slotId=%d) \n ", *(uint32_t*)pointer, diagnostic->slotID);
#endif
              pointer += 4;
            }

            break;
          }

          /** frame header detected **/
          auto frameHeader = reinterpret_cast<compressed::FrameHeader_t*>(pointer);
#ifdef VERBOSE
          printf(" %08x FrameHeader          (numberOfHits=%d) \n ", *(uint32_t*)pointer, frameHeader->numberOfHits);
#endif
          pointer += 4;

          /** loop over hits **/
          for (int i = 0; i < frameHeader->numberOfHits; ++i) {
            auto packedHit = reinterpret_cast<compressed::PackedHit_t*>(pointer);
#ifdef VERBOSE
            printf(" %08x PackedHit            (tdcID=%d) \n ", *(uint32_t*)pointer, packedHit->tdcID);
#endif
            auto indexE = packedHit->channel +
                          8 * packedHit->tdcID +
                          120 * packedHit->chain +
                          240 * (frameHeader->trmID - 3) +
                          2400 * crateHeader->drmID;
            int time = packedHit->time;
            time += (frameHeader->frameID << 13);

            // fill hit
            mDecoder.InsertDigit(crateHeader->drmID, frameHeader->trmID, packedHit->tdcID, packedHit->chain, packedHit->channel, crateOrbit->orbitID, crateHeader->bunchID, frameHeader->frameID << 13, packedHit->time, packedHit->tot);

            pointer += 4;
          }
        }
      }

      pointer = reinterpret_cast<char*>(rdh) + rdh->offsetToNext;
    }
  }

  if (mNCrateOpenTF == 72 && mNCrateOpenTF == mNCrateCloseTF)
    mHasToBePosted = true;

  if (mHasToBePosted) {
    postData(pc);
  }
}

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
