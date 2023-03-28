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

/// @file   CompressedDecodingTask.cxx
/// @author Francesco Noferini
/// @since  2020-02-25
/// @brief  TOF compressed data decoding task

#include "TOFWorkflowUtils/CompressedDecodingTask.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"

#include "CommonUtils/StringUtils.h"
#include "Headers/RAWDataHeader.h"
#include "DataFormatsTOF/CompressedDataFormat.h"
#include "DetectorsRaw/HBFUtils.h"
#include "DataFormatsParameters/GRPObject.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Logger.h"
#include "DetectorsRaw/RDHUtils.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/DataRefUtils.h"
#include "CommonUtils/VerbosityConfig.h"
#include "DetectorsBase/TFIDInfoHelper.h"
#include "TOFBase/Utils.h"

using namespace o2::framework;

namespace o2
{
namespace tof
{

using RDHUtils = o2::raw::RDHUtils;

void CompressedDecodingTask::init(InitContext& ic)
{
  LOG(debug) << "CompressedDecoding init";

  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);

  mMaskNoise = ic.options().get<bool>("mask-noise");
  mNoiseRate = ic.options().get<int>("noise-counts");
  mRowFilter = ic.options().get<bool>("row-filter");

  if (mMaskNoise) {
    mDecoder.maskNoiseRate(mNoiseRate);
  } else {
    mDecoder.maskNoiseRate(-mNoiseRate); // negative means -> flag but not filter
  }

  auto finishFunction = [this]() {
    LOG(debug) << "CompressedDecoding finish";
  };
  ic.services().get<CallbackService>().set<CallbackService::Id::Stop>(finishFunction);
  mTimer.Stop();
  mTimer.Reset();
}

void CompressedDecodingTask::postData(ProcessingContext& pc)
{
  mHasToBePosted = false;
  mDecoder.fillWindows();
  mDecoder.fillDiagnosticFrequency();

  // send output message
  std::vector<o2::tof::Digit>* alldigits = mDecoder.getDigitPerTimeFrame();
  std::vector<o2::tof::ReadoutWindowData>* row = mDecoder.getReadoutWindowData();
  if (mRowFilter) {
    row = mDecoder.getReadoutWindowDataFiltered();
  }

  ReadoutWindowData* last = nullptr;
  o2::InteractionRecord lastIR;
  int lastval = 0;
  if (!row->empty()) {
    last = &row->back();
    lastval = last->first() + last->size();
    lastIR = last->mFirstIR;
  }

  /*
  int nwindowperTF = o2::tof::Utils::getNOrbitInTF() * 3;
  while (row->size() < nwindowperTF) {
    // complete timeframe with empty readout windows
    auto& dummy = row->emplace_back(lastval, 0);
    dummy.mFirstIR = lastIR;
  }
  while (row->size() > nwindowperTF) {
    // remove extra readout windows after a check they are empty
    row->pop_back();
  }
*/

  int n_tof_window = row->size();
  int n_orbits = n_tof_window / 3;
  int digit_size = alldigits->size();

  // LOG(info) << "TOF: N tof window decoded = " << n_tof_window << "(orbits = " << n_orbits << ") with " << digit_size << " digits";

  // add digits in the output snapshot
  pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "DIGITS", 0, Lifetime::Timeframe}, *alldigits);
  pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "READOUTWINDOW", 0, Lifetime::Timeframe}, *row);

  std::vector<uint8_t>& patterns = mDecoder.getPatterns();

  pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "PATTERNS", 0, Lifetime::Timeframe}, patterns);

  std::vector<uint64_t>& errors = mDecoder.getErrors();
  pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "ERRORS", 0, Lifetime::Timeframe}, errors);

  DigitHeader& digitH = mDecoder.getDigitHeader();
  pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "DIGITHEADER", 0, Lifetime::Timeframe}, digitH);

  auto diagnosticFrequency = mDecoder.getDiagnosticFrequency();
  diagnosticFrequency.setTimeStamp(mCreationTime / 1000);
  // add TFIDInfo
  o2::dataformats::TFIDInfo tfinfo;
  o2::base::TFIDInfoHelper::fillTFIDInfo(pc, tfinfo);
  diagnosticFrequency.setTFIDInfo(tfinfo);

  //diagnosticFrequency.print();
  pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "DIAFREQ", 0, Lifetime::Timeframe}, diagnosticFrequency);

  mDecoder.clear();

  mNTF++;
  mNCrateOpenTF = 0;
  mNCrateCloseTF = 0;
}

void CompressedDecodingTask::run(ProcessingContext& pc)
{
  mTimer.Start(false);

  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  const uint32_t norbits = o2::base::GRPGeomHelper::instance().getGRPECS()->getNHBFPerTF();
  Utils::setNOrbitInTF(norbits);
  mDecoder.setNOrbitInTF(norbits);

  static bool isFirstCall = true;
  if (isFirstCall) {
    isFirstCall = false;
    LOG(info) << "N orbits per TF set to " << norbits;
  } else {
    LOG(debug) << "N orbits per TF set to " << norbits;
  }

  mCreationTime = std::chrono::high_resolution_clock::now().time_since_epoch().count() / 1000000;

  //RS set the 1st orbit of the TF from the O2 header, relying on rdhHandler is not good (in fact, the RDH might be eliminated in the derived data)
  mInitOrbit = pc.services().get<o2::framework::TimingInfo>().firstTForbit;
  if (!mConetMode) {
    mDecoder.setFirstIR({0, mInitOrbit});
  }

  decodeTF(pc);

  if (!mConetMode) {
    mHasToBePosted = true;
  } else if (mNCrateOpenTF == mNCrateCloseTF) {
    mHasToBePosted = true;
  }

  if (mHasToBePosted) {
    postData(pc);
  }
  mTimer.Stop();
}

void CompressedDecodingTask::endOfStream(EndOfStreamContext& ec)
{
  LOGF(debug, "TOF CompressedDecoding total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

void CompressedDecodingTask::decodeTF(ProcessingContext& pc)
{
  auto& inputs = pc.inputs();
  const auto& tinfo = pc.services().get<o2::framework::TimingInfo>();
  // if we see requested data type input with 0xDEADBEEF subspec and 0 payload this means that the "delayed message"
  // mechanism created it in absence of real data from upstream. Processor should send empty output to not block the workflow
  {
    static size_t contDeadBeef = 0; // number of times 0xDEADBEEF was seen continuously
    std::vector<InputSpec> dummy{InputSpec{"dummy", ConcreteDataMatcher{"TOF", mDataDesc, 0xDEADBEEF}}};
    for (const auto& ref : InputRecordWalker(inputs, dummy)) {
      const auto dh = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      auto payloadSize = DataRefUtils::getPayloadSize(ref);
      if (payloadSize == 0) {
        auto maxWarn = o2::conf::VerbosityConfig::Instance().maxWarnDeadBeef;
        if (++contDeadBeef <= maxWarn) {
          LOGP(alarm, "Found input [{}/{}/{:#x}] TF#{} 1st_orbit:{} Payload {} : assuming no payload for all links in this TF{}",
               dh->dataOrigin.str, dh->dataDescription.str, dh->subSpecification, tinfo.tfCounter, tinfo.firstTForbit, payloadSize,
               contDeadBeef == maxWarn ? fmt::format(". {} such inputs in row received, stopping reporting", contDeadBeef) : "");
        }
        return;
      }
    }
    contDeadBeef = 0; // if good data, reset the counter
  }

  /** loop over inputs routes **/
  std::vector<InputSpec> sel{InputSpec{"filter", ConcreteDataTypeMatcher{"TOF", "CRAWDATA"}}};
  for (const auto& ref : InputRecordWalker(pc.inputs(), sel)) {
    //  for (auto iit = pc.inputs().begin(), iend = pc.inputs().end(); iit != iend; ++iit) {
    //    if (!iit.isValid()) {
    //      continue;
    //    }

    /** loop over input parts **/
    //    for (auto const& ref : iit) {
    const auto* headerIn = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    auto payloadIn = ref.payload;
    auto payloadInSize = DataRefUtils::getPayloadSize(ref);

    DecoderBase::setDecoderBuffer(payloadIn);
    DecoderBase::setDecoderBufferSize(payloadInSize);
    DecoderBase::run();
    //    }
  }
}

void CompressedDecodingTask::headerHandler(const CrateHeader_t* crateHeader, const CrateOrbit_t* crateOrbit)
{
  if (mConetMode) {
    if (mNCrateOpenTF == 0) {
      mInitOrbit = crateOrbit->orbitID;
      mDecoder.setFirstIR({0, mInitOrbit});
    }

    LOG(debug) << "Crate found" << crateHeader->drmID;
    mNCrateOpenTF++;
  }
}
void CompressedDecodingTask::trailerHandler(const CrateHeader_t* crateHeader, const CrateOrbit_t* crateOrbit,
                                            const CrateTrailer_t* crateTrailer, const Diagnostic_t* diagnostics,
                                            const Error_t* errors)
{
  if (mConetMode) {
    LOG(debug) << "Crate closed " << crateHeader->drmID;
    mNCrateCloseTF++;
  }

  if (mCurrentOrbit > 0) {
    mDecoder.addCrateHeaderData(mCurrentOrbit, crateHeader->drmID, crateHeader->bunchID, crateTrailer->eventCounter);
  } else {
    mDecoder.addCrateHeaderData(crateOrbit->orbitID, crateHeader->drmID, crateHeader->bunchID, crateTrailer->eventCounter);
  }

  // Diagnostics used to fill digit patterns
  auto numberOfDiagnostics = crateTrailer->numberOfDiagnostics;
  auto numberOfErrors = crateTrailer->numberOfErrors;
  for (int i = 0; i < numberOfDiagnostics; i++) {
    const uint32_t* val = reinterpret_cast<const uint32_t*>(&(diagnostics[i]));
    if (mCurrentOrbit > 0) {
      mDecoder.addPattern(*val, crateHeader->drmID, mCurrentOrbit, crateHeader->bunchID); // take orbit from crateHeader instead of crateOrbit (it can be wrong, check also for digits!)
    } else {
      mDecoder.addPattern(*val, crateHeader->drmID, crateOrbit->orbitID, crateHeader->bunchID);
    }

    /*
    int islot = (*val & 15);
    if (islot == 1) {
      if (o2::tof::diagnostic::DRM_HEADER_MISSING & *val) {
        printf("DRM_HEADER_MISSING\n");
      }
      if (o2::tof::diagnostic::DRM_TRAILER_MISSING & *val) {
        printf("DRM_TRAILER_MISSING\n");
      }
      if (o2::tof::diagnostic::DRM_FEEID_MISMATCH & *val) {
        printf("DRM_FEEID_MISMATCH\n");
      }
      if (o2::tof::diagnostic::DRM_ORBIT_MISMATCH & *val) {
        printf("DRM_ORBIT_MISMATCH\n");
      }
      if (o2::tof::diagnostic::DRM_CRC_MISMATCH & *val) {
        printf("DRM_CRC_MISMATCH\n");
      }
      if (o2::tof::diagnostic::DRM_ENAPARTMASK_DIFFER & *val) {
        printf("DRM_ENAPARTMASK_DIFFER\n");
      }
      if (o2::tof::diagnostic::DRM_CLOCKSTATUS_WRONG & *val) {
        printf("DRM_CLOCKSTATUS_WRONG\n");
      }
      if (o2::tof::diagnostic::DRM_FAULTSLOTMASK_NOTZERO & *val) {
        printf("DRM_FAULTSLOTMASK_NOTZERO\n");
      }
      if (o2::tof::diagnostic::DRM_READOUTTIMEOUT_NOTZERO & *val) {
        printf("DRM_READOUTTIMEOUT_NOTZERO\n");
      }
      if (o2::tof::diagnostic::DRM_EVENTWORDS_MISMATCH & *val) {
        printf("DRM_EVENTWORDS_MISMATCH\n");
      }
      if (o2::tof::diagnostic::DRM_MAXDIAGNOSTIC_BIT & *val) {
        printf("DRM_MAXDIAGNOSTIC_BIT\n");
      }
    } else if (islot == 2) {
      if (o2::tof::diagnostic::LTM_HEADER_MISSING & *val) {
        printf("LTM_HEADER_MISSING\n");
      }
      if (o2::tof::diagnostic::LTM_TRAILER_MISSING & *val) {
        printf("LTM_TRAILER_MISSING\n");
      }
      if (o2::tof::diagnostic::LTM_HEADER_UNEXPECTED & *val) {
        printf("LTM_HEADER_UNEXPECTED\n");
      }
      if (o2::tof::diagnostic::LTM_MAXDIAGNOSTIC_BIT & *val) {
        printf("LTM_MAXDIAGNOSTIC_BIT\n");
      }
    } else if (islot < 13) {
      if (o2::tof::diagnostic::TRM_HEADER_MISSING & *val) {
        printf("TRM_HEADER_MISSING\n");
      }
      if (o2::tof::diagnostic::TRM_TRAILER_MISSING & *val) {
        printf("TRM_TRAILER_MISSING\n");
      }
      if (o2::tof::diagnostic::TRM_CRC_MISMATCH & *val) {
        printf("TRM_CRC_MISMATCH\n");
      }
      if (o2::tof::diagnostic::TRM_HEADER_UNEXPECTED & *val) {
        printf("TRM_HEADER_UNEXPECTED\n");
      }
      if (o2::tof::diagnostic::TRM_EVENTCNT_MISMATCH & *val) {
        printf("TRM_EVENTCNT_MISMATCH\n");
      }
      if (o2::tof::diagnostic::TRM_EMPTYBIT_NOTZERO & *val) {
        printf("TRM_EMPTYBIT_NOTZERO\n");
      }
      if (o2::tof::diagnostic::TRM_LBIT_NOTZERO & *val) {
        printf("TRM_LBIT_NOTZERO\n");
      }
      if (o2::tof::diagnostic::TRM_FAULTSLOTBIT_NOTZERO & *val) {
        printf("TRM_FAULTSLOTBIT_NOTZERO\n");
      }
      if (o2::tof::diagnostic::TRM_EVENTWORDS_MISMATCH & *val) {
        printf("TRM_EVENTWORDS_MISMATCH\n");
      }
      if (o2::tof::diagnostic::TRM_DIAGNOSTIC_SPARE1 & *val) {
        printf("TRM_DIAGNOSTIC_SPARE1\n");
      }
      if (o2::tof::diagnostic::TRM_DIAGNOSTIC_SPARE2 & *val) {
        printf("TRM_DIAGNOSTIC_SPARE2\n");
      }
      if (o2::tof::diagnostic::TRM_DIAGNOSTIC_SPARE3 & *val) {
        printf("TRM_DIAGNOSTIC_SPARE3\n");
      }
      if (o2::tof::diagnostic::TRM_MAXDIAGNOSTIC_BIT & *val) {
        printf("TRM_MAXDIAGNOSTIC_BIT\n");
      }

      if (o2::tof::diagnostic::TRMCHAIN_HEADER_MISSING & *val) {
        printf("TRMCHAIN_HEADER_MISSING\n");
      }
      if (o2::tof::diagnostic::TRMCHAIN_TRAILER_MISSING & *val) {
        printf("TRMCHAIN_TRAILER_MISSING\n");
      }
      if (o2::tof::diagnostic::TRMCHAIN_STATUS_NOTZERO & *val) {
        printf("TRMCHAIN_STATUS_NOTZERO\n");
      }
      if (o2::tof::diagnostic::TRMCHAIN_EVENTCNT_MISMATCH & *val) {
        printf("TRMCHAIN_EVENTCNT_MISMATCH\n");
      }
      if (o2::tof::diagnostic::TRMCHAIN_TDCERROR_DETECTED & *val) {
        printf("TRMCHAIN_TDCERROR_DETECTED\n");
      }
      if (o2::tof::diagnostic::TRMCHAIN_BUNCHCNT_MISMATCH & *val) {
        printf("TRMCHAIN_BUNCHCNT_MISMATCH\n");
      }
      if (o2::tof::diagnostic::TRMCHAIN_DIAGNOSTIC_SPARE1 & *val) {
        printf("TRMCHAIN_DIAGNOSTIC_SPARE1\n");
      }
      if (o2::tof::diagnostic::TRMCHAIN_DIAGNOSTIC_SPARE2 & *val) {
        printf("TRMCHAIN_DIAGNOSTIC_SPARE2\n");
      }
      if (o2::tof::diagnostic::TRMCHAIN_MAXDIAGNOSTIC_BIT & *val) {
        printf("TRMCHAIN_MAXDIAGNOSTIC_BIT\n");
      }
    }
    printf("------\n");
    */
  }

  for (int i = 0; i < numberOfErrors; i++) {
    const uint32_t* val = reinterpret_cast<const uint32_t*>(&(errors[i]));
    mDecoder.addError(*val, crateHeader->drmID);
  }
}

void CompressedDecodingTask::rdhHandler(const o2::header::RAWDataHeader* rdh)
{
  const auto& rdhr = *rdh;
  // set first orbtìt here (to be check in future), please not remove this!!!
  mCurrentOrbit = RDHUtils::getHeartBeatOrbit(rdhr);

  // rdh close
  if (RDHUtils::getStop(rdhr) && RDHUtils::getHeartBeatOrbit(rdhr) == o2::tof::Utils::getNOrbitInTF() - 1 + mInitOrbit) {
    mNCrateCloseTF++;
    return;
  }

  // rdh open
  if ((RDHUtils::getPageCounter(rdhr) == 0) && (RDHUtils::getTriggerType(rdhr) & o2::trigger::TF)) {
    mNCrateOpenTF++;
  }
};

void CompressedDecodingTask::frameHandler(const CrateHeader_t* crateHeader, const CrateOrbit_t* crateOrbit,
                                          const FrameHeader_t* frameHeader, const PackedHit_t* packedHits)
{
  for (int i = 0; i < frameHeader->numberOfHits; ++i) {
    auto packedHit = packedHits + i;
    if (mCurrentOrbit > 0) {
      mDecoder.InsertDigit(crateHeader->drmID, frameHeader->trmID, packedHit->tdcID, packedHit->chain, packedHit->channel, mCurrentOrbit, crateHeader->bunchID, frameHeader->frameID << 13, packedHit->time, packedHit->tot);
    } else {
      mDecoder.InsertDigit(crateHeader->drmID, frameHeader->trmID, packedHit->tdcID, packedHit->chain, packedHit->channel, crateOrbit->orbitID, crateHeader->bunchID, frameHeader->frameID << 13, packedHit->time, packedHit->tot);
    }
  }
};

DataProcessorSpec getCompressedDecodingSpec(const std::string& inputDesc, bool conet, bool askDISTSTF)
{
  std::vector<InputSpec> inputs;
  if (askDISTSTF) {
    inputs.emplace_back("stdDist", "FLP", "DISTSUBTIMEFRAME", 0, Lifetime::Timeframe);
  }

  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                          // orbitResetTime
                                                                true,                           // GRPECS=true for nHBF per TF
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);

  //  inputs.emplace_back(std::string("x:TOF/" + inputDesc).c_str(), 0, Lifetime::Optional);
  o2::header::DataDescription dataDesc;
  dataDesc.runtimeInit(inputDesc.c_str());
  inputs.emplace_back("x", ConcreteDataTypeMatcher{o2::header::gDataOriginTOF, dataDesc}, Lifetime::Optional);

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(o2::header::gDataOriginTOF, "DIGITHEADER", 0, Lifetime::Timeframe);
  outputs.emplace_back(o2::header::gDataOriginTOF, "DIGITS", 0, Lifetime::Timeframe);
  outputs.emplace_back(o2::header::gDataOriginTOF, "READOUTWINDOW", 0, Lifetime::Timeframe);
  outputs.emplace_back(o2::header::gDataOriginTOF, "PATTERNS", 0, Lifetime::Timeframe);
  outputs.emplace_back(o2::header::gDataOriginTOF, "ERRORS", 0, Lifetime::Timeframe);
  outputs.emplace_back(o2::header::gDataOriginTOF, "DIAFREQ", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "tof-compressed-decoder",
    inputs,
    //    select(std::string("x:TOF/" + inputDesc).c_str()),
    outputs,
    AlgorithmSpec{adaptFromTask<CompressedDecodingTask>(conet, dataDesc, ccdbRequest)},
    Options{
      {"row-filter", VariantType::Bool, false, {"Filter empty row"}},
      {"mask-noise", VariantType::Bool, false, {"Flag to mask noisy digits"}},
      {"noise-counts", VariantType::Int, 11, {"Counts in a single (TF) payload"}}}};
}

} // namespace tof
} // namespace o2
