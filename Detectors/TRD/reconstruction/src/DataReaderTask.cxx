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

/// @file   DataReaderTask.cxx
/// @author Sean Murray
/// @brief  TRD cru output to tracklet task

#include "TRDReconstruction/DataReaderTask.h"
#include "TRDReconstruction/CruRawReader.h"

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/DataRefUtils.h"
#include "CommonUtils/VerbosityConfig.h"
#include "DataFormatsCTP/TriggerOffsetsParam.h"
#include "DataFormatsTRD/Constants.h"

namespace o2::trd
{

void DataReaderTask::init(InitContext& ic)
{
  LOG(info) << "o2::trd::DataReadTask init";

  mReader.setMaxErrWarnPrinted(ic.options().get<int>("log-max-errors"), ic.options().get<int>("log-max-warnings"));
  mDigitPreviousTotal = 0;
  mTrackletsPreviousTotal = 0;
  mWordsRead = 0;
  mWordsRejected = 0;
}

void DataReaderTask::endOfStream(o2::framework::EndOfStreamContext& ec)
{
}

void DataReaderTask::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (matcher == ConcreteDataMatcher("CTP", "Trig_Offset", 0)) {
    LOG(info) << " CTP/Config/TriggerOffsets updated.";
    o2::ctp::TriggerOffsetsParam::Instance().printKeyValues();
    return;
  }
}

void DataReaderTask::sendData(ProcessingContext& pc, bool blankframe)
{
  if (!blankframe) {
    mReader.buildDPLOutputs(pc);
  } else {
    //ensure the objects we are sending back are indeed blank.
    //TODO maybe put this in buildDPLOutputs so sending all done in 1 place, not now though.
    std::vector<Tracklet64> tracklets;
    std::vector<Digit> digits;
    std::vector<o2::trd::TriggerRecord> triggers;
    LOG(info) << "Sending data onwards with " << digits.size() << " Digits and " << tracklets.size() << " Tracklets and " << triggers.size() << " Triggers and blankframe:" << blankframe;
    pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "DIGITS", 0, Lifetime::Timeframe}, digits);
    pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "TRACKLETS", 0, Lifetime::Timeframe}, tracklets);
    pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "TRKTRGRD", 0, Lifetime::Timeframe}, triggers);
  }
}

void DataReaderTask::updateTimeDependentParams(framework::ProcessingContext& pc)
{
  static bool updateOnlyOnce = false;
  if (!updateOnlyOnce) {
    pc.inputs().get<o2::ctp::TriggerOffsetsParam*>("trigoffset");
    updateOnlyOnce = true;
  }
}

bool DataReaderTask::isTimeFrameEmpty(ProcessingContext& pc)
{
  constexpr auto origin = header::gDataOriginTRD;
  o2::framework::InputSpec dummy{"dummy", framework::ConcreteDataMatcher{origin, header::gDataDescriptionRawData, 0xDEADBEEF}};
  // if we see requested data type input with 0xDEADBEEF subspec and 0 payload.
  // frame detected we have no data and send this instead
  // send empty output so as to not block workflow
  static size_t contDeadBeef = 0; // number of times 0xDEADBEEF was seen continuously
  for (const auto& ref : o2::framework::InputRecordWalker(pc.inputs(), {dummy})) {
    const auto dh = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    auto payloadSize = o2::framework::DataRefUtils::getPayloadSize(ref);
    if (payloadSize == 0) {
      auto maxWarn = o2::conf::VerbosityConfig::Instance().maxWarnDeadBeef;
      if (++contDeadBeef <= maxWarn) {
        LOGP(alarm, "Found input [{}/{}/{:#x}] TF#{} 1st_orbit:{} Payload {} : assuming no payload for all links in this TF{}",
             dh->dataOrigin.str, dh->dataDescription.str, dh->subSpecification, dh->tfCounter, dh->firstTForbit, payloadSize,
             contDeadBeef == maxWarn ? fmt::format(". {} such inputs in row received, stopping reporting", contDeadBeef) : "");
      }
      return true;
    }
  }
  contDeadBeef = 0; // if good data, reset the counter
  return false;
}

void DataReaderTask::run(ProcessingContext& pc)
{
  //NB this is run per time frame on the epn.
  LOG(info) << "TRD Translator Task run";
  updateTimeDependentParams(pc);
  auto dataReadStart = std::chrono::high_resolution_clock::now();

  if (isTimeFrameEmpty(pc)) {
    sendData(pc, true); //send the empty tf data.
    return;
  }

  std::vector<InputSpec> sel{InputSpec{"filter", ConcreteDataTypeMatcher{"TRD", "RAWDATA"}}};
  uint64_t tfCount = 0;
  for (auto& ref : InputRecordWalker(pc.inputs(), sel)) {
    // loop over incoming HBFs from all half-CRUs (typically 128 * 72 iterations per TF)
    const auto* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    tfCount = dh->tfCounter;
    const char* payloadIn = ref.payload;
    auto payloadInSize = DataRefUtils::getPayloadSize(ref);
    if (mOptions[TRDVerboseBit]) {
      LOGP(info, "Found input [{}/{}/{:#x}] TF#{} 1st_orbit:{} Payload {} : ",
           dh->dataOrigin.str, dh->dataDescription.str, dh->subSpecification, dh->tfCounter, dh->firstTForbit, payloadInSize);
    }
    mReader.setDataBuffer(payloadIn);
    mReader.setDataBufferSize(payloadInSize);
    mReader.configure(mTrackletHCHeaderState, mHalfChamberWords, mHalfChamberMajor, mOptions);
    //mReader.setStats(&mTimeFrameStats);
    mReader.run();
    mWordsRead += mReader.getWordsRead();
    mWordsRejected += mReader.getWordsRejected();
    if (mOptions[TRDVerboseBit]) {
      LOG(info) << "relevant vectors to read : " << mReader.sumTrackletsFound() << " tracklets and " << mReader.sumDigitsFound() << " compressed digits";
    }
  }
  mWordsRead += mReader.getWordsRead();
  mWordsRejected += mReader.getWordsRejected();

  sendData(pc, false);
  std::chrono::duration<double, std::milli> dataReadTime = std::chrono::high_resolution_clock::now() - dataReadStart;
  LOGP(info, "Digits: {} ({} TF), Tracklets: {} ({} TF), DataRead in: {:.3f} MB, Rejected: {:.3f} MB for TF {} in {} ms",
       mReader.getDigitsFound(), mReader.getDigitsFound() - mDigitPreviousTotal, mReader.getTrackletsFound(),
       mReader.getTrackletsFound() - mTrackletsPreviousTotal, (float)mWordsRead * 4 / 1024.0 / 1024.0, (float)mWordsRejected * 4 / 1024.0 / 1024.0, tfCount,
       std::chrono::duration_cast<std::chrono::milliseconds>(dataReadTime).count());
  mDigitPreviousTotal = mReader.getDigitsFound();
  mTrackletsPreviousTotal = mReader.getTrackletsFound();
}

} // namespace o2::trd
