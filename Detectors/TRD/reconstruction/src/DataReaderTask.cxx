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
  mReader.setMaxErrWarnPrinted(ic.options().get<int>("log-max-errors"), ic.options().get<int>("log-max-warnings"));
  int nTimeBins = ic.options().get<int>("number-of-TBs");
  if (nTimeBins >= 0) {
    LOGP(info, "Number of time bins set to {} externally", nTimeBins);
    mReader.setNumberOfTimeBins(nTimeBins);
  }
  mReader.configure(mTrackletHCHeaderState, mHalfChamberWords, mHalfChamberMajor, mOptions);
  mProcessEveryNthTF = ic.options().get<int>("every-nth-tf");
}

void DataReaderTask::endOfStream(o2::framework::EndOfStreamContext& ec)
{
  LOGF(info, "At EoS we have read: %lu Digits, %lu Tracklets. Received %.3f MB input data and rejected %.3f MB",
       mDigitsTotal, mTrackletsTotal, mDatasizeInTotal / (1024. * 1024.), (float)mWordsRejectedTotal * 4. / (1024. * 1024.));
  mReader.printHalfChamberHeaderReport();
}

void DataReaderTask::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (matcher == ConcreteDataMatcher("CTP", "Trig_Offset", 0)) {
    LOG(info) << " CTP/Config/TriggerOffsets updated.";
    o2::ctp::TriggerOffsetsParam::Instance().printKeyValues();
    return;
  } else if (matcher == ConcreteDataMatcher("TRD", "LinkToHcid", 0)) {
    LOG(info) << "Updated Link ID to HCID mapping";
    mReader.setLinkMap((const o2::trd::LinkToHCIDMapping*)obj);
    return;
  }
}

void DataReaderTask::updateTimeDependentParams(framework::ProcessingContext& pc)
{
  if (!mInitOnceDone) {
    pc.inputs().get<o2::ctp::TriggerOffsetsParam*>("trigoffset");
    pc.inputs().get<o2::trd::LinkToHCIDMapping*>("linkToHcid");
    mInitOnceDone = true;
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
  const auto& tinfo = pc.services().get<o2::framework::TimingInfo>();
  if (tinfo.globalRunNumberChanged) { // new run is starting
    mInitOnceDone = false;
  }
  updateTimeDependentParams(pc);
  auto dataReadStart = std::chrono::high_resolution_clock::now();

  if ((mNTFsProcessed++ % mProcessEveryNthTF != 0) || isTimeFrameEmpty(pc)) {
    mReader.buildDPLOutputs(pc);
    mReader.reset();
    return;
  }

  size_t datasizeInTF = 0;
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
    mReader.run();
    datasizeInTF += payloadInSize;
    if (mOptions[TRDVerboseBit]) {
      LOG(info) << "relevant vectors to read : " << mReader.getTrackletsFound() << " tracklets and " << mReader.getDigitsFound() << " compressed digits";
    }
  }

  mReader.buildDPLOutputs(pc);
  std::chrono::duration<double, std::milli> dataReadTime = std::chrono::high_resolution_clock::now() - dataReadStart;
  LOGP(info, "Digits: {}, Tracklets: {}, DataRead in: {:.3f} MB, Rejected: {:.3f} kB for TF {} in {} ms",
       mReader.getDigitsFound(), mReader.getTrackletsFound(), (float)datasizeInTF / (1024. * 1024.), (float)mReader.getWordsRejected() * 4. / 1024., tfCount,
       std::chrono::duration_cast<std::chrono::milliseconds>(dataReadTime).count());
  mDigitsTotal += mReader.getDigitsFound();
  mTrackletsTotal += mReader.getTrackletsFound();
  mDatasizeInTotal += datasizeInTF;
  mWordsRejectedTotal += mReader.getWordsRejected();
  mReader.reset();
}

} // namespace o2::trd
