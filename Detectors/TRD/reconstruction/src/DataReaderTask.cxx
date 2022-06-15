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

#include "DataFormatsTRD/Constants.h"
#include <TH3F.h>
#include "TH2F.h"
#include "TFile.h"

//using namespace o2::framework;

namespace o2::trd
{

void DataReaderTask::setParsingErrorLabels()
{
  //TODO this is not working, come back at some point fix, not essential.
  mParsingErrors->GetXaxis()->SetBinLabel(TRDParsingNoError, "TRDParsingNoError");
  mParsingErrors->GetXaxis()->SetBinLabel(TRDParsingUnrecognisedVersion, "TRDParsingUnrecognisedVersion");
  mParsingErrors->GetXaxis()->SetBinLabel(TRDParsingBadDigt, "TRDParsingBadDigt");
  mParsingErrors->GetXaxis()->SetBinLabel(TRDParsingBadTracklet, "TRDParsingBadTracklet");
  mParsingErrors->GetXaxis()->SetBinLabel(TRDParsingDigitEndMarkerWrongState, "TRDParsingDigitEndMarkerWrongState");
  mParsingErrors->GetXaxis()->SetBinLabel(TRDParsingDigitMCMHeaderSanityCheckFailure, "TRDParsingDigitMCMHeaderSanityCheckFailure");
  mParsingErrors->GetXaxis()->SetBinLabel(TRDParsingDigitROBDecreasing, "TRDParsingDigitROBDecreasing");
  mParsingErrors->GetXaxis()->SetBinLabel(TRDParsingDigitMCMNotIncreasing, "TRDParsingDigitMCMNotIncreasing");
  mParsingErrors->GetXaxis()->SetBinLabel(TRDParsingDigitADCMaskMismatch, "TRDParsingDigitADCMaskMismatch");
  mParsingErrors->GetXaxis()->SetBinLabel(TRDParsingDigitADCMaskAdvanceToEnd, "TRDParsingDigitADCMaskAdvanceToEnd");
  mParsingErrors->GetXaxis()->SetBinLabel(TRDParsingDigitMCMHeaderBypassButStateMCMHeader, "TRDParsingDigitMCMHeaderBypassButStateMCMHeader");
  mParsingErrors->GetXaxis()->SetBinLabel(TRDParsingDigitEndMarkerStateButReadingMCMADCData, "TRDParsingDigitEndMarkerStateButReadingMCMADCData");
  /*                       TRDParsingDigitADCChannel21,
                           TRDParsingDigitADCChannelGT22,
                           TRDParsingDigitGT10ADCs,
                           TRDParsingDigitSanityCheck,
                           TRDParsingDigitExcessTimeBins,
                           TRDParsingDigitParsingExitInWrongState,
                           TRDParsingDigitStackMisMatch,
                           TRDParsingDigitLayerMisMatch,
                           TRDParsingDigitSectorMisMatch,
                           TRDParsingTrackletCRUPaddingWhileParsingTracklets,
                           TRDParsingTrackletBit11NotSetInTrackletHCHeader,
                           TRDParsingTrackletHCHeaderSanityCheckFailure,
                           TRDParsingTrackletMCMHeaderSanityCheckFailure,
                           TRDParsingTrackletMCMHeaderButParsingMCMData,
                           TRDParsingTrackletStateMCMHeaderButParsingMCMData,
                           TRDParsingTrackletTrackletCountGTThatDeclaredInMCMHeader,
                           TRDParsingTrackletInvalidTrackletCount,
                           TRDParsingTrackletPadRowIncreaseError,
                           TRDParsingTrackletColIncreaseError,
                           TRDParsingTrackletNoTrackletEndMarker,
                           TRDParsingTrackletExitingNoTrackletEndMarker
                           */
}

void DataReaderTask::buildHistograms()
{
  if (mRootOutput) {
    mParseErrors = new TList();
    mLinkErrors = new TList();
    mRootFile = new TFile(mHistogramsFilename.c_str(), "recreate"); // make this changeable from command line
    std::array<std::string, 10> linkerrortitles = {"Count of Link had no errors during tf",
                                                   "Count of # times Linkerrors 0x1 seen per tf",
                                                   "Count of # time Linkerrors 0x2 seen per tf",
                                                   "Count of any Linkerror seen during tf",
                                                   "Link was seen with no data (empty) in a tf",
                                                   "Link was seen with data during a tf",
                                                   "Links seen with corrupted data during tf",
                                                   "Links seen with out corrupted data during tf", "", ""};
    //lets hack this for some graphs
    mTimeFrameTime = new TH1F("timeframetime", "Time taken per time frame", 10000, 0, 10000);
    mTrackletParsingTime = new TH1F("tracklettime", "Time taken per tracklet block", 1000, 0, 1000);
    mDigitParsingTime = new TH1F("digittime", "Time taken per digit block", 1000, 0, 1000);
    mCruTime = new TH1F("crutime", "Time taken per cru link", 1000, 0, 1000);
    mPackagingTime = new TH1F("packagingtime", "Time to package the eventrecord and copy the output", 1000, 0, 1000);
    mDataVersions = new TH1F("dataversions", "Data versions major.minor seen in data", 65000, 0, 65000);
    mDataVersionsMajor = new TH1F("dataversionsmajor", "Data versions major", 256, 0, 256);
    mParsingErrors = new TH1F("parseerrors", "Parsing Errors seen in data", 256, 0, 256);
    int count = 0;
    for (int count = 0; count < constants::MAXPARSEERRORHISTOGRAMS; ++count) {
      std::string label = fmt::format("parsingerrors_{0}", count);
      std::string title = fmt::format("linkerrors_{0}", count);
      TH2F* h = new TH2F(label.c_str(), title.c_str(), 36, 0, 36, 30, 0, 30);
      mParseErrors->Add(h);
    }
    count = 0;
    for (int count = 0; count < constants::MAXLINKERRORHISTOGRAMS; ++count) {
      std::string label = fmt::format("linkerrors_{0}", count);
      std::string title = linkerrortitles[count];
      TH2F* h = new TH2F(label.c_str(), title.c_str(), 36, 0, 36, 30, 0, 30);
      mLinkErrors->Add(h);
    }
    mTimeFrameTime->GetXaxis()->SetTitle("Time taken in ms");
    mCruTime->GetXaxis()->SetTitle("Time taken in ms");
    mTrackletParsingTime->GetXaxis()->SetTitle("Time taken in #mus");
    mDigitParsingTime->GetXaxis()->SetTitle("Time taken in #mus");
    mPackagingTime->GetXaxis()->SetTitle("Time taken in #mus");
    mTimeFrameTime->GetYaxis()->SetTitle("Counts");
    mTrackletParsingTime->GetYaxis()->SetTitle("Counts");
    mDigitParsingTime->GetYaxis()->SetTitle("Counts");
    mCruTime->GetYaxis()->SetTitle("Counts");
    mPackagingTime->GetYaxis()->SetTitle("Counts");
    mDataVersions->GetYaxis()->SetTitle("Counts");
    mDataVersions->GetYaxis()->SetTitle("Counts");
    mDataVersionsMajor->GetYaxis()->SetTitle("Counts");
    mDataVersionsMajor->GetXaxis()->SetTitle("Version major");
    mParsingErrors->GetYaxis()->SetTitle("Erorr Types");
    mParsingErrors->GetXaxis()->SetTitle("Erorr Types");
    for (int count = 0; count < constants::MAXPARSEERRORHISTOGRAMS; ++count) {
      TH2F* h = (TH2F*)mParseErrors->At(count);
      h->GetXaxis()->SetTitle("Sector*2 + side");
      h->GetXaxis()->CenterTitle(kTRUE);
      h->GetYaxis()->SetTitle("Stack_Layer");
      h->GetYaxis()->CenterTitle(kTRUE);
    }
    for (int count = 0; count < constants::MAXLINKERRORHISTOGRAMS; ++count) {
      TH2F* h = (TH2F*)mLinkErrors->At(count);
      h->GetXaxis()->SetTitle("Sector*2 + side");
      h->GetXaxis()->CenterTitle(kTRUE);
      h->GetYaxis()->SetTitle("Stack_Layer");
      h->GetYaxis()->CenterTitle(kTRUE);
      for (int s = 0; s < o2::trd::constants::NSTACK; ++s) {
        for (int l = 0; l < o2::trd::constants::NLAYER; ++l) {
          std::string label = fmt::format("{0}_{1}", s, l);
          int pos = s * o2::trd::constants::NLAYER + l + 1;
          h->GetYaxis()->SetBinLabel(pos, label.c_str());
        }
      }
    }
    mReader.setHistos(mLinkErrors, mParseErrors);
    //mReader.setParsingHistos();
    mReader.setTimeHistos(mTimeFrameTime, mTrackletParsingTime,
                          mDigitParsingTime, mCruTime, mPackagingTime,
                          mDataVersions, mDataVersionsMajor, mParsingErrors);
  }
}
void DataReaderTask::init(InitContext& ic)
{
  LOG(info) << "o2::trd::DataReadTask init";

  auto finishFunction = [this]() {
    mReader.checkSummary();
  };
  mReader.setMaxErrWarnPrinted(ic.options().get<int>("log-max-errors"), ic.options().get<int>("log-max-warnings"));
  buildHistograms(); // if requested create all the histograms
  ic.services().get<CallbackService>().set(CallbackService::Id::Stop, finishFunction);
}

void DataReaderTask::endOfStream(o2::framework::EndOfStreamContext& ec)
{
  if (mRootOutput) {
    mTimeFrameTime->Draw();
    mTrackletParsingTime->Draw();
    mDigitParsingTime->Draw();
    mCruTime->Draw();
    mPackagingTime->Draw();
    mDataVersions->Draw();
    mDataVersionsMajor->Draw();
    mParsingErrors->Draw();
    for (int count = 0; count < constants::MAXPARSEERRORHISTOGRAMS; ++count) {
      TH2F* h = (TH2F*)mParseErrors->At(count);
      h->Draw();
    }
    for (int count = 0; count < constants::MAXLINKERRORHISTOGRAMS; ++count) {
      TH2F* h = (TH2F*)mLinkErrors->At(count);
      h->Draw();
    }
    for (int count = 0; count < constants::MAXPARSEERRORHISTOGRAMS; ++count) {
      TH2F* h = (TH2F*)mParseErrors->At(count);
      h->Write();
    }
    for (int count = 0; count < constants::MAXLINKERRORHISTOGRAMS; ++count) {
      TH2F* h = (TH2F*)mLinkErrors->At(count);
      h->Write();
    }
    mTimeFrameTime->Write();
    mTrackletParsingTime->Write();
    mDigitParsingTime->Write();
    mCruTime->Write();
    mPackagingTime->Write();
    mDataVersions->Write();
    mDataVersionsMajor->Write();
    mParsingErrors->Write();

    mRootFile->Close();
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
  auto dataReadStart = std::chrono::high_resolution_clock::now();

  if (isTimeFrameEmpty(pc)) {
    sendData(pc, true); //send the empty tf data.
    return;
  }
  uint64_t total1 = 0, total2 = 0;

  std::vector<InputSpec> sel{InputSpec{"filter", ConcreteDataTypeMatcher{"TRD", "RAWDATA"}}};
  uint64_t tfCount = 0;
  for (auto& ref : InputRecordWalker(pc.inputs(), sel)) {
    auto inputprocessingstart = std::chrono::high_resolution_clock::now(); // measure total processing time
    const auto* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    tfCount = dh->tfCounter;
    const char* payloadIn = ref.payload;
    auto payloadInSize = DataRefUtils::getPayloadSize(ref);
    if (mHeaderVerbose) {
      LOGP(info, "Found input [{}/{}/{:#x}] TF#{} 1st_orbit:{} Payload {} : ",
           dh->dataOrigin.str, dh->dataDescription.str, dh->subSpecification, dh->tfCounter, dh->firstTForbit, payloadInSize);
    }
    if (!mCompressedData) { //we have raw data coming in from flp
      if (mVerbose) {
        LOG(info) << " parsing non compressed data in the data reader task with a payload of " << payloadInSize << " payload size";
      }
      //          LOG(info) << "start of data is at ref.payload=0x"<< std::hex << " payloadSize:0x" << payloadInSize <<" dh->headerSize:0x" <<dh->headerSize;
      total1 += payloadInSize;
      total2 += dh->headerSize;
      //          LOG(info) << "start of data is at ref.payload=0x"<< std::hex << " total1:0x" << total1 <<" total2:0x" <<total2;
      mReader.setDataBuffer(payloadIn);
      mReader.setDataBufferSize(payloadInSize);
      mReader.configure(mTrackletHCHeaderState, mHalfChamberWords, mHalfChamberMajor, mOptions);
      //mReader.setStats(&mTimeFrameStats);
      mReader.run();
      mWordsRead += mReader.getWordsRead();
      mWordsRejected += mReader.getWordsRejected();
      if (mVerbose) {
        LOG(info) << "relevant vectors to read : " << mReader.sumTrackletsFound() << " tracklets and " << mReader.sumDigitsFound() << " compressed digits";
      }
    } else { // we have compressed data coming in from flp.
      mCompressedReader.setDataBuffer(payloadIn);
      mCompressedReader.setDataBufferSize(payloadInSize);
      mCompressedReader.configure(mOptions);
      mCompressedReader.run();
    }
  }

  sendData(pc, false);
  std::chrono::duration<double, std::milli> dataReadTime = std::chrono::high_resolution_clock::now() - dataReadStart;
  if (mRootOutput) {
    mTimeFrameTime->Fill((int)std::chrono::duration_cast<std::chrono::milliseconds>(dataReadTime).count());
  }
  LOGP(info, "Digits: {}, Tracklets: {}, DataRead in: {:.3f} MB, Rejected: {:.3f} MB for TF {} in {} ms",
       mReader.getDigitsFound(), mReader.getTrackletsFound(), (float)mWordsRead * 4 / 1024.0 / 1024.0, (float)mWordsRejected * 4 / 1024.0 / 1024.0, tfCount,
       std::chrono::duration_cast<std::chrono::milliseconds>(dataReadTime).count());
}

} // namespace o2::trd
