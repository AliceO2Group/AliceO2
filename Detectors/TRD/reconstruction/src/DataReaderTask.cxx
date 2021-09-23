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
#include "Framework/RawDeviceService.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/InputRecordWalker.h"

#include "DataFormatsTRD/Constants.h"

#include <fairmq/FairMQDevice.h>
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
  LOG(INFO) << "o2::trd::DataReadTask init";

  auto finishFunction = [this]() {
    mReader.checkSummary();
  };
  buildHistograms(); // if requested create all the histograms
  ic.services().get<CallbackService>().set(CallbackService::Id::Stop, finishFunction);
  mDataDesc = "RAWDATA";
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
    mReader.buildDPLOutputs(pc, mDataVerbose);
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
  for (const auto& ref : o2::framework::InputRecordWalker(pc.inputs(), {dummy})) {
    const auto dh = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    if (dh->payloadSize == 0) {
      LOGP(INFO, "Found blank input input [{}/{}/{:#x}] TF#{} 1st_orbit:{} Payload {} : ",
           dh->dataOrigin.str, dh->dataDescription.str, dh->subSpecification, dh->tfCounter, dh->firstTForbit, dh->payloadSize);
      return true;
    }
  }
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
  /* set encoder output buffer */
  char bufferOut[o2::trd::constants::HBFBUFFERMAX];
  int loopcounter = 0;
  uint64_t tfcounter = 0;
  uint64_t inputcounter = 0;
  auto device = pc.services().get<o2::framework::RawDeviceService>().device();
  auto outputRoutes = pc.services().get<o2::framework::RawDeviceService>().spec().outputs;
  auto fairMQChannel = outputRoutes.at(0).channel;
  /* loop over inputs routes */
  for (auto iit = pc.inputs().begin(), iend = pc.inputs().end(); iit != iend; ++iit) {
    if (!iit.isValid()) {
      continue;
    }
    /* loop over input parts */
    int inputpartscount = 0;
    int emptyframe = 0;
    tfcounter++;
    for (auto const& ref : iit) {
      auto inputprocessingstart = std::chrono::high_resolution_clock::now(); // measure total processing time
      if (mVerbose) {
        const auto dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
        LOGP(info, "Found input [{}/{}/{:#x}] TF#{} 1st_orbit:{} Payload {} : ",
             dh->dataOrigin.str, dh->dataDescription.str, dh->subSpecification, dh->tfCounter, dh->firstTForbit, dh->payloadSize);
      }
      const auto* headerIn = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      auto payloadIn = ref.payload;
      auto payloadInSize = headerIn->payloadSize;
      //    const auto dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      if (std::string(headerIn->dataDescription.str) != std::string("DISTSUBTIMEFRAMEFLP")) {
        if (!mCompressedData) { //we have raw data coming in from flp
          if (mVerbose) {
            LOG(info) << " parsing non compressed data in the data reader task with a payload of " << payloadInSize << " payload size";
          }
          //          LOG(info) << "start of data is at ref.payload=0x"<< std::hex << " headerIn->payloadSize:0x" << headerIn->payloadSize <<" headerIn->headerSize:0x" <<headerIn->headerSize;
          total1 += headerIn->payloadSize;
          total2 += headerIn->headerSize;
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
      } // ignore the input of DISTSUBTIMEFRAMEFLP
    }

    std::chrono::duration<double, std::milli> dataReadTime = std::chrono::high_resolution_clock::now() - dataReadStart;
    LOG(info) << "Processing time for Data reading  " << std::chrono::duration_cast<std::chrono::milliseconds>(dataReadTime).count() << "ms";
    if (mRootOutput) {
      mTimeFrameTime->Fill((int)std::chrono::duration_cast<std::chrono::milliseconds>(dataReadTime).count());
    }
    LOG(info) << "[" << tfcounter << "] D:" << mReader.getDigitsFound() << " T:" << mReader.getTrackletsFound();
    /* output */
    sendData(pc, false);
  }
    if (!mCompressedData) {
      LOG(info) << "Digits found : " << mReader.getDigitsFound();
      LOG(info) << "Tracklets found : " << mReader.getTrackletsFound();
      LOG(info) << "DataRead in :" << mWordsRead * 4 << " bytes";
      LOG(info) << "DataRejected in :" << mWordsRejected * 4 << " bytes";
      LOG(info) << "DataRetention :bad/good" << (double)mWordsRejected / (double)mWordsRead << "";
      LOG(info) << "Total % good data bad/(good+bad)" << (double)mWordsRejected / ((double)mWordsRead + (double)mWordsRejected) * 100.0 << " %";
    }
}

} // namespace o2::trd
