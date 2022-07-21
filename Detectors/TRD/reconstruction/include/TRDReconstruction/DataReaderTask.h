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

/// @file   DataReaderTask.h
/// @author Sean Murray
/// @brief  TRD epn task to read incoming data

#ifndef O2_TRD_DATAREADERTASK
#define O2_TRD_DATAREADERTASK

#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"
#include "TRDReconstruction/CruRawReader.h"
#include "TRDReconstruction/CompressedRawReader.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/Digit.h"
#include "DataFormatsTRD/RawDataStats.h"
#include <fstream>

#include "TGraph2D.h"
#include "TH3F.h"
#include <TH2F.h>

using namespace o2::framework;

namespace o2::trd
{

class DataReaderTask : public Task
{
 public:
  DataReaderTask(int tracklethcheader, int halfchamberwords, int halfchambermajor, std::string histofilename, std::bitset<16> option) : mCompressedData(option[TRDCompressedDataBit]), mByteSwap(option[TRDByteSwapBit]), mFixDigitEndCorruption(option[TRDFixDigitCorruptionBit]), mTrackletHCHeaderState(tracklethcheader), mHalfChamberWords(halfchamberwords), mHalfChamberMajor(halfchambermajor), mHistogramsFilename(histofilename), mVerbose(option[TRDVerboseBit]), mHeaderVerbose(option[TRDHeaderVerboseBit]), mDataVerbose(option[TRDDataVerboseBit]), mEnableTimeInfo(option[TRDEnableTimeInfoBit]), mEnableStats(option[TRDEnableStatsBit]), mRootOutput(option[TRDEnableRootOutputBit]), mIgnoreTrackletHCHeader(option[TRDIgnoreTrackletHCHeaderBit]), mIgnoreDigitHCHeader(option[TRDIgnoreDigitHCHeaderBit]), mOptions(option) {}
  ~DataReaderTask() override = default;
  void init(InitContext& ic) final;
  void sendData(ProcessingContext& pc, bool blankframe = false);
  void run(ProcessingContext& pc) final;
  bool isTimeFrameEmpty(ProcessingContext& pc);
  void endOfStream(o2::framework::EndOfStreamContext& ec) override;
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final;

 private:
  void updateTimeDependentParams(framework::ProcessingContext& pc);
  CruRawReader mReader;                  // this will do the parsing, of raw data passed directly through the flp(no compression)
  CompressedRawReader mCompressedReader; //this will handle the incoming compressed data from the flp
                                         // in both cases we pull the data from the vectors build message and pass on.
                                         // they will internally produce a vector of digits and a vector tracklets and associated indexing.
                                         // TODO templatise this and 2 versions of datareadertask, instantiated with the relevant parser.

  bool mVerbose{false};          // verbos output general debuggign and info output.
  bool mDataVerbose{false};      // verbose output of data unpacking
  bool mHeaderVerbose{false};    // verbose output of headers
  bool mCompressedData{false};   // are we dealing with the compressed data from the flp (send via option)
  bool mByteSwap{true};          // whether we are to byteswap the incoming data, mc is not byteswapped, raw data is (too be changed in cru at some point)
  bool mEnableTimeInfo{false};   // enable the timing of timeframe,cru,digit,tracklet processing.
  bool mEnableStats{false};      // enable the taking of stats in the rawdatastats class
  bool mRootOutput{false};       // enable the writing of histos.root, a poor mans qc, mostly for debugging.
  bool mIgnoreDigitHCHeader{false};    // ignore this header for the purposes of data cross checking use the rdh/cru as authoritative
  bool mIgnoreTrackletHCHeader{false}; // ignore this header for data validity checks, this and the above are use to parse corrupted data.
  std::bitset<16> mOptions;            // stores the incoming of the above bools, useful to be able to send this on instead of the individual ones above
                                       // the above bools make the code more readable hence still here.

  uint64_t mWordsRead = 0;
  uint64_t mWordsRejected = 0;
  int mTrackletHCHeaderState{0}; // what to do about tracklethcheader, 0 never there, 2 always there, 1 there iff tracklet data, i.e. only there if next word is *not* endmarker 10001000.
  int mHalfChamberWords{0};      // if the halfchamber header is effectively blanked major.minor = 0.0 and halfchamberwords=0 then this value is used as the number of additional words to try recover the data
  int mHalfChamberMajor{0};      // if the halfchamber header is effectively blanked major.minor = 0.0 and halfchamberwords=0 then this value is used as the major version to try recover the data
  std::string mHistogramsFilename; // filename to use for histograms.
  o2::header::DataDescription mUserDataDescription = o2::header::gDataDescriptionInvalid; // alternative user-provided description to pick
  bool mFixDigitEndCorruption{false};                                                     // fix the parsing of corrupt end of digit data. bounce over it.
  o2::trd::TRDDataCountersPerTimeFrame mTimeFrameStats;                                   // TODO for compressed data this is going to come in for each subtimeframe and we need to collate them.
  uint64_t mDigitPreviousTotal;                                                           // store the previous timeframes totals for tracklets and digits, to be able to get a diferential total
  uint64_t mTrackletsPreviousTotal;
};

} // namespace o2::trd

#endif // O2_TRD_DATAREADERTASK
