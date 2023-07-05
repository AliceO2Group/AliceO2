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
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/Digit.h"
#include "DataFormatsTRD/RawDataStats.h"
#include <fstream>

using namespace o2::framework;

namespace o2::trd
{

class DataReaderTask : public Task
{
 public:
  DataReaderTask(int tracklethcheader, int halfchamberwords, int halfchambermajor, std::bitset<16> option) : mTrackletHCHeaderState(tracklethcheader), mHalfChamberWords(halfchamberwords), mHalfChamberMajor(halfchambermajor), mOptions(option) {}
  ~DataReaderTask() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  bool isTimeFrameEmpty(ProcessingContext& pc);
  void endOfStream(o2::framework::EndOfStreamContext& ec) override;
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final;

 private:
  void updateTimeDependentParams(framework::ProcessingContext& pc);
  CruRawReader mReader; // this will do the parsing, of raw data passed directly through the flp(no compression)
                        // we pull the data from the vectors build message and pass on.
                        // they will internally produce a vector of digits and a vector tracklets and associated indexing.

  bool mVerbose{false};          // verbos output general debuggign and info output.
  bool mDataVerbose{false};      // verbose output of data unpacking
  bool mHeaderVerbose{false};    // verbose output of headers
  bool mCompressedData{false};   // are we dealing with the compressed data from the flp (send via option)
  int mProcessEveryNthTF{1};     // to parse only every n-th TF and send empty output for the rest
  bool mInitOnceDone{false};     // flag for requesting new CCDB object upon global run number change
  std::bitset<16> mOptions;            // stores the incoming of the above bools, useful to be able to send this on instead of the individual ones above
                                       // the above bools make the code more readable hence still here.

  int mTrackletHCHeaderState{0}; // what to do about tracklethcheader, 0 never there, 2 always there, 1 there iff tracklet data, i.e. only there if next word is *not* endmarker 10001000.
  int mHalfChamberWords{0};      // if the halfchamber header is effectively blanked major.minor = 0.0 and halfchamberwords=0 then this value is used as the number of additional words to try recover the data
  int mHalfChamberMajor{0};      // if the halfchamber header is effectively blanked major.minor = 0.0 and halfchamberwords=0 then this value is used as the major version to try recover the data
  o2::header::DataDescription mUserDataDescription = o2::header::gDataDescriptionInvalid; // alternative user-provided description to pick
  bool mFixDigitEndCorruption{false};                                                     // fix the parsing of corrupt end of digit data. bounce over it.
  uint64_t mDatasizeInTotal{0};                                                           // accumulate the total data size read in bytes
  uint64_t mWordsRejectedTotal{0};                                                        // accumulate the total number of words rejected
  uint64_t mDigitsTotal{0};                                                               // accumulate the total number of digits read
  uint64_t mTrackletsTotal{0};                                                            // accumulate the total number os tracklets read
  size_t mNTFsProcessed{0};                                                               // keep track of the total number of TFs processed
};

} // namespace o2::trd

#endif // O2_TRD_DATAREADERTASK
