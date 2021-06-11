// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "TRDBase/Digit.h"
//#include "DataFormatsTRD/FlpStats.h"

#include <fstream>

using namespace o2::framework;

namespace o2::trd
{

class DataReaderTask : public Task
{
 public:
  DataReaderTask(bool compresseddata, bool byteswap, bool verbose, bool headerverbose, bool dataverbose) : mCompressedData(compresseddata), mByteSwap(byteswap), mVerbose(verbose), mHeaderVerbose(headerverbose), mDataVerbose(dataverbose) {}
  ~DataReaderTask() override = default;
  void init(InitContext& ic) final;
  void sendData(ProcessingContext& pc);
  void run(ProcessingContext& pc) final;

 private:
  CruRawReader mReader;                  // this will do the parsing, of raw data passed directly through the flp(no compression)
  CompressedRawReader mCompressedReader; //this will handle the incoming compressed data from the flp
                                         // in both cases we pull the data from the vectors build message and pass on.
                                         // they will internally produce a vector of digits and a vector tracklets and associated indexing.
                                         // TODO templatise this and 2 versions of datareadertask, instantiated with the relevant parser.
  std::vector<Tracklet64> mTracklets;
  std::vector<Digit> mDigits;
  std::vector<o2::trd::TriggerRecord> mTriggers;
  //  std::vector<o2::trd::FlpStats> mStats;

  bool mVerbose{false};        // verbos output general debuggign and info output.
  bool mDataVerbose{false};    // verbose output of data unpacking
  bool mHeaderVerbose{false};  // verbose output of headers
  bool mCompressedData{false}; // are we dealing with the compressed data from the flp (send via option)
  bool mByteSwap{true};        // whether we are to byteswap the incoming data, mc is not byteswapped, raw data is (too be changed in cru at some point)
};

} // namespace o2::trd

#endif // O2_TRD_DATAREADERTASK
