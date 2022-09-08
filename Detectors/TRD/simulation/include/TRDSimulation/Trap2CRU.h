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

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//  TRD Trap2CRU class                                                       //
//  Convert simulated digits and tracklets into a CRU data stream            //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#ifndef ALICE_O2_TRD_TRAP2CRU_H
#define ALICE_O2_TRD_TRAP2CRU_H

#include <string>
#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/Digit.h"
#include "DataFormatsTRD/Constants.h"
#include "DetectorsRaw/RawFileWriter.h"

namespace o2
{
namespace trd
{

class Trap2CRU
{
 public:
  Trap2CRU() = default;
  Trap2CRU(const std::string& outputDir, const std::string& inputDigitsFilename, const std::string& inputTrackletsFilename);

  // entry point for processing, is called from trap2raw.cxx
  void readTrapData();

  // sort digits and tracklets by link ID
  void sortDataToLinks();

  // open digits and tracklets files for reading
  void openInputFiles();

  // main processing function, called for every TRD trigger and creates the raw data stream for each half CRU
  void convertTrapData(o2::trd::TriggerRecord const& trigrecord, const int& triggercount);

  // settings
  void setFilePer(std::string fileper) { mFilePer = fileper; };
  void setOutputDir(std::string outdir) { mOutputDir = outdir; };
  void setVerbosity(int verbosity) { mVerbosity = verbosity; }
  void setTrackletHCHeader(int tracklethcheader) { mUseTrackletHCHeader = tracklethcheader; }
  void setTimeStamp(long ts) { mTimeStamp = ts; }

  // make the writer available in trap2raw.cxx for configuration
  o2::raw::RawFileWriter& getWriter() { return mWriter; }
  // build the half cru header holding the lengths of all links amongst other things.
  uint32_t buildHalfCRUHeader(HalfCRUHeader& header, const uint32_t bc, const uint32_t halfcru, bool isCalibTrigger);

  // write digits for single MCM into raw stream (include DigitMCMHeader and ADC mask)
  int buildDigitRawData(const int digitstartindex, const int digitendindex, const int mcm, const int rob, const uint32_t triggercount);
  // write tracklets for single MCM into raw stream (includes TrackletMCMHeader)
  int buildTrackletRawData(unsigned int trackletIndexStart);
  // write two digit end markers
  void writeDigitEndMarkers();
  // write two tracklet end markers
  void writeTrackletEndMarkers();
  // write digit HC header (two headers are written)
  void writeDigitHCHeaders(const int eventcount, uint32_t hcId);
  // write tracklet HC header
  void writeTrackletHCHeader(int hcid, int eventcount);

 private:
  uint8_t mLinkID{constants::TRDLINKID}; // always 15 for TRD
  uint16_t mCruID{0};                    // built into the FeeID
  uint16_t mFeeID{0};                    // front end id defining the cru sm:8 bits, blank 3 bits, side:1,blank 3 bits, end point:1
  uint8_t mEndPointID{0};                // end point on the cru in question, there are 2 pci end points per cru

  // settings
  std::string mFilePer; // how to split up the raw data files, sm:per supermodule, halfcru: per half cru, cru: per cru, all: singular file.
  int mVerbosity{0};    // currently only 2 levels: 0 - OFF, 1 - verbose output
  std::string mOutputDir;
  int mUseTrackletHCHeader{0}; // 0 - don't write header, 1 - write header if tracklets available, 2 - always write header

  // input
  // digits
  std::string mInputDigitsFileName;
  TFile* mDigitsFile;
  TTree* mDigitsTree;
  std::vector<Digit> mDigits, *mDigitsPtr = &mDigits;
  // tracklets and trigger records
  std::string mInputTrackletsFileName;
  TFile* mTrackletsFile;
  TTree* mTrackletsTree;
  std::vector<Tracklet64> mTracklets, *mTrackletsPtr{&mTracklets};
  std::vector<o2::trd::TriggerRecord> mTrackletTriggerRecords, *mTrackletTriggerRecordsPtr{&mTrackletTriggerRecords};

  // helpers
  long mTimeStamp{0};                          // used to retrieve the correct link to HCID mapping from CCDB
  const LinkToHCIDMapping* mLinkMap = nullptr; // to retrieve HCID from Link ID
  std::vector<uint32_t> mDigitsIndex; // input digits are sorted using this index array
  char* mRawDataPtr{nullptr};         // points to the current position in the raw data where we are writing
  uint64_t mCurrentTracklet{0}; //the tracklet we are currently busy adding
  uint64_t mCurrentDigit{0};    //the digit we are currently busy adding
  uint64_t mTotalTrackletsWritten{0}; // count the total number of tracklets written to the raw data
  uint64_t mTotalDigitsWritten{0};    // count the total number of digits written to the raw data

  const o2::raw::HBFUtils& mSampler = o2::raw::HBFUtils::Instance();
  o2::raw::RawFileWriter mWriter{"TRD"};

  ClassDefNV(Trap2CRU, 4);
};

} // end namespace trd
} // end namespace o2
#endif
