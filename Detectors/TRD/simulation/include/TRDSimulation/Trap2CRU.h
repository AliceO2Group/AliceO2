// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//  TRD Trap2CRU class                                                       //
//  Class to take the trap output that arrives at the cru and produce        //
//  the cru output. I suppose a cru simulator                                //
///////////////////////////////////////////////////////////////////////////////

#ifndef ALICE_O2_TRD_TRAP2CRU_H
#define ALICE_O2_TRD_TRAP2CRU_H

#include <string>
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/LinkRecord.h"
#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/Digit.h"
//#include "DetectorsRaw/HBFUtils.h"
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
  //Trap2CRU(const std::string& outputDir, const std::string& inputFilename, const std::string& inputDigitsFilename, const std::string& inputTrackletsFilename);
  void readTrapData();
  void convertTrapData(o2::trd::TriggerRecord const& trigrecord, const int& triggercount);
  // default for now will be file per half cru as per the files Guido did for us.
  void setFilePer(std::string fileper) { mFilePer = fileper; };
  std::string getFilePer() { return mFilePer; };
  void setOutputDir(std::string outdir) { mOutputDir = outdir; };
  std::string getOutputDir() { return mOutputDir; };
  void setDigitRate(int digitrate) { mDigitRate = digitrate; };
  int getDigitRate() { return mDigitRate; };
  int getVerbosity() { return mVerbosity; }
  void setVerbosity(int verbosity) { mVerbosity = verbosity; }
  void sortDataToLinks();
  o2::raw::RawFileWriter& getWriter() { return mWriter; }
  uint32_t buildHalfCRUHeader(HalfCRUHeader& header, const uint32_t bc, const uint32_t halfcru); // build the half cru header holding the lengths of all links amongst other things.
  void linkSizePadding(uint32_t linksize, uint32_t& crudatasize, uint32_t& padding);             // pad the link data stream to align with 256 bit words.
  void openInputFiles();
  void setTrackletHCHeader(bool tracklethcheader) { mUseTrackletHCHeader = tracklethcheader; }
  bool isTrackletOnLink(int link, int trackletpos); // is the current tracklet on the the current link
  bool isDigitOnLink(int link, int digitpos);       // is the current digit on the current link
  int buildDigitRawData(const int digitstartindex, const int digitendindex, const int mcm, const int rob, const uint32_t triggercount);
  int buildTrackletRawData(const int trackletindex, const int linkid); // from the current position in the tracklet vector, build the outgoing data for the current mcm the tracklet is on.
  int writeDigitEndMarker();                                           // write the digit end marker 0x0 0x0
  int writeTrackletEndMarker();                                        // write the tracklet end maker 0x10001000 0x10001000
  int writeHCHeader(const int eventcount, uint32_t linkid);            // write the HalfChamberHeader into the stream, after the tracklet endmarker and before the digits.

  bool digitindexcompare(const o2::trd::Digit& A, const o2::trd::Digit& B);
  //boohhl digitindexcompare(const unsigned int A, const unsigned int B);
  o2::trd::Digit& getDigitAt(const int i) { return mDigits[mDigitsIndex[i]]; };

  void mergetriggerDigitRanges();

 private:
  int mfileGranularity; /// per link or per half cru for each file
  uint32_t mLinkID;     // always 15 for TRD
  uint16_t mCruID;      // built into the FeeID
  uint64_t mFeeID;      // front end id defining the cru sm:8 bits, blank 3 bits, side:1,blank 3 bits, end point:1
  uint32_t mEndPointID; // end point on the cru in question, there are 2 pci end points per cru
  std::string mFilePer; // how to split up the raw data files, sm:per supermodule, halfcru: per half cru, cru: per cru, all: singular file.
  //  std::string mInputFileName;
  std::string mOutputFileName;
  int mVerbosity{0};
  std::string mOutputDir;
  uint32_t mSuperPageSizeInB;
  int mDigitRate = 1000;
  int mEventDigitCount = 0;
  //HalfCRUHeader mHalfCRUHeader;
  //TrackletMCMHeader mTrackletMCMHeader;
  // TrackletMCMData mTrackletMCMData;
  bool mUseTrackletHCHeader{false};
  std::vector<char> mRawData; // store for building data event for a single half cru
  uint32_t mRawDataPos = 0;
  char* mRawDataPtr{nullptr};
  // locations to store the incoming data branches

  // incoming digit information
  std::string mInputDigitsFileName;
  TFile* mDigitsFile;
  TTree* mDigitsTree;
  std::vector<Digit> mDigits, *mDigitsPtr = &mDigits;
  std::vector<uint32_t> mDigitsIndex;
  std::vector<o2::trd::TriggerRecord> mDigitTriggerRecords;
  std::vector<o2::trd::TriggerRecord>* mDigitTriggerRecordsPtr = &mDigitTriggerRecords;

  //incoming tracklet information
  std::string mInputTrackletsFileName;
  TFile* mTrackletsFile;
  TTree* mTrackletsTree;
  std::vector<Tracklet64> mTracklets;
  std::vector<Tracklet64>* mTrackletsPtr = &mTracklets;
  std::vector<uint32_t> mTrackletsIndex;
  std::vector<o2::trd::TriggerRecord> mTrackletTriggerRecords;
  std::vector<o2::trd::TriggerRecord>* mTrackletTriggerRecordsPtr = &mTrackletTriggerRecords;
  uint64_t mCurrentTracklet{0}; //the tracklet we are currently busy adding
  uint64_t mCurrentDigit{0};    //the digit we are currently busy adding

  const o2::raw::HBFUtils& mSampler = o2::raw::HBFUtils::Instance();
  o2::raw::RawFileWriter mWriter{"TRD"};

  ClassDefNV(Trap2CRU, 3);
};

} // end namespace trd
} // end namespace o2
#endif
