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
//#include "DetectorsRaw/HBFUtils.h"
#include "DetectorsRaw/RawFileWriter.h"

namespace o2
{
namespace trd
{

class Trap2CRU
{
  static constexpr int NumberOfCRU = 36;
  static constexpr int NumberOfHalfCRU = 72;
  static constexpr int NumberOfFLP = 12;
  static constexpr int CRUperFLP = 3;
  static constexpr int WordSizeInBytes = 256; // word size in bits, everything is done in 256 bit blocks.
  static constexpr int WordSize = 8;          // 8 standard 32bit words.
  static constexpr int NLinksPerHalfCRU = 15;
  static constexpr uint32_t PaddWord = 0xeeee; // pad word to fill 256bit blocks or entire block for no data case.
                                               //TODO come back and change the mapping of 1077 channels to a lut and probably configurable.
                                               //
 public:
  Trap2CRU() = default;
  Trap2CRU(const std::string& outputDir, const std::string& inputFilename);
  void readTrapData(const std::string& otuputDir, const std::string& inputFilename, int superPageSizeInB);
  void convertTrapData(o2::trd::TriggerRecord const& TrigRecord);
  // default for now will be file per half cru as per the files Guido did for us.
  // TODO come back and give users a choice.
  //       void setFilePerLink(){mfileGranularity = mgkFilesPerLink;};
  //       bool getFilePerLink(){return (mfileGranularity==mgkFilesPerLink);};
  //       void setFilePerHalfCRU(){mfileGranularity = mgkFilesPerHalfCRU;};
  //       bool getFilePerHalfCRU(){return (mfileGranularity==mgkFilesPerHalfCRU);};  //
  int getVerbosity() { return mVerbosity; }
  void setVerbosity(int verbosity) { mVerbosity = verbosity; }
  void buildCRUPayLoad();
  o2::raw::RawFileWriter& getWriter() { return mWriter; }
  uint32_t buildCRUHeader(HalfCRUHeader& header, uint32_t bc, uint32_t halfcru, int startlinkrecord);
  void linkSizePadding(uint32_t linksize, uint32_t& crudatasize, uint32_t& padding);

 private:
  int mfileGranularity; /// per link or per half cru for each file
  uint32_t mLinkID;
  uint16_t mCruID;
  uint64_t mFeeID;
  uint32_t mEndPointID;
  std::string mInputFilename;
  std::string mOutputFilename;
  int mVerbosity = 0;
  HalfCRUHeader mHalfCRUHeader;
  TrackletMCMHeader mTrackletMCMHeader;
  TrackletMCMData mTrackletMCMData;

  std::vector<char> mRawData; // store for building data event for a single half cru
  uint32_t mRawDataPos = 0;
  TFile* mTrapRawFile;
  TTree* mTrapRawTree; // incoming data tree
  // locations to store the incoming data branches
  std::vector<o2::trd::LinkRecord> mLinkRecords, *mLinkRecordsPtr = &mLinkRecords;
  std::vector<o2::trd::TriggerRecord> mTriggerRecords, *mTriggerRecordsPtr = &mTriggerRecords;
  std::vector<uint32_t> mTrapRawData, *mTrapRawDataPtr = &mTrapRawData;

  const o2::raw::HBFUtils& mSampler = o2::raw::HBFUtils::Instance();
  o2::raw::RawFileWriter mWriter{"TRD"};

  ClassDefNV(Trap2CRU, 1);
};

} // end namespace trd
} // end namespace o2
#endif
