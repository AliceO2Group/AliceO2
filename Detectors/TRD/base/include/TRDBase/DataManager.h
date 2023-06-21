// Copyright 2019-2023 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TRD_DATAMANAGER_H_
#define ALICEO2_TRD_DATAMANAGER_H_

///
/// \file   DataManager.h
/// \author Thomas Dietel, tom@dietel.net
///


#include "DataFormatsTRD/Digit.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/Hit.h"

#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "CommonDataFormat/TFIDInfo.h"

#include <TTreeReaderArray.h>


#include <vector>
// #include <array>
#include <map>
// #include <ostream>
// #include <filesystem>

class TFile;
class TTreeReader;


namespace o2::trd
{

/// \namespace rawdisp
/// \brief Raw data display and analysis
///
/// This namespace provides helper classes to display low-level TRD data.
///
/// origin: TRD
/// \author Thomas Dietel, tom@dietel.net
namespace rawdisp
{


/// A struct that can be used to calculate unique identifiers per pad row, to
/// be used to split ranges by pad row.
struct PadRowID
{
  template<typename T>
  static uint32_t key(const T &x) 
  { 
    return 100*x.getDetector() + x.getPadRow(); 
  }
};

/// A struct that can be used to calculate unique identifiers for MCMs, to be
/// used to split ranges by MCM.
struct MCM_ID
{
  template<typename T>
  static uint32_t key(const T &x) 
  { 
    return 1000*x.getDetector() + 8*x.getPadRow() + 4*(x.getROB()%2) + x.getMCM()%4;
  }

  static uint32_t key(const T &x) 
  { 
    return 1000*x.getDetector() + 8*x.getPadRow() + 4*(x.getROB()%2) + x.getMCM()%4;
  }

  static int getDetector(uint32_t k) {return k/1000;}
  // static int getPadRow(key) {return (key%1000) / 8;}
  static int getMcmRowCol(uint32_t k) {return k%1000;}
  
};

/// range of entries in another container (similar to boost::range)
template <typename value_t, typename container_t>
struct myrange {
  typedef typename container_t::iterator iterator;

  iterator &begin() { return b; }
  iterator &end() { return e; }
  iterator b, e;

  size_t length() { return e - b; }
};


struct RawDataSpan
{
public:
  myrange<o2::trd::Digit, TTreeReaderArray<o2::trd::Digit>> digits;
  myrange<o2::trd::Tracklet64, TTreeReaderArray<o2::trd::Tracklet64>> tracklets;
  myrange<o2::trd::Hit, TTreeReaderArray<o2::trd::Hit>> hits;
  // myrange<ChamberSpacePoint, std::vector<ChamberSpacePoint>> trackpoints;

  // sort digits, tracklets and space points by detector, pad row, column
  void sort();

  std::map<uint32_t, RawDataSpan> ByMCM();

  template<typename keyfunc>
  std::map<uint32_t,RawDataSpan> IterateBy();

//   pair<int, int> getMaxADCsumAndChannel();
//   int getMaxADCsum(){ return getMaxADCsumAndChannel().first; }

//   int getDetector() { if (mDetector == -1) calculateCoordinates(); return mDetector; } 
//   int getPadRow() { if (mPadRow == -1) calculateCoordinates(); return mPadRow; } 
  
// protected:
//   // The following variables cache the calculations of raw coordinates:
//   //   non-negative values indicate the actual position
//   //   -1 indicates that the values has not been calculated yet
//   //   -2 indicates that the value is not unique in this span 
//   int mDetector{-1};
//   int mPadRow{-1};

//   void calculateCoordinates();

};

// struct RawEvent : public RawDataSpan
// {
//   // std::vector<o2::tpc::TrackTPC> tpctracks;
//   std::vector<o2::dataformats::TrackTPCITS> tracks;
//   std::vector<ChamberSpacePoint> evtrackpoints;
// };



/// access TRD low-level data
class DataManager
{

public:
  DataManager(std::filesystem::path dir = ".");
  // DataManager(std::string dir = "./");

  void SetMatchWindowTPC(float min, float max)
  { mMatchTimeMinTPC=min; mMatchTimeMaxTPC=max; }

  bool NextTimeFrame();
  bool NextEvent();

  /// access time frame info
  o2::dataformats::TFIDInfo GetTimeFrameInfo();

  // TTreeReaderArray<o2::tpc::TrackTPC> *GetTimeFrameTPCTracks() {return mTpcTracks; }
  TTreeReaderArray<o2::dataformats::TrackTPCITS> *GetTimeFrameTracks() { return mTracks; }

  // access event info
  RawDataSpan GetEvent();
  float GetTriggerTime();

  size_t GetTimeFrameNumber() { return mTimeFrameNo; }
  size_t GetEventNumber() { return mEventNo; }

  std::string DescribeFiles();
  std::string DescribeTimeFrame();
  std::string DescribeEvent();

private: 
  TFile *mMainfile{0};
  TTree* mDatatree{0}; // tree and friends from digits, tracklets files
  TTreeReader *mDatareader{0};

  TTreeReaderArray<o2::trd::Digit>* mDigits{0};
  TTreeReaderArray<o2::trd::Tracklet64>* mTracklets{0};
  TTreeReaderArray<o2::trd::TriggerRecord>* mTrgRecords{0};

  TFile *mHitsFile{0};
  TTree* mHitsTree{0};  
  TTreeReader* mHitsReader{0};  
  TTreeReaderArray<o2::trd::Hit>* mHits{0};

  TTreeReaderArray<o2::dataformats::TrackTPCITS> *mTracks{0};
  // TTreeReaderArray<o2::tpc::TrackTPC> *mTpcTracks{0};

  o2::trd::TriggerRecord mTriggerRecord;

  std::vector<o2::dataformats::TFIDInfo> *mTFIDs{0};

  size_t mTimeFrameNo{0}, mEventNo{0};
  float mMatchTimeMinTPC{-10.0}, mMatchTimeMaxTPC{20.0};

  // template <typename T>
  // TTreeReaderArray<T> *AddReaderArray(std::string file, std::string tree, std::string branch);

  template <typename T>
  void AddReaderArray(TTreeReaderArray<T> *& array, std::filesystem::path file, std::string tree, std::string branch);

  // TrackExtrapolator extra;
};



} // namespace rawdisp

} // namespace o2::trd

#endif // ALICEO2_TRD_DATAMANAGER_H_
