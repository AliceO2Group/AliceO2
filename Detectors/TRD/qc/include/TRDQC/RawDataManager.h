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

#ifndef ALICEO2_TRD_RawDataManager_H_
#define ALICEO2_TRD_RawDataManager_H_

///
/// \file   RawDataManager.h
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
#include <filesystem>

class TFile;
class TTreeReader;

namespace o2::trd
{


/// PadRowID is a struct to calculate unique identifiers per pad row.
/// The struct can be passed as a template parameter to the RawDataSpan::IterateBy
/// method to split the data span by pad row and iterate over the pad rows.
struct PadRowID
{
  /// The static `key` method calculates a padrow ID for digits and tracklets
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

/// RawDataSpan holds ranges pointing to parts of other containers
/// This class helps with restricting the view of digits/trackets/... to part
/// of a timeframe, e.g. data from one event, detector, padrow or MCM.
/// It also provides methods to help with iterating over the data
/// The main data members are public, and can be accessed without setters/getters.
struct RawDataSpan
{
public:
  myrange<o2::trd::Digit, TTreeReaderArray<o2::trd::Digit>> digits;
  myrange<o2::trd::Tracklet64, TTreeReaderArray<o2::trd::Tracklet64>> tracklets;
  myrange<o2::trd::Hit, TTreeReaderArray<o2::trd::Hit>> hits;
  // myrange<ChamberSpacePoint, std::vector<ChamberSpacePoint>> trackpoints;

  /// Sort digits, tracklets and space points by detector, pad row, column
  /// The digits, tracklets, hits and other future data members must be sorted
  /// for the IterateBy method.  
  void sort();

  /// Return a vector with one data span for each MCM that has digits, tracklets or both 
  /// IterateBy provides a more flexible interface, and should replace this method.
  std::vector<RawDataSpan> ByMCM();

  /// Return a vector with data spans, split according to the keyfunc struct
  /// The keyfunc struct must have a method `key` to calculate a key for tracklets
  /// and digits that uniquely identifies the span this digit or tracklet will be 
  /// part of. The key must be monotonically increasing for the digits and tracklets
  // stored in the raw data span.
  template<typename keyfunc>
  std::vector<RawDataSpan> IterateBy();

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


/// RawDataManager: read raw, MC and reconstruced data files and loop over them
///
/// The input for the RawDataManager are directories with raw and, optionally,
/// reconstructed and Monte-Carlo files. It scans for available files, reads
/// them and provides access functions to loop over time frames and events.
/// Supported data files:
///   - trdtracklets.root (required): TRD tracklets and trigger records
///   - trddigits.root: TRD digits - might only be available for some events
///   - o2sim_HitsTRD.root: TRD MC hits with global and TRD chamber coordinates
///   - o2_tfidinfo.root (reconstructed data only): time frame information
///
/// The following file types had some support in previous private branches,
/// but need further cleanup before integration into O2:
///   -  o2match_itstpc.root: ITS-TPC tracks
///   -  tpctracks.root: TPC-only tracks
class RawDataManager
{

public:
  RawDataManager(std::filesystem::path dir = ".");
  // RawDataManager(std::string dir = "./");

  // void SetMatchWindowTPC(float min, float max)
  // { mMatchTimeMinTPC=min; mMatchTimeMaxTPC=max; }

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
  // access to TRD digits and tracklets
  TFile *mMainfile{0}; // the main trdtracklets.root file
  TTree* mDatatree{0}; // tree and friends from digits, tracklets files
  TTreeReader *mDatareader{0};

  TTreeReaderArray<o2::trd::Digit>* mDigits{0};
  TTreeReaderArray<o2::trd::Tracklet64>* mTracklets{0};
  TTreeReaderArray<o2::trd::TriggerRecord>* mTrgRecords{0};

  // access to Monte-Carlo hits
  TFile *mHitsFile{0};
  TTree* mHitsTree{0};  
  TTreeReader* mHitsReader{0};  
  TTreeReaderArray<o2::trd::Hit>* mHits{0};

  // access tracks
  TTreeReaderArray<o2::dataformats::TrackTPCITS> *mTracks{0};
  // TTreeReaderArray<o2::tpc::TrackTPC> *mTpcTracks{0};

  // time frame information (for data only)
  std::vector<o2::dataformats::TFIDInfo> *mTFIDs{0};

  // current trigger record
  o2::trd::TriggerRecord mTriggerRecord;

  // time frame and event counters 
  size_t mTimeFrameNo{0}, mEventNo{0};

  // matching parameters for tracks
  // float mMatchTimeMinTPC{-10.0}, mMatchTimeMaxTPC{20.0};

  template <typename T>
  void AddReaderArray(TTreeReaderArray<T> *& array, std::filesystem::path file, std::string tree, std::string branch);

  // TrackExtrapolator extra;
};



} // namespace o2::trd

#endif // ALICEO2_TRD_RawDataManager_H_
