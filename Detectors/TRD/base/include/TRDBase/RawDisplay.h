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

#ifndef ALICEO2_TRD_RAWDISPLAY_H_
#define ALICEO2_TRD_RAWDISPLAY_H_

///
/// \file   RawDisplay.h
/// \author Thomas Dietel, tom@dietel.net
///


#include "DataFormatsTRD/Digit.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/Hit.h"
// #include "DataFormatsTRD/HelperMethods.h"


#include "TRDBase/Geometry.h"

// #include "DetectorsBase/GeometryManager.h"
// #include "DetectorsBase/Propagator.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
// #include <DataFormatsTPC/TrackTPC.h>
#include "CommonDataFormat/TFIDInfo.h"

#include <TTreeReaderArray.h>


#include <vector>
#include <array>
#include <map>
#include <ostream>
#include <filesystem>

class TFile;
class TTreeReader;
class TPad;

// template<typename T>
// class TTreeReaderArray<T>;

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

// float PadPositionMCM(o2::trd::Tracklet64 &tracklet);
// int getMCMCol(int irob, int imcm);
float PadColF(o2::trd::Tracklet64 &tracklet);
// float UncalibratedPad(o2::trd::Tracklet64 &tracklet);
float SlopeF(o2::trd::Tracklet64 &trkl);


/// A position in spatial (x,y,z) and raw/digit coordinates (det,row,col,tb).
class ChamberSpacePoint
{
 public:
  ChamberSpacePoint(int det = -999) : mDetector(det){};
  // ChamberSpacePoint(o2::track::TrackParCov& t);

  /// check if the space point has been initialized
  bool isValid() const { return mDetector >= 0; }

  /// spatial x coordinate of space point
  float getX() const { return mX; }

  /// spatial y coordinate of space point
  float getY() const { return mY; }

  /// spatial z coordinate of space point
  float getZ() const { return mZ; }

  /// detector number corresponding to space point
  int getDetector() const { return mDetector; }

  /// pad row within detector of space point 
  int getPadRow() const { return mPadrow; }

  /// pad position (a.k.a. column) within pad row
  float getPadCol() const { return mPadcol; }

  /// time coordinate in drift direction
  float getTimeBin() const { return mTimebin; }

  /// calculate MCM corresponding to pad row/column
  // int getMCM() const { return o2::trd::HelperMethods::getMCMfromPad(mPadrow, mPadcol); }

  /// calculate readout board corresponding to pad row/column
  // int getROB() const { return o2::trd::HelperMethods::getROBfromPad(mPadrow, mPadcol); }

 protected:
  float mX, mY, mZ;
  int mDetector;
  int mPadrow;
  float mPadcol, mTimebin;

  static constexpr float xscale = 1.0 / (o2::trd::Geometry::cheight() + o2::trd::Geometry::cspace());
  static constexpr float xoffset = o2::trd::Geometry::getTime0(0);
  static constexpr float alphascale = 1.0 / o2::trd::Geometry::getAlpha();
};

std::ostream& operator<<(std::ostream& os, const ChamberSpacePoint& p);


/// range of entries in another container (similar to boost::range)
template <typename value_t, typename container_t>
struct myrange {
  typedef typename container_t::iterator iterator;

  iterator &begin() { return b; }
  iterator &end() { return e; }
  iterator b, e;

  size_t length() { return e - b; }
};


class CoordinateTransformer
{
public:
  static CoordinateTransformer* instance()
  {
    static CoordinateTransformer mCTrans;
    return &mCTrans;
  }

  std::array<float, 3> Local2RCT(int det, float x, float y, float z);
  std::array<float, 3> Local2RCT(o2::trd::Hit& hit) 
  { return Local2RCT(hit.GetDetectorID(), hit.getLocalT(), hit.getLocalC(), hit.getLocalR()); }

  float GetVdrift() { return mVdrift; }
  void SetVdrift(float x) { mVdrift = x; }

  float GetT0() { return mT0; }
  void SetT0(float x) { mT0 = x; }

protected:
  o2::trd::Geometry* mGeo;
  float mVdrift{1.5625};
  float mT0{4.0};

private:
  CoordinateTransformer();

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


struct RawEvent : public RawDataSpan
{
  // std::vector<o2::tpc::TrackTPC> tpctracks;
  std::vector<o2::dataformats::TrackTPCITS> tracks;
  std::vector<ChamberSpacePoint> evtrackpoints;
};

TPad *DrawMCM(RawDataSpan &mcm, TPad *pad=NULL);



// template<typename classifier>
// class RawDataPartitioner : public map<uint32_t, RawDataSpan>
// {
// public:

//   RawDataPartitioner(RawDataSpan event)
//   {
//     // sort digits and tracklets
//     event.digits.sort(order_digit);
//     event.tracklets.sort(order_tracklet);
//     event.trackpoints.sort(order_spacepoint);

//     // add all the digits to a map that contains all the 
//     for (auto cur = event.digits.b; cur != event.digits.e; /* noop */ ) {
//       // let's look for pairs where the comparision function is true, i.e.
//       // where two adjacent elements do not have equal keys
//       auto nxt = std::adjacent_find(cur, event.digits.e, classifier::comp_digits);
//       // if adjacent_find found a match, it returns the first element of the
//       // adjacent pair, but we need the second one -> increment the result
//       if (nxt != event.digits.e)
//       {
//         ++nxt;
//       }
//       // finally, we add the found elements to the map with the current range
//       (*this)[classifier::key(*cur)].digits.b = cur; 
//       (*this)[classifier::key(*cur)].digits.e = nxt;
//       cur = nxt;
//     }

//     for (auto cur = event.tracklets.b; cur != event.tracklets.e; /* noop */) {
//       auto nxt = std::adjacent_find(cur, event.tracklets.e, classifier::comp_tracklets);
//       if (nxt != event.tracklets.e) {
//         ++nxt;
//       }
//       (*this)[classifier::key(*cur)].tracklets.b = cur; 
//       (*this)[classifier::key(*cur)].tracklets.e = nxt;
//       cur = nxt;
//     }

//     for (auto cur = event.trackpoints.b; cur != event.trackpoints.e; /* noop */) {
//       auto nxt = std::adjacent_find(cur, event.trackpoints.e, classifier::comp_trackpoints);
//       if (nxt != event.trackpoints.e) { ++nxt; }
//       (*this)[classifier::key(*cur)].trackpoints.b = cur; 
//       (*this)[classifier::key(*cur)].trackpoints.e = nxt;
//       cur = nxt;
//     }
//     cout << "Found " << this->size() << " spans" << endl;
//   }
// };

// // ========================================================================
// // ========================================================================
// //
// // Track extrapolation
// //
// // ========================================================================
// // ========================================================================

// class TrackExtrapolator
// {
//  public:
//   TrackExtrapolator();

//   ChamberSpacePoint extrapolate(o2::track::TrackParCov& t, int layer);

//  private:
//   bool adjustSector(o2::track::TrackParCov& t);
//   int getSector(float alpha);

//   o2::trd::Geometry* mGeo;
//   o2::base::Propagator* mProp;
// };

// TrackExtrapolator::TrackExtrapolator()
//   // : mGeo(o2::trd::Geometry::instance())
//   // : mProp(o2::base::Propagator::Instance())
// {
//   cout << "TrackExtrapolator: load geometry" << endl;
//   o2::base::GeometryManager::loadGeometry();
//   cout << "TrackExtrapolator: load B-field" << endl;
//   o2::base::Propagator::initFieldFromGRP();

//   cout << "TrackExtrapolator: instantiate geometry" << endl;

//   mGeo = o2::trd::Geometry::instance();
//   mGeo->createPadPlaneArray();
//   mGeo->createClusterMatrixArray();

//   mProp = o2::base::Propagator::Instance();
// }

// bool TrackExtrapolator::adjustSector(o2::track::TrackParCov& t)
// {
//   float alpha = mGeo->getAlpha();
//   float xTmp = t.getX();
//   float y = t.getY();
//   float yMax = t.getX() * TMath::Tan(0.5f * alpha);
//   float alphaCurr = t.getAlpha();
//   if (fabs(y) > 2 * yMax) {
//     printf("Skipping track crossing two sector boundaries\n");
//     return false;
//   }
//   int nTries = 0;
//   while (fabs(y) > yMax) {
//     if (nTries >= 2) {
//       printf("Skipping track after too many tries of rotation\n");
//       return false;
//     }
//     int sign = (y > 0) ? 1 : -1;
//     float alphaNew = alphaCurr + alpha * sign;
//     if (alphaNew > TMath::Pi()) {
//       alphaNew -= 2 * TMath::Pi();
//     } else if (alphaNew < -TMath::Pi()) {
//       alphaNew += 2 * TMath::Pi();
//     }
//     if (!t.rotate(alphaNew)) {
//       return false;
//     }
//     if (!mProp->PropagateToXBxByBz(t, xTmp)) {
//       return false;
//     }
//     y = t.getY();
//     ++nTries;
//   }
//   return true;
// }

// int TrackExtrapolator::getSector(float alpha)
// {
//   if (alpha < 0) {
//     alpha += 2.f * TMath::Pi();
//   } else if (alpha >= 2.f * TMath::Pi()) {
//     alpha -= 2.f * TMath::Pi();
//   }
//   return (int)(alpha * NSECTOR / (2.f * TMath::Pi()));
// }


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
  RawEvent GetEvent();
  float GetTriggerTime();

  size_t GetTimeFrameNumber() { return mTimeFrameNo; }
  size_t GetEventNumber() { return mEventNo; }

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

#endif // ALICEO2_TRD_RAWDISPLAY_H_
