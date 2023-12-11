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

#include "TRDQC/RawDataManager.h"

#include <RtypesCore.h>
#include <TSystem.h>
#include <TFile.h>
#include <TTree.h>
#include <boost/range/distance.hpp>
#include <boost/range/iterator_range_core.hpp>
#include <iterator>
#include "TRDQC/CoordinateTransformer.h"
#include "Framework/Logger.h"

#include <set>
#include <utility>

using namespace o2::trd;

/// comparison function to order digits by det / row / MCM / -channel
bool comp_digit(const o2::trd::Digit& a, const o2::trd::Digit& b)
{
  if (a.getDetector() != b.getDetector()) {
    return a.getDetector() < b.getDetector();
  }

  if (a.getPadRow() != b.getPadRow()) {
    return a.getPadRow() < b.getPadRow();
  }

  if (a.getROB() != b.getROB()) {
    return a.getROB() < b.getROB();
  }

  if (a.getMCM() != b.getMCM()) {
    return a.getMCM() < b.getMCM();
  }

  // sort channels in descending order, to ensure ordering of pad columns
  if (a.getChannel() != b.getChannel()) {
    return a.getChannel() > b.getChannel();
  }

  return true;
}

/// comparison function to order tracklets by det / row / MCM / channel
bool comp_tracklet(const o2::trd::Tracklet64& a, const o2::trd::Tracklet64& b)
{
  // upper bits of hcid and padrow from Tracklet64 word
  const uint64_t det_row_mask = 0x0ffde00000000000;

  // lowest bit of hcid (side), MCM col and pos from Tracklet64 word
  const uint64_t col_pos_mask = 0x00011fff00000000;

  auto a_det_row = a.getTrackletWord() & det_row_mask;
  auto b_det_row = b.getTrackletWord() & det_row_mask;

  if (a_det_row != b_det_row) {
    return a_det_row < b_det_row;
  }

  auto a_col_pos = a.getTrackletWord() & col_pos_mask;
  auto b_col_pos = b.getTrackletWord() & col_pos_mask;

  return a_col_pos < b_col_pos;
};

bool comp_spacepoint(const ChamberSpacePoint& a, const ChamberSpacePoint& b)
{
  if (a.getDetector() != b.getDetector()) {
    return a.getDetector() < b.getDetector();
  }

  if (a.getPadRow() != b.getPadRow()) {
    return a.getPadRow() < b.getPadRow();
  }

  if (a.getPadCol() != b.getPadCol()) {
    return a.getPadCol() < b.getPadCol();
  }

  return true;
}

void RawDataSpan::sort()
{
  std::stable_sort(std::begin(digits), std::end(digits), comp_digit);
  std::stable_sort(std::begin(tracklets), std::end(tracklets), comp_tracklet);
  std::stable_sort(std::begin(hits), std::end(hits), comp_spacepoint);
}

template <typename keyfunc>
std::vector<RawDataSpan> RawDataSpan::iterateBy()
{
  // an map for keeping track which ranges correspond to which key
  std::map<uint32_t, RawDataSpan> spanmap;

  // sort digits and tracklets
  sort();

  // add all the digits to a map
  for (auto cur = digits.begin(); cur != digits.end(); /* noop */) {
    // calculate the key of the current (first unprocessed) digit
    auto key = keyfunc::key(*cur);
    // find the first digit with a different key
    auto nxt = std::find_if(cur, digits.end(), [key](auto x) { return keyfunc::key(x) != key; });
    // store the range cur:nxt in the map
    spanmap[key].digits = boost::make_iterator_range(cur, nxt);
    // continue after this range
    cur = nxt;
  }

  // add tracklets to the map
  for (auto cur = tracklets.begin(); cur != tracklets.end(); /* noop */) {
    auto key = keyfunc::key(*cur);
    auto nxt = std::find_if(cur, tracklets.end(), [key](auto x) { return keyfunc::key(x) != key; });
    spanmap[key].tracklets = boost::make_iterator_range(cur, nxt);
    cur = nxt;
  }

  // spanmap contains all TRD data - either digits or tracklets. Now we insert hit information into these spans. The
  // tricky part is that space points or hits can belong to more than one MCM, i.e. they could appear in two spans.
  // We keep the begin iterator for each key in a map
  std::map<uint32_t, std::vector<HitPoint>::iterator> firsthit;
  for (auto cur = hits.begin(); cur != hits.end(); ++cur) {
    // calculate the keys for this hit
    auto keys = keyfunc::keys(*cur);
    // if we are not yet aware of this key, register the current hit as the first hit
    for (auto key : keys) {
      firsthit.insert({key, cur});
    }
    // remote the keys from the firsthit map that are no longer found in the hits
    for (auto it = firsthit.cbegin(); it != firsthit.cend(); /* no increment */) {
      if (keys.find(it->first) == keys.end()) {
        spanmap[it->first].hits = boost::make_iterator_range(it->second, cur);
        it = firsthit.erase(it);
      } else {
        ++it;
      }
    }
  }

  // convert the map of spans into a vector, as we do not need the access by key
  // and longer, and having a vector makes the looping by the user easier.
  std::vector<RawDataSpan> spans;
  transform(spanmap.begin(), spanmap.end(), back_inserter(spans), [](auto const& pair) { return pair.second; });

  return spans;
}

/// PadRowID is a struct to calculate unique identifiers per pad row.
/// The struct can be passed as a template parameter to the RawDataSpan::IterateBy
/// method to split the data span by pad row and iterate over the pad rows.
struct PadRowID {
  /// The static `key` method calculates a padrow ID for digits and tracklets
  template <typename T>
  static uint32_t key(const T& x)
  {
    return 100 * x.getDetector() + x.getPadRow();
  }

  static std::set<uint32_t> keys(const o2::trd::ChamberSpacePoint& x)
  {
    uint32_t key = 100 * x.getDetector() + x.getPadRow();
    return {key};
  }

  static bool match(const uint32_t key, const o2::trd::ChamberSpacePoint& x)
  {
    return key == 100 * x.getDetector() + x.getPadRow();
  }
};

// instantiate the template to iterate by padrow
template std::vector<RawDataSpan> RawDataSpan::iterateBy<PadRowID>();

// non-template wrapper function to keep PadRowID within the .cxx file
std::vector<RawDataSpan> RawDataSpan::iterateByPadRow() { return iterateBy<PadRowID>(); }

/// A struct that can be used to calculate unique identifiers for MCMs, to be
/// used to split ranges by MCM.
struct MCM_ID {
  template <typename T>
  static uint32_t key(const T& x)
  {
    return 1000 * x.getDetector() + 8 * x.getPadRow() + 4 * (x.getROB() % 2) + x.getMCM() % 4;
  }

  static std::set<uint32_t> keys(const o2::trd::ChamberSpacePoint& x)
  {
    uint32_t detrow = 1000 * x.getDetector() + 8 * x.getPadRow();
    uint32_t mcmcol = uint32_t(x.getPadCol() / float(o2::trd::constants::NCOLMCM));

    // float c = x.getPadCol() - float(mcmcol * o2::trd::constants::NCOLMCM);
    float c = x.getMCMChannel(mcmcol);

    if (c >= 19.0 && mcmcol >= 1) {
      return {detrow + mcmcol - 1, detrow + mcmcol};
    } else if (c <= 1.0 && mcmcol <= 6) {
      return {detrow + mcmcol, detrow + mcmcol + 1};
    } else {
      return {detrow + mcmcol};
    }
  }

  static int getDetector(uint32_t k) { return k / 1000; }
  // static int getPadRow(key) {return (key%1000) / 8;}
  static int getMcmRowCol(uint32_t k) { return k % 1000; }
};

// template instantion and non-template wrapper function
template std::vector<RawDataSpan> RawDataSpan::iterateBy<MCM_ID>();
std::vector<RawDataSpan> RawDataSpan::iterateByMCM() { return iterateBy<MCM_ID>(); }

// I started to implement a struct to iterate by detector, but did not finish this
// struct DetectorID {
//   /// The static `key` method calculates a padrow ID for digits and tracklets
//   template <typename T>
//   static uint32_t key(const T& x)
//   {
//     return x.getDetector();
//   }

//   static std::vector<uint32_t> keys(const o2::trd::ChamberSpacePoint& x)
//   {
//     uint32_t key = x.getDetector();
//     return {key};
//   }

//   static bool match(const uint32_t key, const o2::trd::ChamberSpacePoint& x)
//   {
//     return key == x.getDetector();
//   }
// };

std::vector<TrackSegment> RawDataSpan::makeMCTrackSegments()
{
  // define a struct to keep track of the first and last MC hit of one track in one chamber
  struct SegmentInfo {
    // The first hit is the hit closest to the anode region, i.e. with the largest x coordinate.
    size_t firsthit{0}; //
    // The last hit is the hit closest to the radiator, i.e. with the smallest x coordinate.
    size_t lasthit{0};
    float start{-999.9}; // local x cordinate of the first hit, init value ensures any hit updates
    float end{999.9};    // local x cordinate of the last hit, init value ensures any hit updates
  };
  // Keep information about found track segments in a map indexed by track ID and detector number.
  // If the span only covers (part of) a detector, the detector information is redundant, but in
  // the case of processing a whole event, the distinction by detector will be needed.
  std::map<std::pair<int, int>, SegmentInfo> trackSegmentInfo;

  for (int iHit = 0; iHit < hits.size(); ++iHit) {
    auto hit = hits[iHit];

    // in the following, we will look for track segments using hits in the drift region
    if (hit.isFromDriftRegion()) {
      // The first hit is the hit closest to the anode region, i.e. with the largest x coordinate.
      auto id = std::make_pair(hit.getID(), hit.getDetector());
      if (hit.getX() > trackSegmentInfo[id].start) {
        trackSegmentInfo[id].firsthit = iHit;
        trackSegmentInfo[id].start = hit.getX();
      }
      // The last hit is the hit closest to the radiator, i.e. with the smallest x coordinate.
      if (hit.getX() < trackSegmentInfo[id].end) {
        trackSegmentInfo[id].lasthit = iHit;
        trackSegmentInfo[id].end = hit.getX();
      }
    }
  } // hit loop

  std::vector<TrackSegment> trackSegments;
  for (auto x : trackSegmentInfo) {
    auto trackid = x.first.first;
    auto detector = x.first.second;
    auto firsthit = hits[x.second.firsthit];
    auto lasthit = hits[x.second.lasthit];
    trackSegments.emplace_back(firsthit, lasthit, trackid);
  }
  return trackSegments;
}

/// The RawDataManager constructor: connects all data files and sets up trees, readers etc.
RawDataManager::RawDataManager(std::filesystem::path dir)
{

  if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
    O2ERROR("'%s' is not a directory", dir.c_str());
    return;
  }

  // We allways need the trigger records, which are stored in trdtracklets.root.
  // While at it, let's also set up reading the tracklets.
  if (!std::filesystem::exists(dir / "trdtracklets.root")) {
    O2ERROR("'tracklets.root' not found in directory '%s'", dir.c_str());
    return;
  }

  mMainFile = new TFile((dir / "trdtracklets.root").c_str());
  mMainFile->GetObject("o2sim", mDataTree);

  // set up the branches we want to read
  mDataTree->SetBranchAddress("Tracklet", &mTracklets);
  mDataTree->SetBranchAddress("TrackTrg", &mTrgRecords);

  if (std::filesystem::exists(dir / "trddigits.root")) {
    mDataTree->AddFriend("o2sim", (dir / "trddigits.root").c_str());
    mDataTree->SetBranchAddress("TRDDigit", &mDigits);
  }

  if (std::filesystem::exists(dir / "o2match_itstpc.root")) {
    mDataTree->AddFriend("matchTPCITS", (dir / "o2match_itstpc.root").c_str());
    mDataTree->SetBranchAddress("TPCITS", &mTracks);
  }

  // For data, we need info about time frames to match ITS and TPC tracks to trigger records.
  if (std::filesystem::exists(dir / "o2_tfidinfo.root")) {
    TFile fInTFID((dir / "o2_tfidinfo.root").c_str());
    mTFIDs = (std::vector<o2::dataformats::TFIDInfo>*)fInTFID.Get("tfidinfo");
  }

  // For MC, we first read the collision context
  if (std::filesystem::exists(dir / "collisioncontext.root")) {
    TFile fInCollCtx((dir / "collisioncontext.root").c_str());
    mCollisionContext = (o2::steer::DigitizationContext*)fInCollCtx.Get("DigitizationContext");
    // mCollisionContext->printCollisionSummary();
  }

  // We create the MC TTree using event header and tracks from the kinematics file
  if (std::filesystem::exists(dir / "o2sim_Kine.root")) {
    mMCFile = new TFile((dir / "o2sim_Kine.root").c_str());
    mMCFile->GetObject("o2sim", mMCTree);
    mMCTree->SetBranchAddress("MCEventHeader.", &mMCEventHeader);
    mMCTree->SetBranchAddress("MCTrack", &mMCTracks);
  }

  // We then add the TRD hits to the MC tree
  if (mMCFile && std::filesystem::exists(dir / "o2sim_HitsTRD.root")) {
    mMCTree->AddFriend("o2sim", (dir / "o2sim_HitsTRD.root").c_str());
    mMCTree->SetBranchAddress("TRDHit", &mHits);
  }
}

bool RawDataManager::nextTimeFrame()
{
  if (!mDataTree->GetEntry(mTimeFrameNo)) {
    // loading time frame will fail at end of file
    return false;
  }

  mEventNo = 0;
  mTimeFrameNo++;

  O2INFO("Loaded data for time frame #%d with %d TRD trigger records, %d digits and %d tracklets",
         mTimeFrameNo, mTrgRecords->size(), mDigits->size(), mTracklets->size());

  return true;
}

bool RawDataManager::nextEvent()
{
  // get the next trigger record
  if (mEventNo >= mTrgRecords->size()) {
    return false;
  }
  mTriggerRecord = mTrgRecords->at(mEventNo);
  O2INFO("Processing event: orbit %d bc %04d with %d digits and %d tracklets",
         mTriggerRecord.getBCData().orbit, mTriggerRecord.getBCData().bc,
         mTriggerRecord.getNumberOfDigits(), mTriggerRecord.getNumberOfTracklets());

  if (mCollisionContext) {

    // clear MC data
    mHitPoints.clear();

    for (int i = 0; i < mCollisionContext->getNCollisions(); ++i) {
      auto evrec = mCollisionContext->getEventRecords()[i];
      if (abs(mTriggerRecord.getBCData().differenceInBCNS(evrec)) <= 3000) {
        // if (mMCReader) {
        mMCTree->GetEntry(i);
        // }

        O2INFO("Loaded matching MC event #%d with time offset %f ns and %d hits",
               i, mTriggerRecord.getBCData().differenceInBCNS(evrec), mHits->size());

        // convert hits to spacepoints
        auto ctrans = o2::trd::CoordinateTransformer::instance();
        for (auto& hit : *mHits) {
          mHitPoints.emplace_back(ctrans->MakeSpacePoint(hit), hit.GetCharge());
        }
      }
    }
  }

  mEventNo++;
  return true;
}

RawDataSpan RawDataManager::getEvent()
{
  RawDataSpan ev;

  ev.digits = boost::make_iterator_range_n(mDigits->begin() + mTriggerRecord.getFirstDigit(), mTriggerRecord.getNumberOfDigits());
  ev.tracklets = boost::make_iterator_range_n(mTracklets->begin() + mTriggerRecord.getFirstTracklet(), mTriggerRecord.getNumberOfTracklets());

  ev.hits = boost::make_iterator_range(mHitPoints.begin(), mHitPoints.end());

  auto evtime = getTriggerTime();

  // if (tpctracks) {
  //   for (auto &track : *mTpcTracks) {
  //     //   // auto tracktime = track.getTimeMUS().getTimeStamp();
  //     auto dtime = track.getTime0() / 5.0 - evtime;
  //     if (dtime > mMatchTimeMinTPC && dtime < mMatchTimeMaxTPC) {
  //       ev.mTpcTracks.push_back(track);
  //     }
  //   }
  // }

  if (mTracks) {
    for (auto& track : *mTracks) {
      //   // auto tracktime = track.getTimeMUS().getTimeStamp();
      // auto dtime = track.getTimeMUS().getTimeStamp() - evtime;
      // if (dtime > mMatchTimeMinTPC && dtime < mMatchTimeMaxTPC) {
      //   ev.tracks.push_back(track);

      // for(int ly=0; ly<6; ++ly) {
      //   auto point = extra.extrapolate(track.getParamOut(), ly);
      //   if (point.isValid()) {
      //     ev.evtrackpoints.push_back(point);
      //   }
      // }
      // }
    }
  }

  // ev.trackpoints.begin() = ev.evtrackpoints.begin();
  // ev.trackpoints.end() = ev.evtrackpoints.end();

  return ev;
}

o2::dataformats::TFIDInfo RawDataManager::getTimeFrameInfo()
{
  if (mTFIDs) {
    return mTFIDs->at(mTimeFrameNo - 1);
  } else {
    return o2::dataformats::TFIDInfo();
  }
}

float RawDataManager::getTriggerTime()
{
  auto tfid = getTimeFrameInfo();

  if (tfid.isDummy()) {
    return mTriggerRecord.getBCData().bc2ns() * 1e-3;
  } else {
    o2::InteractionRecord intrec = {0, tfid.firstTForbit};
    return mTriggerRecord.getBCData().differenceInBCMUS(intrec);
  }
}

std::string RawDataManager::describeFiles()
{
  std::ostringstream out;
  if (!mMainFile) {
    out << "RawDataManager is not connected to any files" << std::flush;
    return out.str();
  }
  if (!mDataTree) {
    out << "ERROR: main datatree not connected" << std::flush;
    return out.str();
  }
  out << "Main file:" << mMainFile->GetPath() << " has " << mDataTree->GetEntries() << " time frames " << std::endl;
  if (mDataTree->GetFriend("TRDDigit")) {
    out << "digits" << std::endl;
  }
  if (mDataTree->GetFriend("TPCITS")) {
    out << "tpc its matches" << std::endl;
  }

  if (mTFIDs) {
    out << mTFIDs->size() << " TFIDs were read from o2_tfidinfo.root" << std::flush;
  }
  return out.str();
}

std::string RawDataManager::describeTimeFrame()
{
  std::ostringstream out;
  out << "## Time frame " << mTimeFrameNo << ": ";
  // out << mDatareader->GetEntries() << "";
  return out.str();
}

std::string RawDataManager::describeEvent()
{
  std::ostringstream out;
  out << "## TF:Event " << mTimeFrameNo << ":" << mEventNo << ":  "
      //  << hits->getsize() << " hits   "
      << mTriggerRecord.getNumberOfDigits() << " digits and "
      << mTriggerRecord.getNumberOfTracklets() << " tracklets";
  return out.str();
}
