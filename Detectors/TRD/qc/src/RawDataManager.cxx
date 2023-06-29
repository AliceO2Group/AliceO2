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

#include <TSystem.h>
#include <TFile.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <iterator>
#include "TRDQC/CoordinateTransformer.h"
#include "Framework/Logger.h"

using namespace o2::trd;

/// comparison function to order digits by det / row / MCM / -channel
bool comp_digit(const o2::trd::Digit& a, const o2::trd::Digit& b)
{
  if (a.getDetector() != b.getDetector())
    return a.getDetector() < b.getDetector();

  if (a.getPadRow() != b.getPadRow())
    return a.getPadRow() < b.getPadRow();

  if (a.getROB() != b.getROB())
    return a.getROB() < b.getROB();

  if (a.getMCM() != b.getMCM())
    return a.getMCM() < b.getMCM();

  // sort channels in descending order, to ensure ordering of pad columns
  if (a.getChannel() != b.getChannel())
    return a.getChannel() > b.getChannel();

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

  if (a_det_row != b_det_row)
    return a_det_row < b_det_row;

  auto a_col_pos = a.getTrackletWord() & col_pos_mask;
  auto b_col_pos = b.getTrackletWord() & col_pos_mask;

  return a_col_pos < b_col_pos;
};

bool comp_spacepoint(const ChamberSpacePoint& a, const ChamberSpacePoint& b)
{
  if (a.getDetector() != b.getDetector())
    return a.getDetector() < b.getDetector();

  if (a.getPadRow() != b.getPadRow())
    return a.getPadRow() < b.getPadRow();

  if (a.getPadCol() != b.getPadCol())
    return a.getPadCol() < b.getPadCol();

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
// std::vector<RawDataSpan> RawDataSpan::IterateBy()
{
  // an map for keeping track which ranges correspond to which key
  std::map<uint32_t, RawDataSpan> spanmap;

  // sort digits and tracklets
  sort();

  // add all the digits to a map
  for (auto cur = digits.b; cur != digits.e; /* noop */) {
    // calculate the key of the current (first unprocessed) digit
    auto key = keyfunc::key(*cur);
    // find the first digit with a different key
    auto nxt = std::find_if(cur, digits.e, [key](auto x) { return keyfunc::key(x) != key; });
    // store the range cur:nxt in the map
    spanmap[key].digits.b = cur;
    spanmap[key].digits.e = nxt;
    // continue after this range
    cur = nxt;
  }

  // add tracklets to the map
  for (auto cur = tracklets.b; cur != tracklets.e; /* noop */) {
    auto key = keyfunc::key(*cur);
    auto nxt = std::find_if(cur, tracklets.e, [key](auto x) { return keyfunc::key(x) != key; });
    spanmap[key].tracklets.b = cur;
    spanmap[key].tracklets.e = nxt;
    cur = nxt;
  }

  // spanmap contains all TRD data - either digits or tracklets
  // Now we complete these spans with hit information
  for (auto cur = hits.b; cur != hits.e; ++cur) {
    for (auto key : keyfunc::keys(*cur)) {
      if (spanmap[key].hits.b == spanmap[key].hits.e) {
        spanmap[key].hits.b = cur;
        spanmap[key].hits.e = cur;
      }
      ++spanmap[key].hits.e;
    }
  }

  // for (auto cur = event.trackpoints.b; cur != event.trackpoints.e; /* noop */) {
  //   auto nxt = std::adjacent_find(cur, event.trackpoints.e, classifier::comp_trackpoints);
  //   if (nxt != event.trackpoints.e) { ++nxt; }
  //   (*this)[classifier::key(*cur)].trackpoints.b = cur;
  //   (*this)[classifier::key(*cur)].trackpoints.e = nxt;
  //   cur = nxt;
  // }
  // cout << "Found " << this->size() << " spans" << endl;

  // for (auto &[key, sp] : spanmap) {
  //   std::cout << key << ": " << sp.digits.length() << " digits, "
  //        << sp.tracklets.length() << " tracklets, "
  //        << sp.hits.length() << " hits" << std::endl;
  // }

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

  static std::vector<uint32_t> keys(const o2::trd::ChamberSpacePoint& x)
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

  static std::vector<uint32_t> keys(const o2::trd::ChamberSpacePoint& x)
  {
    uint32_t detrow = 1000 * x.getDetector() + 8 * x.getPadRow();
    uint32_t mcmcol = uint32_t(x.getPadCol() / float(o2::trd::constants::NCOLMCM));

    float c = x.getPadCol() - float(mcmcol * o2::trd::constants::NCOLMCM);

    if (c <= 2.0 && mcmcol >= 1) {
      return {detrow + mcmcol - 1, detrow + mcmcol};
    } else if (c >= 20 && mcmcol <= 6) {
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
  mDataReader = new TTreeReader(mDataTree);

  // set up the branches we want to read
  mTracklets = new TTreeReaderArray<o2::trd::Tracklet64>(*mDataReader, "Tracklet");
  mTrgRecords = new TTreeReaderArray<o2::trd::TriggerRecord>(*mDataReader, "TrackTrg");

  if (std::filesystem::exists(dir / "trddigits.root")) {
    mDataTree->AddFriend("o2sim", (dir / "trddigits.root").c_str());
    mDigits = new TTreeReaderArray<o2::trd::Digit>(*mDataReader, "TRDDigit");
  }

  if (std::filesystem::exists(dir / "o2match_itstpc.root")) {
    mDataTree->AddFriend("matchTPCITS", (dir / "o2match_itstpc.root").c_str());
    mDigits = new TTreeReaderArray<o2::trd::Digit>(*mDataReader, "TPCITS");
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
    mMCReader = new TTreeReader(mMCTree);
    mMCEventHeader = new TTreeReaderValue<o2::dataformats::MCEventHeader>(*mMCReader, "MCEventHeader.");
    mMCTracks = new TTreeReaderArray<o2::MCTrackT<Float_t>>(*mMCReader, "MCTrack");
  }

  // We then add the TRD hits to the MC tree
  if (mMCFile && std::filesystem::exists(dir / "o2sim_HitsTRD.root")) {
    mMCTree->AddFriend("o2sim", (dir / "o2sim_HitsTRD.root").c_str());
    mHits = new TTreeReaderArray<o2::trd::Hit>(*mMCReader, "TRDHit");
  }
}

bool RawDataManager::nextTimeFrame()
{
  if (!mDataReader->Next()) {
    // loading time frame will fail at end of file
    return false;
  }

  mEventNo = 0;
  mTimeFrameNo++;

  O2INFO("Loaded data for time frame #%d with %d TRD trigger records, %d digits and %d tracklets",
         mTimeFrameNo, mTrgRecords->GetSize(), mDigits->GetSize(), mTracklets->GetSize());

  return true;
}

bool RawDataManager::nextEvent()
{
  // get the next trigger record
  if (mEventNo >= mTrgRecords->GetSize()) {
    return false;
  }

  mTriggerRecord = mTrgRecords->At(mEventNo);
  O2INFO("Processing event: orbit %d bc %04d with %d digits and %d tracklets",
         mTriggerRecord.getBCData().orbit, mTriggerRecord.getBCData().bc,
         mTriggerRecord.getNumberOfDigits(), mTriggerRecord.getNumberOfTracklets());

  if (mCollisionContext) {

    // clear MC data
    mHitPoints.clear();

    for (int i = 0; i < mCollisionContext->getNCollisions(); ++i) {
      auto evrec = mCollisionContext->getEventRecords()[i];
      if (abs(mTriggerRecord.getBCData().differenceInBCNS(evrec)) <= 3000) {
        if (mMCReader) {
          mMCReader->SetEntry(i);
        }

        O2INFO("Loaded matching MC event #%d with time offset %f ns and %d hits",
               i, mTriggerRecord.getBCData().differenceInBCNS(evrec), mHits->GetSize());

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
  ev.digits.b = mDigits->begin() + mTriggerRecord.getFirstDigit();
  ev.digits.e = ev.digits.begin() + mTriggerRecord.getNumberOfDigits();
  ev.tracklets.b = mTracklets->begin() + mTriggerRecord.getFirstTracklet();
  ev.tracklets.e = ev.tracklets.begin() + mTriggerRecord.getNumberOfTracklets();

  ev.hits.b = mHitPoints.begin();
  ev.hits.e = mHitPoints.end();

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

  // ev.trackpoints.b = ev.evtrackpoints.begin();
  // ev.trackpoints.e = ev.evtrackpoints.end();

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
      //  << hits->GetSize() << " hits   "
      << mTriggerRecord.getNumberOfDigits() << " digits and "
      << mTriggerRecord.getNumberOfTracklets() << " tracklets";
  return out.str();
}
