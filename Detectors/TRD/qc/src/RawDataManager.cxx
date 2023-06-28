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

bool comp_spacepoint(const ChamberSpacePoint &a, const ChamberSpacePoint &b)
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
std::vector<RawDataSpan> RawDataSpan::IterateBy()
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
  std::cout << "-----------------" << std::endl;
  for (auto cur = hits.b; cur != hits.e; ++cur) {
    // std::cout << (*cur) << std::endl;
    for (auto key: keyfunc::keys(*cur)) {
      if (spanmap[key].hits.b == spanmap[key].hits.e) {
        spanmap[key].hits.b = cur;
        spanmap[key].hits.e = cur;
        // std::cout << "START " << key << ": " << (*spanmap[key].hits.b) << std::endl;
      }
      ++spanmap[key].hits.e;
      // std::cout << " INCR " << key << ": " << std::distance(spanmap[key].hits.b, spanmap[key].hits.e) << std::endl;
      // std::cout << std::endl;
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

template std::vector<RawDataSpan> RawDataSpan::IterateBy<PadRowID>();
template std::vector<RawDataSpan> RawDataSpan::IterateBy<MCM_ID>();

// template <typename T>
// int mcmkey(const T& x)
// {
//   return 1000 * x.getDetector() + 10 * x.getPadRow() + 4 * (x.getROB() % 2) + x.getMCM() % 4;
// }

// std::vector<RawDataSpan> RawDataSpan::ByMCM()
// {
//   // we manage all
//   std::map<uint32_t, RawDataSpan> spanmap;

//   // sort digits and tracklets
//   sort();

//   // add all the digits to a map
//   for (auto cur = digits.b; cur != digits.e; /* noop */) {
//     // calculate the key of the current (first unprocessed) digit
//     auto key = mcmkey(*cur);
//     // find the first digit with a different key
//     auto nxt = std::find_if(cur, digits.e, [key](auto x) { return mcmkey(x) != key; });
//     // store the range cur:nxt in the map
//     spanmap[key].digits.b = cur;
//     spanmap[key].digits.e = nxt;
//     // continue after this range
//     cur = nxt;
//   }

//   // add tracklets to the map
//   for (auto cur = tracklets.b; cur != tracklets.e; /* noop */) {
//     auto key = mcmkey(*cur);
//     auto nxt = std::find_if(cur, tracklets.e, [key](auto x) { return mcmkey(x) != key; });
//     spanmap[key].tracklets.b = cur;
//     spanmap[key].tracklets.e = nxt;
//     cur = nxt;
//   }

//   // for (auto cur = event.trackpoints.b; cur != event.trackpoints.e; /* noop */) {
//   //   auto nxt = std::adjacent_find(cur, event.trackpoints.e, classifier::comp_trackpoints);
//   //   if (nxt != event.trackpoints.e) { ++nxt; }
//   //   (*this)[classifier::key(*cur)].trackpoints.b = cur;
//   //   (*this)[classifier::key(*cur)].trackpoints.e = nxt;
//   //   cur = nxt;
//   // }
//   // cout << "Found " << this->size() << " spans" << endl;

//   std::vector<RawDataSpan> spans;

//   transform(spanmap.begin(), spanmap.end(), back_inserter(spans), [](auto const& pair) { return pair.second; });

//   return spans;
// }

RawDataManager::RawDataManager(std::filesystem::path dir)
{

  if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
    std::cerr << "'" << dir << "' is not a directory" << std::endl;
    return;
  }

  // We allways need the trigger records, which are stored in trdtracklets.root.
  // While at it, let's also set up reading the tracklets.
  if (!std::filesystem::exists(dir / "trdtracklets.root")) {
    std::cerr << "'tracklets.root' not found in directory '" << dir << "'" << std::endl;
    return;
  }

  mMainfile = new TFile((dir / "trdtracklets.root").c_str());
  mMainfile->GetObject("o2sim", mDatatree);
  mDatareader = new TTreeReader(mDatatree);

  // set up the branches we want to read
  mTracklets = new TTreeReaderArray<o2::trd::Tracklet64>(*mDatareader, "Tracklet");
  mTrgRecords = new TTreeReaderArray<o2::trd::TriggerRecord>(*mDatareader, "TrackTrg");

  // For data, we need info about time frames to match ITS and TPC tracks to trigger records.
  if (std::filesystem::exists(dir / "o2_tfidinfo.root")) {
    TFile fInTFID((dir / "o2_tfidinfo.root").c_str());
    mTFIDs = (std::vector<o2::dataformats::TFIDInfo>*)fInTFID.Get("tfidinfo");
  }

  // For MC, we read the collision context
  if (std::filesystem::exists(dir / "collisioncontext.root")) {
    TFile fInCollCtx((dir / "collisioncontext.root").c_str());
    mCollisionContext = (o2::steer::DigitizationContext*) fInCollCtx.Get("DigitizationContext");
    mCollisionContext->printCollisionSummary();
  }

  // If there are MC hits, we load them
  if (std::filesystem::exists(dir / "o2sim_HitsTRD.root")) {
    mHitsFile = new TFile((dir / "o2sim_HitsTRD.root").c_str());
    mHitsFile->GetObject("o2sim", mHitsTree);
    mHitsReader = new TTreeReader(mHitsTree);
    mHits = new TTreeReaderArray<o2::trd::Hit>(*mHitsReader, "TRDHit");
    std::cout << "connect MC hits file" << std::endl;
  }

  if (std::filesystem::exists(dir / "o2sim_MCHeader.root")) {
    mMCFile = new TFile((dir / "o2sim_MCHeader.root").c_str());
    mMCFile->GetObject("o2sim", mMCTree);
    mMCReader = new TTreeReader(mMCTree);
    mMCEventHeader = new TTreeReaderValue<o2::dataformats::MCEventHeader>(*mMCReader, "MCEventHeader.");
    std::cout << "connect MC event header file" << std::endl;
  }

  // Let's try to add other data
  addReaderArray(mDigits, dir / "trddigits.root", "o2sim", "TRDDigit");
  // AddReaderArray(mTpcTracks, dir + "tpctracks.root", "tpcrec", "TPCTracks");
  addReaderArray(mTracks, dir / "o2match_itstpc.root", "matchTPCITS", "TPCITS");

  // ConnectMCHitsFile(dir+"o2sim_HitsTRD.root");
}

template <typename T>
void RawDataManager::addReaderArray(TTreeReaderArray<T>*& array, std::filesystem::path file, std::string tree, std::string branch)
{
  if (!std::filesystem::exists(file)) {
    // return value TRUE indicates file was not found
    return;
  }

  // the file exists, everything else should work
  mDatatree->AddFriend(tree.c_str(), file.c_str());
  array = new TTreeReaderArray<T>(*mDatareader, branch.c_str());
}

bool RawDataManager::nextTimeFrame()
{
  if (mDatareader->Next()) {
    mEventNo = 0;
    mTimeFrameNo++;

    // Fix of position/slope should no longer be necessary
    // for (auto &tracklet : *mTracklets) {
    //   tracklet.setPosition(tracklet.getPosition() ^ 0x80);
    //   tracklet.setSlope(tracklet.getSlope() ^ 0x80);
    // }

    return true;
  } else {
    return false;
  }
}

bool RawDataManager::nextEvent()
{
  // get the next trigger record
  if (mEventNo >= mTrgRecords->GetSize()) {
    return false;
  }

  mTriggerRecord = mTrgRecords->At(mEventNo);
  std::cout << mTriggerRecord << std::endl;


  if (mCollisionContext) {

    // clear MC data
    mHitPoints.clear();

    for (int i=0; i<mCollisionContext->getNCollisions(); ++i) {
      if ( abs(mTriggerRecord.getBCData().differenceInBCNS(mCollisionContext->getEventRecords()[i])) == 0 ) {
        std::cout << "using collision " << i << std::endl;
        if (mHitsReader) {
          mHitsReader->SetEntry(i);
        }
        if(mMCReader) {
          mMCReader->SetEntry(i);
        }

        // convert hits to spacepoints
        auto ctrans = o2::trd::CoordinateTransformer::instance();
        for( auto& hit : *mHits) {
          mHitPoints.emplace_back(ctrans->MakeSpacePoint(hit));
        }
      }
    }
    // // load the hits for the next event
    // if (mHitsReader) {
    //   if (mEventNo == 1) {
    //     std::cout << "skip 2 MC events" << std::endl;
    //     mHitsReader->Next();
    //     mHitsReader->Next();
    //   }

    //   if (!mHitsReader->Next()) {
    //     std::cout << "no hits found for event " << mTimeFrameNo << ":" << mEventNo << std::endl;
    //     return false;
    //   }

  }

  mEventNo++;
  if(mMCReader) {
    O2INFO("loaded event: MC time = %fns, ID %d", mMCEventHeader->Get()->GetT(), mMCEventHeader->Get()->GetEventID());
    mMCEventHeader->Get()->printInfo();
  }
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
  if (!mMainfile) {
    out << "RawDataManager is not connected to any files" << std::flush;
    return out.str();
  }
  if (!mDatatree) {
    out << "ERROR: main datatree not connected" << std::flush;
    return out.str();
  }
  out << "Main file:" << mMainfile->GetPath() << " has " << mDatatree->GetEntries() << " time frames " << std::endl;
  if (mDatatree->GetFriend("TRDDigit")) {
    out << "digits" << std::endl;
  }
  if (mDatatree->GetFriend("TPCITS")) {
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
