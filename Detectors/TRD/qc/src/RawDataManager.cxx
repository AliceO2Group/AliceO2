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

using namespace o2::trd;

/// comparison function to order digits by det / row / MCM / channel
bool comp_digit(const o2::trd::Digit &a, const o2::trd::Digit &b)
{
  if (a.getDetector() != b.getDetector())
    return a.getDetector() < b.getDetector();

  if (a.getPadRow() != b.getPadRow())
    return a.getPadRow() < b.getPadRow();

  if (a.getROB() != b.getROB())
    return a.getROB() < b.getROB();

  if (a.getMCM() != b.getMCM())
    return a.getMCM() < b.getMCM();

  if (a.getChannel() != b.getChannel())
    return a.getChannel() < b.getChannel();

  return true;
}


/// comparison function to order tracklets by det / row / MCM / channel
bool comp_tracklet(const o2::trd::Tracklet64 &a, const o2::trd::Tracklet64 &b)
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

// bool comp_spacepoint(const ChamberSpacePoint &a, const ChamberSpacePoint &b)
// {
//   if (a.getDetector() != b.getDetector())
//     return a.getDetector() < b.getDetector();

//   if (a.getPadRow() != b.getPadRow())
//     return a.getPadRow() < b.getPadRow();

//   if (a.getPadCol() != b.getPadCol())
//     return a.getPadCol() < b.getPadCol();

//   return true;
// }

void RawDataSpan::sort()
{
  std::stable_sort(std::begin(digits), std::end(digits), comp_digit);
  std::stable_sort(std::begin(tracklets), std::end(tracklets), comp_tracklet);
}

template<typename keyfunc>
std::vector<RawDataSpan> RawDataSpan::IterateBy()
// std::vector<RawDataSpan> RawDataSpan::IterateBy()
{
  // an map for keeping track which ranges correspond to which key
  std::map<uint32_t,RawDataSpan> spanmap;

  // sort digits and tracklets
  sort();

  // add all the digits to a map
  for (auto cur = digits.b; cur != digits.e; /* noop */ ) {
    // calculate the key of the current (first unprocessed) digit
    auto key = keyfunc::key(*cur);
    // find the first digit with a different key
    auto nxt = std::find_if(cur, digits.e, [key](auto x) {return keyfunc::key(x) != key;});
    // store the range cur:nxt in the map
    spanmap[key].digits.b = cur; 
    spanmap[key].digits.e = nxt;
    // continue after this range
    cur = nxt;
  }

  // add tracklets to the map
  for (auto cur = tracklets.b; cur != tracklets.e; /* noop */) {
    auto key = keyfunc::key(*cur);
    auto nxt = std::find_if(cur, tracklets.e, [key](auto x) {return keyfunc::key(x) != key;});
    spanmap[key].tracklets.b = cur; 
    spanmap[key].tracklets.e = nxt;
    cur = nxt;
  }

  // for (auto cur = event.trackpoints.b; cur != event.trackpoints.e; /* noop */) {
  //   auto nxt = std::adjacent_find(cur, event.trackpoints.e, classifier::comp_trackpoints);
  //   if (nxt != event.trackpoints.e) { ++nxt; }
  //   (*this)[classifier::key(*cur)].trackpoints.b = cur; 
  //   (*this)[classifier::key(*cur)].trackpoints.e = nxt;
  //   cur = nxt;
  // }
  // cout << "Found " << this->size() << " spans" << endl;

  std::vector<RawDataSpan> spans;

  transform(spanmap.begin(), spanmap.end(), back_inserter(spans), [](auto const& pair) { return pair.second; });
  
  return spans;
}

template std::vector<RawDataSpan> RawDataSpan::IterateBy<PadRowID>();
template std::vector<RawDataSpan> RawDataSpan::IterateBy<MCM_ID>();


template<typename T>
int mcmkey(const T &x) { 
  return 1000*x.getDetector() + 10*x.getPadRow() + 4*(x.getROB()%2) + x.getMCM()%4;
}

std::vector<RawDataSpan> RawDataSpan::ByMCM()
{
  // we manage all 
  std::map<uint32_t,RawDataSpan> spanmap;

  // sort digits and tracklets
  sort();

  // add all the digits to a map
  for (auto cur = digits.b; cur != digits.e; /* noop */ ) {
    // calculate the key of the current (first unprocessed) digit
    auto key = mcmkey(*cur);
    // find the first digit with a different key
    auto nxt = std::find_if(cur, digits.e, [key](auto x) {return mcmkey(x) != key;});
    // store the range cur:nxt in the map
    spanmap[key].digits.b = cur; 
    spanmap[key].digits.e = nxt;
    // continue after this range
    cur = nxt;
  }

  // add tracklets to the map
  for (auto cur = tracklets.b; cur != tracklets.e; /* noop */) {
    auto key = mcmkey(*cur);
    auto nxt = std::find_if(cur, tracklets.e, [key](auto x) {return mcmkey(x) != key;});
    spanmap[key].tracklets.b = cur; 
    spanmap[key].tracklets.e = nxt;
    cur = nxt;
  }

  // for (auto cur = event.trackpoints.b; cur != event.trackpoints.e; /* noop */) {
  //   auto nxt = std::adjacent_find(cur, event.trackpoints.e, classifier::comp_trackpoints);
  //   if (nxt != event.trackpoints.e) { ++nxt; }
  //   (*this)[classifier::key(*cur)].trackpoints.b = cur; 
  //   (*this)[classifier::key(*cur)].trackpoints.e = nxt;
  //   cur = nxt;
  // }
  // cout << "Found " << this->size() << " spans" << endl;

  std::vector<RawDataSpan> spans;

  transform(spanmap.begin(), spanmap.end(), back_inserter(spans), [](auto const& pair) { return pair.second; });

  return spans;
}


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
  if (std::filesystem::exists(dir/"o2_tfidinfo.root")) {
    TFile fInTFID((dir/"o2_tfidinfo.root").c_str());
    mTFIDs = (std::vector<o2::dataformats::TFIDInfo> *)fInTFID.Get("tfidinfo");
  }

  // If there are MC hits, we load them
  if (std::filesystem::exists(dir/"o2sim_HitsTRD.root")) {
    mHitsFile = new TFile((dir/"o2sim_HitsTRD.root").c_str());
    mHitsFile->GetObject("o2sim", mHitsTree);
    mHitsReader = new TTreeReader(mHitsTree);
    mHits = new TTreeReaderArray<o2::trd::Hit>(*mHitsReader, "TRDHit");
    std::cout << "connect MC hits file" << std::endl;
  }


  // Let's try to add other data
  AddReaderArray(mDigits, dir / "trddigits.root", "o2sim", "TRDDigit");
  // AddReaderArray(mTpcTracks, dir + "tpctracks.root", "tpcrec", "TPCTracks");
  AddReaderArray(mTracks, dir / "o2match_itstpc.root", "matchTPCITS", "TPCITS");

  // ConnectMCHitsFile(dir+"o2sim_HitsTRD.root");
}

template <typename T>
void RawDataManager::AddReaderArray(TTreeReaderArray<T> *&array, std::filesystem::path file, std::string tree, std::string branch)
{
  if (!std::filesystem::exists(file)) {
    // return value TRUE indicates file was not found 
    return;
  }

  // the file exists, everything else should work
  mDatatree->AddFriend(tree.c_str(), file.c_str());
  array = new TTreeReaderArray<T>(*mDatareader, branch.c_str());
}

bool RawDataManager::NextTimeFrame()
{
  if (mDatareader->Next())
  {
    mEventNo = 0;
    mTimeFrameNo++;

    // Fix of position/slope should no longer be necessary
    // for (auto &tracklet : *mTracklets) {
    //   tracklet.setPosition(tracklet.getPosition() ^ 0x80);
    //   tracklet.setSlope(tracklet.getSlope() ^ 0x80);
    // }

    return true;
  }
  else
  {
    return false;
  }
}

bool RawDataManager::NextEvent()
{
  // get the next trigger record
  if (mEventNo >= mTrgRecords->GetSize()) {
    return false;
  }

  // load the hits for the next event
  if (mHitsReader) {
    if ( ! mHitsReader->Next() ) {
      std::cout << "no hits found for event " << mTimeFrameNo << ":" << mEventNo << std::endl;
      return false;
    }
  }

  mTriggerRecord = mTrgRecords->At(mEventNo);
  // std::cout << mTriggerRecord << std::endl;

  mEventNo++;
  return true;
}

RawDataSpan RawDataManager::GetEvent()
{
  RawDataSpan ev;
  ev.digits.b = mDigits->begin() + mTriggerRecord.getFirstDigit();
  ev.digits.e = ev.digits.begin() + mTriggerRecord.getNumberOfDigits();
  ev.tracklets.b = mTracklets->begin() + mTriggerRecord.getFirstTracklet();
  ev.tracklets.e = ev.tracklets.begin() + mTriggerRecord.getNumberOfTracklets();

  if (mHits) {
    ev.hits.b = mHits->begin();
    ev.hits.e = mHits->end();
  } else {
    ev.hits.b = TTreeReaderArray<o2::trd::Hit>::iterator();
    ev.hits.e = TTreeReaderArray<o2::trd::Hit>::iterator();
  }

  auto evtime = GetTriggerTime();

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
    for (auto &track : *mTracks) {
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

o2::dataformats::TFIDInfo RawDataManager::GetTimeFrameInfo()
{
  if (mTFIDs) {
    return mTFIDs->at(mTimeFrameNo-1);
  } else {
    return o2::dataformats::TFIDInfo();
  }
}

float RawDataManager::GetTriggerTime()
{
  auto tfid = GetTimeFrameInfo();

  if (tfid.isDummy()) {
    return mTriggerRecord.getBCData().bc2ns() * 1e-3;
  } else {
    o2::InteractionRecord intrec = {0, tfid.firstTForbit};
    return mTriggerRecord.getBCData().differenceInBCMUS(intrec);
  }
}

std::string RawDataManager::DescribeFiles()
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

std::string RawDataManager::DescribeTimeFrame()
{
  std::ostringstream out;
  out << "## Time frame " << mTimeFrameNo << ": ";
  // out << mDatareader->GetEntries() << "";      
  return out.str();
}

std::string RawDataManager::DescribeEvent()
{
  std::ostringstream out;
  out << "## TF:Event " << mTimeFrameNo << ":" << mEventNo << ":  "
      //  << hits->GetSize() << " hits   "
      << mTriggerRecord.getNumberOfDigits() << " digits and "
      << mTriggerRecord.getNumberOfTracklets() << " tracklets";
  return out.str();
}


