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


#include "TRDBase/RawDisplay.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/HelperMethods.h"


#include <TSystem.h>
#include <TFile.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TCanvas.h>
#include <TPad.h>
#include <TH2.h>
#include <TLine.h>
#include <TMarker.h>

#include <ostream>
#include <sstream>

using namespace o2::trd;
using namespace o2::trd::rawdisp;

CoordinateTransformer::CoordinateTransformer() 
: mGeo(o2::trd::Geometry::instance()) 
{
  mGeo->createPadPlaneArray();
}

std::array<float, 3> CoordinateTransformer::Local2RCT(int det, float x, float y, float z)
{
  std::array<float, 3> rct;

  auto padPlane = mGeo->getPadPlane((det)%6, (det/6)%5);

  // array<double,3> rct;

  double iPadLen = padPlane->getLengthIPad();
  double oPadLen = padPlane->getLengthOPad();
  int nRows = padPlane->getNrows();

  double lengthCorr = padPlane->getLengthOPad()/padPlane->getLengthIPad();

  // calculate position based on inner pad length
  rct[0] = - z / padPlane->getLengthIPad() + padPlane->getNrows()/2;

  // correct row for outer pad rows
  if (rct[0] <= 1.0) {
    rct[0] = 1.0 - (1.0-rct[0])*lengthCorr;
  }

  if (rct[0] >= double(nRows-1)) {
    rct[0] = double(nRows-1) + (rct[0] - double(nRows-1))*lengthCorr;
  }

  // sanity check: is the padrow coordinate reasonable?
  if ( rct[0] < 0.0 || rct[0] > double(nRows) ) {
    std::cout << "ERROR: hit with z=" << z << ", padrow " << rct[0]
          << " outside of chamber" << std::endl;
  }

  // simple conversion of pad / local y coordinate
  // ignore different width of outer pad
  rct[1] = y / padPlane->getWidthIPad() + 144./2.;

  // time coordinate
  if (x<-0.35) {
    // drift region
    rct[2] = mT0 - (x+0.35) / (mVdrift/10.0);
  } else {
    // anode region: very rough guess
    rct[2] = mT0 - 1.0 + fabs(x);
  }

  rct[1] += (x + 0.35) * mExB;
  return rct;
}

std::array<float, 3> CoordinateTransformer::OrigLocal2RCT(int det, float x, float y, float z)
{
  std::array<float, 3> rct;

  auto padPlane = mGeo->getPadPlane((det)%6, (det/6)%5);

  // array<double,3> rct;

  double iPadLen = padPlane->getLengthIPad();
  double oPadLen = padPlane->getLengthOPad();
  int nRows = padPlane->getNrows();

  double lengthCorr = padPlane->getLengthOPad()/padPlane->getLengthIPad();

  // calculate position based on inner pad length
  rct[0] = - z / padPlane->getLengthIPad() + padPlane->getNrows()/2;

  // correct row for outer pad rows
  if (rct[0] <= 1.0) {
    rct[0] = 1.0 - (1.0-rct[0])*lengthCorr;
  }

  if (rct[0] >= double(nRows-1)) {
    rct[0] = double(nRows-1) + (rct[0] - double(nRows-1))*lengthCorr;
  }

  // sanity check: is the padrow coordinate reasonable?
  if ( rct[0] < 0.0 || rct[0] > double(nRows) ) {
    std::cout << "ERROR: hit with z=" << z << ", padrow " << rct[0]
          << " outside of chamber" << std::endl;
  }

  // simple conversion of pad / local y coordinate
  // ignore different width of outer pad
  rct[1] = y / padPlane->getWidthIPad() + 144./2.;

  // time coordinate
  if (x<-0.35) {
    // drift region
    rct[2] = mT0 - (x+0.35) / (mVdrift/10.0);
  } else {
    // anode region: very rough guess
    rct[2] = mT0 - 1.0 + fabs(x);
  }

  return rct;
}



std::ostream& operator<<(std::ostream& os, const ChamberSpacePoint& p)
{
  os << "( " << std::setprecision(5) << p.getX() 
     << " / " << std::setprecision(5) << p.getY() 
     << " / " << std::setprecision(6) << p.getZ() << ") <-> ["
     << p.getDetector() << "." << p.getPadRow() 
     << " pad " << std::setprecision(5) << p.getPadCol() << "]";
  return os;
}


/// comparison function to sort digits by det / row / MCM / channel
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


/// comparison function to sort tracklets by det / row / MCM / channel
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

void rawdisp::RawDataSpan::sort()
{
  std::stable_sort(std::begin(digits), std::end(digits), comp_digit);
  std::stable_sort(std::begin(tracklets), std::end(tracklets), comp_tracklet);
}

template<typename T>
int mcmkey(const T &x) { 
  return 1000*x.getDetector() + 10*x.getPadRow() + 4*(x.getROB()%2) + x.getMCM()%4;
}

template<typename keyfunc>
std::map<uint32_t,RawDataSpan> RawDataSpan::IterateBy()
// std::vector<RawDataSpan> RawDataSpan::IterateBy()
{
  // an map for keeping track which ranges correspond to which key
  std::map<uint32_t,RawDataSpan> resultmap;

  // sort digits and tracklets
  sort();

  // add all the digits to a map
  for (auto cur = digits.b; cur != digits.e; /* noop */ ) {
    // calculate the key of the current (first unprocessed) digit
    auto key = keyfunc::key(*cur);
    // find the first digit with a different key
    auto nxt = std::find_if(cur, digits.e, [key](auto x) {return keyfunc::key(x) != key;});
    // store the range cur:nxt in the map
    resultmap[key].digits.b = cur; 
    resultmap[key].digits.e = nxt;
    // continue after this range
    cur = nxt;
  }

  // add tracklets to the map
  for (auto cur = tracklets.b; cur != tracklets.e; /* noop */) {
    auto key = keyfunc::key(*cur);
    auto nxt = std::find_if(cur, tracklets.e, [key](auto x) {return keyfunc::key(x) != key;});
    resultmap[key].tracklets.b = cur; 
    resultmap[key].tracklets.e = nxt;
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
  return resultmap;
}

template std::map<uint32_t,RawDataSpan> RawDataSpan::IterateBy<PadRowID>();
template std::map<uint32_t,RawDataSpan> RawDataSpan::IterateBy<MCM_ID>();


std::map<uint32_t,RawDataSpan> RawDataSpan::ByMCM()
{
  std::map<uint32_t,RawDataSpan> result;

  // sort digits and tracklets
  sort();

  // add all the digits to a map
  for (auto cur = digits.b; cur != digits.e; /* noop */ ) {
    // calculate the key of the current (first unprocessed) digit
    auto key = mcmkey(*cur);
    // find the first digit with a different key
    auto nxt = std::find_if(cur, digits.e, [key](auto x) {return mcmkey(x) != key;});
    // store the range cur:nxt in the map
    result[key].digits.b = cur; 
    result[key].digits.e = nxt;
    // continue after this range
    cur = nxt;
  }

  // add tracklets to the map
  for (auto cur = tracklets.b; cur != tracklets.e; /* noop */) {
    auto key = mcmkey(*cur);
    auto nxt = std::find_if(cur, tracklets.e, [key](auto x) {return mcmkey(x) != key;});
    result[key].tracklets.b = cur; 
    result[key].tracklets.e = nxt;
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
  return result;
}


// float PadPositionMCM(o2::trd::Tracklet64 &tracklet)
// {
//   return 12.0 - (tracklet.getPositionBinSigned() * constants::GRANULARITYTRKLPOS);
// }

// int getMCMCol(int irob, int imcm)
// {
//   return (imcm % constants::NMCMROBINCOL) + constants::NMCMROBINCOL * (irob % 2);
// }

/// Modified version of o2::trd::Tracklet64::getPadCol returning a float
float rawdisp::PadColF(o2::trd::Tracklet64 &tracklet)
{
  // obtain pad number relative to MCM center
  float padLocal = tracklet.getPositionBinSigned() * constants::GRANULARITYTRKLPOS;
  // MCM number in column direction (0..7)
  int mcmCol = (tracklet.getMCM() % constants::NMCMROBINCOL) + constants::NMCMROBINCOL * (tracklet.getROB() % 2);

  // original calculation
  // FIXME: understand why the offset seems to be 6 pads and not nChannels / 2 = 10.5
  //return CAMath::Nint(6.f + mcmCol * ((float)constants::NCOLMCM) + padLocal);

  // my calculation
  return float((mcmCol + 1) * constants::NCOLMCM) + padLocal - 10.0;
}

// float UncalibratedPad(o2::trd::Tracklet64 &tracklet)
// {
//   float y = tracklet.getUncalibratedY();
//   int mcmCol = (tracklet.getMCM() % NMCMROBINCOL) + NMCMROBINCOL * (tracklet.getROB() % 2);
//   // one pad column has 144 pads, the offset of -63 is the center of the first MCM in that column
//   // which is connected to the pads -63 - 9 = -72 to -63 + 9 = -54
//   // float offset = -63.f + ((float)NCOLMCM) * mcmCol;
//   float padWidth = 0.635f + 0.03f * (tracklet.getDetector() % NLAYER);
//   return y / padWidth + 71.0;
// }

float rawdisp::SlopeF(o2::trd::Tracklet64 &trkl)
{
  return - trkl.getSlopeBinSigned() * constants::GRANULARITYTRKLSLOPE / constants::ADDBITSHIFTSLOPE;
}


// void rawdisp::RawDataSpan::calculateCoordinates()
// {
//   for (auto& x : digits) {
//     if (mDetector == -1) {
//       mDetector = x.getDetector();
//     } else if (mDetector != x.getDetector()) {
//       mDetector = -2;
//     }
//     if (mPadRow == -1) {
//       mPadRow = x.getPadRow();
//     } else if (mPadRow != x.getPadRow()) {
//       mPadRow = -2;
//     }
//   }
//   for (auto& x : tracklets) {
//     if (mDetector == -1) {
//       mDetector = x.getDetector();
//     } else if (mDetector != x.getDetector()) {
//       mDetector = -2;
//     }
//     if (mPadRow == -1) {
//       mPadRow = x.getPadRow();
//     } else if (mPadRow != x.getPadRow()) {
//       mPadRow = -2;
//     }
//   }
// }



// pair<int, int> RawDataSpan::getMaxADCsumAndChannel()
// {
//   int maxch = -1, maxval = -1;
//   for (auto digit : digits) {
//     if (digit.getADCsum() > maxval) {
//       maxval = digit.getADCsum();
//       maxch = digit.getChannel();
//     }
//   // std::cout << digit << std::endl;
//   }

//   return make_pair(maxval, maxch);
// }




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
//   std::cout << "TrackExtrapolator: load geometry" << std::endl;
//   o2::base::GeometryManager::loadGeometry();
//   std::cout << "TrackExtrapolator: load B-field" << std::endl;
//   o2::base::Propagator::initFieldFromGRP();

//   std::cout << "TrackExtrapolator: instantiate geometry" << std::endl;

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

// ChamberSpacePoint TrackExtrapolator::extrapolate(o2::track::TrackParCov& track, int layer)
// {

//   // ChamberSpacePoint pos;

//   // if (draw)
//   //   printf("Found ITS-TPC track with time %f us\n", trkITSTPC.getTimeMUS().getTimeStamp());

//   double minPtTrack = 0.5;

//   // const auto& track = trkITSTPC.getParamOut();
//   if (fabs(track.getEta()) > 0.84 || track.getPt() < minPtTrack) {
//     // no chance to find tracklets for these tracks
//     return ChamberSpacePoint(-999);
//   }

//   if (!mProp->PropagateToXBxByBz(track, mGeo->getTime0(layer))) {
//     // if (draw)
//     printf("Failed track propagation into layer %i\n", layer);
//     return ChamberSpacePoint(-999);
//   }
//   if (!adjustSector(track)) {
//     // if (draw)
//     printf("Failed track rotation in layer %i\n", layer);
//     // return false;
//   }

//   // if (draw)
//   // printf("Track has alpha of %f in layer %i. X(%f), Y(%f), Z(%f). Eta(%f), Pt(%f)\n", track.getAlpha(), layer, track.getX(), track.getY(), track.getZ(), track.getEta(), track.getPt());

//   return ChamberSpacePoint(track, mGeo);

//   // pos.x = track.getX();
//   // pos.y = track.getY();
//   // pos.z = track.getZ();

//   // }
//   //   ++nPointsTrigger;
//   //   trackMap.insert(std::make_pair(std::make_tuple(geo->getDetector(layer, stack, sector), row, col), iTrack));
//   //   int rowGlb = stack < 3 ? row + stack * 16 : row + 44 + (stack - 3) * 16; // pad row within whole sector
//   //   int colGlb = col + sector * 144;                                         // pad column number from 0 to NSectors * 144
//   //   hTracks[layer]->SetBinContent(rowGlb + 1, colGlb + 1, 4);
//   // }

//   // return true;
// }

// ========================================================================
// ========================================================================
//
// the DataManager class
//
// ========================================================================
// ========================================================================



DataManager::DataManager(std::filesystem::path dir)
// rawdisp::DataManager::DataManager(std::string dir)
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
void DataManager::AddReaderArray(TTreeReaderArray<T> *&array, std::filesystem::path file, std::string tree, std::string branch)
{
  if (!std::filesystem::exists(file)) {
    // return value TRUE indicates file was not found 
    return;
  }

  // the file exists, everything else should work
  mDatatree->AddFriend(tree.c_str(), file.c_str());
  array = new TTreeReaderArray<T>(*mDatareader, branch.c_str());
}

bool DataManager::NextTimeFrame()
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

bool DataManager::NextEvent()
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

RawEvent DataManager::GetEvent()
{
  RawEvent ev;
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

o2::dataformats::TFIDInfo DataManager::GetTimeFrameInfo()
{
  if (mTFIDs) {
    return mTFIDs->at(mTimeFrameNo-1);
  } else {
    return o2::dataformats::TFIDInfo();
  }
}

float DataManager::GetTriggerTime()
{
  auto tfid = GetTimeFrameInfo();

  if (tfid.isDummy()) {
    return mTriggerRecord.getBCData().bc2ns() * 1e-3;
  } else {
    o2::InteractionRecord intrec = {0, tfid.firstTForbit};
    return mTriggerRecord.getBCData().differenceInBCMUS(intrec);
  }
}

std::string DataManager::DescribeFiles()
{
  std::ostringstream out;
  if (!mMainfile) {
    out << "DataManager is not connected to any files" << std::flush;
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

std::string DataManager::DescribeTimeFrame()
{
  std::ostringstream out;
  out << "## Time frame " << mTimeFrameNo << ": ";
  // out << mDatareader->GetEntries() << "";      
  return out.str();
}

std::string DataManager::DescribeEvent()
{
  std::ostringstream out;
  out << "## TF:Event " << mTimeFrameNo << ":" << mEventNo << ":  "
      //  << hits->GetSize() << " hits   "
      << mTriggerRecord.getNumberOfDigits() << " digits and "
      << mTriggerRecord.getNumberOfTracklets() << " tracklets";
  return out.str();
}


// void DataManager::ConnectMCHitsFile(std::string fname)
// {
//   // ----------------------------------------------------------------------
//   // set up data structures for reading

//   if (fhits || trhits) {
//     cerr << "Hits file seems to be connected." << std::endl;
//     return;
//   }

//   fhits = new TFile(fname.c_str());
//   fhits->GetObject("o2sim", trhits);

//   rdrhits = new TTreeReader(trhits);
//   hits = new TTreeReaderArray<o2::trd::Hit>(*rdrhits, "TRDHit");
// }


// ========================================================================
// ========================================================================
//
// Drawing routines
//
// ========================================================================
// ========================================================================

// TVirtualPad *DrawPadRow(RawDataSpan &padrow, TVirtualPad *pad = NULL, TH2F *adcmap = NULL)
// {
//   // auto x = *padrow.digits.begin();
//   // string desc = fmt::format("{:m}", x);
//   string name = fmt::format("det{:03d}_row{:d}",
//                             padrow.getDetector(), padrow.getPadRow());
//   string desc = name;

//   std::cout << "Plotting padrow " << name << std::endl;
//   if (pad == NULL) {
//     pad = new TCanvas(desc.c_str(), desc.c_str(), 1200, 500);
//     pad->cd();
//   }

//   if (adcmap == NULL) {
//     adcmap = new TH2F(name.c_str(), (desc + ";pad;time bin").c_str(), 144, 0., 144., 30, 0., 30.);
//   }

//   for (auto digit : padrow.digits) {
//     if (digit.isSharedDigit()) { continue; }

//     auto adc = digit.getADC();
//     for (int tb = 0; tb < 30; ++tb) {
//       adcmap->Fill(digit.getPadCol(), tb, adc[tb]);
//     }
//   }
//   adcmap->SetStats(0);
//   adcmap->Draw("colz");
//   adcmap->Draw("text,same");

//   TLine trkl;
//   trkl.SetLineWidth(2);
//   trkl.SetLineColor(kRed);
//   // trkl.SetLineStyle(kDotted);

//   TLine trkl2;
//   trkl2.SetLineWidth(4);
//   trkl2.SetLineStyle(kDashed);
//   trkl2.SetLineColor(kBlack);

//   for (auto tracklet : padrow.tracklets)
//   {
//     auto pos = PadPosition(tracklet);
//     auto ypos = UncalibratedPad(tracklet);
//     auto slope = Slope(tracklet);
//     trkl.DrawLine(pos, 0, pos - 30*slope, 30);
//     // trkl2.DrawLine(ypos, 0, ypos - 30*slope, 30);
//   }

//   TMarker mrk;
//   mrk.SetMarkerStyle(20);
//   mrk.SetMarkerSize(8);
//   mrk.SetMarkerColor(kGreen);
//   for (auto point : padrow.trackpoints)
//   {
//     mrk.DrawMarker(point.getPadCol(), 0);
//     std::cout << point.getDetector() << " / "
//          << point.getPadRow() << " / "
//          << point.getPadCol() << std::endl;
//     // auto pos = PadPosition(tracklet);
//     // auto ypos = UncalibratedPad(tracklet);
//     // auto slope = Slope(tracklet);
//     // trkl.DrawLine(pos, 0, pos - 30*slope, 30);
//     // trkl2.DrawLine(ypos, 0, ypos - 30*slope, 30);
//   }

//   return pad;
// }

// TPad *DrawMCM(RawDataSpan &mcm, TPad *pad)
// {
//   auto x = *mcm.digits.begin();
//   string desc = fmt::format("{:m}", x);
//   string name = fmt::format("det{:03d}_rob{:d}_mcm{:02d}",
//                             x.getDetector(), x.getROB(), x.getMCM());

//   if (pad == NULL)
//   {
//     pad = new TCanvas(desc.c_str(), desc.c_str(), 800, 600);
//   }
//   else
//   {
//     pad->SetName(name.c_str());
//     pad->SetTitle(desc.c_str());
//   }
//   pad->cd();

//   TH2F *digit_disp = new TH2F(desc.c_str(), (desc + ";ADC channel;time bin").c_str(), 21, 0., 21., 30, 0., 30.);

//   for (auto digit : mcm.digits)
//   {
//     auto adc = digit.getADC();
//     for (int tb = 0; tb < 30; ++tb)
//     {
//       digit_disp->Fill(digit.getChannel(), tb, adc[tb]);
//     }
//   }
//   digit_disp->SetStats(0);
//   digit_disp->Draw("colz");

//   TLine trkl;
//   trkl.SetLineColor(kRed);
//   trkl.SetLineWidth(3);

//   for (auto tracklet : mcm.tracklets)
//   {
//     auto pos = PadPositionMCM(tracklet);
//     auto slope = Slope(tracklet);
//     trkl.DrawLine(pos, 0, pos + 30 * slope, 30);
//   }

//   return pad;
// }

// void DataManager(std::string dir=".")
// {
//   class DataManager dataman(dir);
//   // std::cout << dataman << std::endl;
// }

TPad* rawdisp::DrawMCM(RawDataSpan &mcm, TPad *pad)
{
  auto x = *mcm.digits.begin();

  int det = x.getDetector();
  std::string name = Form("det%03d_rob%d_mcm%02d", det, x.getROB(), x.getMCM());
  std::string desc = Form("Detector %02d_%d_%d (%03d) - MCM %d:%02d", det/30, (det%30)/6, det%6, det, x.getROB(), x.getMCM());;

   // MCM column number on ROC [0..7]
  int mcmcol = x.getMCM() % constants::NMCMROBINCOL + HelperMethods::getROBSide(x.getROB()) * constants::NMCMROBINCOL;

  float firstpad = mcmcol * constants::NCOLMCM - 1;
  float lastpad = (mcmcol+1) * constants::NCOLMCM + 2;

  if (pad == NULL) {
    pad = new TCanvas(name.c_str(), desc.c_str(), 800, 600);
    std::cout << "create canvas " << desc << std::endl;
  } else {
    pad->SetName(name.c_str());
    pad->SetTitle(desc.c_str());
  }
  pad->cd();

  std::cout << firstpad << " - " << lastpad << std::endl;
  TH2F *digit_disp = new TH2F(name.c_str(), (desc + ";pad;time bin").c_str(), 21, firstpad, lastpad, 30, 0., 30.);

  for (auto digit : mcm.digits) {
    auto adc = digit.getADC();
    for (int tb = 0; tb < 30; ++tb) {
      digit_disp->Fill(digit.getPadCol(), tb, adc[tb]);
    }
  }
  digit_disp->SetStats(0);
  digit_disp->Draw("colz");

  TLine trkl;
  trkl.SetLineColor(kRed);
  trkl.SetLineWidth(3);

  for (auto tracklet : mcm.tracklets) {
    auto pos = PadColF(tracklet);
    auto slope = SlopeF(tracklet);
    trkl.DrawLine(pos, 0, pos + 30 * slope, 30);
  }

  TMarker clustermarker;
  clustermarker.SetMarkerColor(kRed);
  clustermarker.SetMarkerStyle(2);
  clustermarker.SetMarkerSize(1.5);

  TMarker cogmarker;
  cogmarker.SetMarkerColor(kGreen);
  cogmarker.SetMarkerStyle(3);
  cogmarker.SetMarkerSize(1.5);
  for(int t=1; t<=digit_disp->GetNbinsY(); ++t) {
    for(int p=2; p<=digit_disp->GetNbinsX()-1; ++p) {
      // cout << p << "/" << t << " -> " << digit_disp->GetBinContent(i,j) << endl;
        double baseline = 9.5;
        double left = digit_disp->GetBinContent(p-1,t) - baseline;
        double centre = digit_disp->GetBinContent(p,t) - baseline;
        double right = digit_disp->GetBinContent(p+1,t) - baseline;
        if (centre > left && centre > right) {
          double pos = 0.5 * log(right/left) / log(centre*centre / left / right);
          double clpos = digit_disp->GetXaxis()->GetBinCenter(p) + pos;
          double cog = (right - left) / (right+centre+left) + digit_disp->GetXaxis()->GetBinCenter(p);
          // cout << "t=" << t << " p=" << p 
          //      << ":   ADCs = " << left << " / " << centre << " / " << right
          //      << "   pos = " << pos << " ~ " << clpos
          //      << endl;
          clustermarker.DrawMarker(clpos, t-0.5);
          cogmarker.DrawMarker(cog, t-0.5);
        }
    }
  }


  // TODO: At the moment, hits are not propagated during splitting of a RawDataSpan, therefore this code does not work yet.
  // TMarker hitmarker;
  // hitmarker.SetMarkerColor(kBlack);
  // hitmarker.SetMarkerStyle(38);

  // auto ct = CoordinateTransformer::instance();
  // for (auto hit : mcm.hits) {
  //   std::cout << hit.GetCharge() << std::endl;
  //   auto rct = ct->Local2RCT(hit);
  //   hitmarker.SetMarkerSize(hit.GetCharge() / 50.);
  //   hitmarker.DrawMarker(rct[1], rct[2]);
  // }

  return pad;
}

rawdisp::MCMDisplay::MCMDisplay(RawDataSpan &mcmdata, TVirtualPad *pad)
: mDataSpan(mcmdata)
{
  int det=-1,rob=-1,mcm=-1;

  if(std::distance(mDataSpan.digits.begin(), mDataSpan.digits.end())) {
    auto x = *mDataSpan.digits.begin();
    det = x.getDetector();
    rob = x.getROB();
    mcm = x.getMCM();
  } else if(std::distance(mDataSpan.tracklets.begin(), mDataSpan.tracklets.end())) {
    auto x = *mDataSpan.tracklets.begin();
    det = x.getDetector();
    rob = x.getROB();
    mcm = x.getMCM();
  } else {
    std::cerr << "ERROR: found neither digits nor tracklets in MCM" << std::endl;
    assert(false);
  }

  mName = Form("det%03d_rob%d_mcm%02d", det, rob, mcm);
  mDesc = Form("Detector %02d_%d_%d (%03d) - MCM %d:%02d", det/30, (det%30)/6, det%6, det, rob, mcm);;

   // MCM column number on ROC [0..7]
  int mcmcol = mcm % constants::NMCMROBINCOL + HelperMethods::getROBSide(rob) * constants::NMCMROBINCOL;

  mFirstPad = mcmcol * constants::NCOLMCM - 1;
  mLastPad = (mcmcol+1) * constants::NCOLMCM + 2;

  if (pad == NULL) {
    mPad = new TCanvas(mName.c_str(), mDesc.c_str(), 800, 600);
    std::cout << "create canvas " << mDesc << std::endl;
  } else {
    mPad = pad;
    mPad->SetName(mName.c_str());
    mPad->SetTitle(mDesc.c_str());
  }
}

void rawdisp::MCMDisplay::DrawDigits()
{
  mPad->cd();

  std::cout << mFirstPad << " - " << mLastPad << std::endl;
  if (mDigitsHisto) {
    delete mDigitsHisto;
  }
  mDigitsHisto = new TH2F(mName.c_str(), (mDesc + ";pad;time bin").c_str(), (mLastPad-mFirstPad), mFirstPad, mLastPad, 30, 0., 30.);

  for (auto digit : mDataSpan.digits) {
    auto adc = digit.getADC();
    for (int tb = 0; tb < 30; ++tb) {
      mDigitsHisto->Fill(digit.getPadCol(), tb, adc[tb]);
    }
  }
  mDigitsHisto->SetStats(0);
  mDigitsHisto->Draw("colz");
}

void rawdisp::MCMDisplay::DrawTracklets()
{
  mPad->cd();

  TLine trkl;
  trkl.SetLineColor(kRed);
  trkl.SetLineWidth(3);

  for (auto tracklet : mDataSpan.tracklets) {
    auto pos = PadColF(tracklet);
    auto slope = SlopeF(tracklet);
    trkl.DrawLine(pos, 0, pos + 30 * slope, 30);
  }

}

void rawdisp::MCMDisplay::DrawClusters()
{
  mPad->cd();

  if(!mDigitsHisto) {
    DrawDigits();
  }

  TMarker clustermarker;
  clustermarker.SetMarkerColor(kRed);
  clustermarker.SetMarkerStyle(2);
  clustermarker.SetMarkerSize(1.5);

  TMarker cogmarker;
  cogmarker.SetMarkerColor(kGreen);
  cogmarker.SetMarkerStyle(3);
  cogmarker.SetMarkerSize(1.5);
  for(int t=1; t<=mDigitsHisto->GetNbinsY(); ++t) {
    for(int p=2; p<=mDigitsHisto->GetNbinsX()-1; ++p) {
      // cout << p << "/" << t << " -> " << mDigitsHisto->GetBinContent(i,j) << endl;
        double baseline = 9.5;
        double left = mDigitsHisto->GetBinContent(p-1,t) - baseline;
        double centre = mDigitsHisto->GetBinContent(p,t) - baseline;
        double right = mDigitsHisto->GetBinContent(p+1,t) - baseline;
        if (centre > left && centre > right) {
          double pos = 0.5 * log(right/left) / log(centre*centre / left / right);
          double clpos = mDigitsHisto->GetXaxis()->GetBinCenter(p) + pos;
          double cog = (right - left) / (right+centre+left) + mDigitsHisto->GetXaxis()->GetBinCenter(p);
          // cout << "t=" << t << " p=" << p 
          //      << ":   ADCs = " << left << " / " << centre << " / " << right
          //      << "   pos = " << pos << " ~ " << clpos
          //      << endl;
          clustermarker.DrawMarker(clpos, t-0.5);
          cogmarker.DrawMarker(cog, t-0.5);
        }
    }
  }


  // TODO: At the moment, hits are not propagated during splitting of a RawDataSpan, therefore this code does not work yet.
  // TMarker hitmarker;
  // hitmarker.SetMarkerColor(kBlack);
  // hitmarker.SetMarkerStyle(38);

  // auto ct = CoordinateTransformer::instance();
  // for (auto hit : mcm.hits) {
  //   std::cout << hit.GetCharge() << std::endl;
  //   auto rct = ct->Local2RCT(hit);
  //   hitmarker.SetMarkerSize(hit.GetCharge() / 50.);
  //   hitmarker.DrawMarker(rct[1], rct[2]);
  // }

}

