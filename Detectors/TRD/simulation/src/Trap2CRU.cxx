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

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//  TRD Trap2CRU class                                                       //
//  Class to take the trap output that arrives at the cru and produce        //
//  the cru output. A data mapping more than a cru simulator                 //
///////////////////////////////////////////////////////////////////////////////

#include <string>

#include "Headers/RAWDataHeader.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsCTP/TriggerOffsetsParam.h"
#include "DetectorsRaw/HBFUtils.h"
#include "CCDB/BasicCCDBManager.h"
#include "DetectorsRaw/RawFileWriter.h"
#include "TRDSimulation/Trap2CRU.h"
#include "CommonUtils/StringUtils.h"
#include "TFile.h"
#include "TTree.h"
#include <TStopwatch.h>
#include <cassert>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <array>
#include <vector>
#include <typeinfo>
#include "fairlogger/Logger.h"

using namespace o2::raw;

namespace o2
{
namespace trd
{

struct TRDCRUMapping {
  int32_t flpid;       // hostname of flp
  int32_t cruHWID = 0; // cru ID taken from ecs
  int32_t HCID = 0;    // hcid of first link
};

//this should probably come from ccdb or some authoritive source.
//I doubt this is going to change very often, but ... famous last words.
//
const TRDCRUMapping trdHWMap[constants::NHALFCRU / 2] =
  {
    {166, 250, 0},
    {166, 583, 0},
    {166, 585, 0},
    {167, 248, 0},
    {167, 249, 0},
    {167, 596, 0},
    {168, 246, 0},
    {168, 247, 0},
    {168, 594, 0},
    {169, 252, 0},
    {169, 253, 0},
    {169, 254, 0},
    {170, 245, 0},
    {170, 593, 0},
    {170, 595, 0},
    {171, 258, 0},
    {171, 259, 0},
    {171, 260, 0},
    {172, 579, 0},
    {172, 581, 0},
    {172, 586, 0},
    {173, 578, 0},
    {173, 580, 0},
    {173, 597, 0},
    {174, 256, 0},
    {174, 582, 0},
    {174, 587, 0},
    {175, 251, 0},
    {175, 255, 0},
    {175, 588, 0},
    {176, 264, 0},
    {176, 591, 0},
    {176, 592, 0},
    {177, 263, 0},
    {177, 589, 0},
    {177, 590, 0}};

Trap2CRU::Trap2CRU(const std::string& outputDir, const std::string& inputdigitsfilename, const std::string& inputtrackletsfilename)
{
  mOutputDir = outputDir;
  mInputDigitsFileName = inputdigitsfilename;
  mInputTrackletsFileName = inputtrackletsfilename;
  mCurrentDigit = 0;
  mCurrentTracklet = 0;
}

void Trap2CRU::openInputFiles()
{
  mDigitsFile = TFile::Open(mInputDigitsFileName.data());
  if (mDigitsFile != nullptr && !mDigitsFile->IsZombie()) {
    mDigitsTree = (TTree*)mDigitsFile->Get("o2sim");
    mDigitsTree->SetBranchAddress("TRDDigit", &mDigitsPtr); // the branch with the actual digits
  } else {
    LOG(warn) << " cant open file containing digit tree";
  }
  mTrackletsFile = TFile::Open(mInputTrackletsFileName.data());
  if (mTrackletsFile != nullptr && !mTrackletsFile->IsZombie()) {
    mTrackletsTree = (TTree*)mTrackletsFile->Get("o2sim");
    mTrackletsTree->SetBranchAddress("Tracklet", &mTrackletsPtr);              // the branch with the actual tracklets.
    mTrackletsTree->SetBranchAddress("TrackTrg", &mTrackletTriggerRecordsPtr); // branch with trigger records for digits
  } else {
    LOG(fatal) << " cant open file containing tracklet tree";
  }
}

void Trap2CRU::sortDataToLinks()
{
  auto sortstart = std::chrono::high_resolution_clock::now();
  //build indexes
  // digits first
  mDigitsIndex.resize(mDigits.size());
  std::iota(mDigitsIndex.begin(), mDigitsIndex.end(), 0);

  for (auto& trig : mTrackletTriggerRecords) {
    if (trig.getNumberOfTracklets() > 0) {
      if (mVerbosity) {
        LOG(debug) << " sorting tracklets from : " << trig.getFirstTracklet() << " till " << trig.getFirstTracklet() + trig.getNumberOfTracklets();
      }
      // sort to link order *NOT* hcid order ...
      // link is defined by stack,layer,halfchamberside.
      // tracklet data we have hcid,padrow,colum.
      // hcid/2 = detector, detector implies stack and layer, and hcid odd/even gives side.
      std::stable_sort(std::begin(mTracklets) + trig.getFirstTracklet(), std::begin(mTracklets) + trig.getNumberOfTracklets() + trig.getFirstTracklet(),
                       [this](auto&& t1, auto&& t2) {
                         int link1 = HelperMethods::getLinkIDfromHCID(t1.getHCID());
                         int link2 = HelperMethods::getLinkIDfromHCID(t2.getHCID());
                         if (link1 != link2) {
                           return link1 < link2;
                         }
                         if (t1.getPadRow() != t2.getPadRow()) {
                           return t1.getPadRow() < t2.getPadRow();
                         }
                         return t1.getMCM() < t2.getMCM();
                       });
    }
    if (trig.getNumberOfDigits() != 0) {
      if (mVerbosity) {
        LOG(debug) << " sorting digits from : " << trig.getFirstDigit() << " till " << trig.getFirstDigit() + trig.getNumberOfDigits();
      }
      std::stable_sort(mDigitsIndex.begin() + trig.getFirstDigit(), mDigitsIndex.begin() + trig.getNumberOfDigits() + trig.getFirstDigit(),
                       [this](const uint32_t i, const uint32_t j) {
             int link1=HelperMethods::getLinkIDfromHCID(mDigits[i].getHCId());
             int link2=HelperMethods::getLinkIDfromHCID(mDigits[j].getHCId());
             if(link1!=link2){return link1<link2;}
             if(mDigits[i].getROB() != mDigits[j].getROB()){return (mDigits[i].getROB() < mDigits[j].getROB());}
             if(mDigits[i].getMCM() != mDigits[j].getMCM()){return (mDigits[i].getMCM() < mDigits[j].getMCM());}
             return (mDigits[i].getChannel() < mDigits[j].getChannel()); });
    }
  }

  if (mVerbosity) {
    std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - sortstart;
    LOG(info) << "TRD Digit/Tracklet Sorting took " << duration.count() << " s";
    int triggercount = 0;
    for (auto& trig : mTrackletTriggerRecords) {

      LOG(info) << "Trigger: " << triggercount << " with T " << trig.getBCData().asString();
      LOG(info) << "Tracklets from:" << trig.getFirstTracklet() << " with " << trig.getNumberOfTracklets();
      LOG(info) << "Digits from:" << trig.getFirstDigit() << " with " << trig.getNumberOfDigits();
      if (trig.getNumberOfTracklets() > 0) {
        int firsttracklet = trig.getFirstTracklet();
        int numtracklets = trig.getNumberOfTracklets();
        for (int trackletcount = firsttracklet; trackletcount < firsttracklet + numtracklets; ++trackletcount) {
          LOG(info) << "Tracklet : " << trackletcount << " details : supermodule:" << std::dec << mTracklets[trackletcount].getHCID() << std::hex << " tracklet:" << mTracklets[trackletcount];
        }
      } else {
        LOG(info) << "No Tracklets for this trigger";
      }
      if (trig.getNumberOfDigits() != 0) {
        int firstdigit = trig.getFirstDigit();
        int numdigits = trig.getNumberOfDigits();
        for (int digitcount = firstdigit; digitcount < firstdigit + numdigits; ++digitcount) {
          LOG(info) << "Digit indexed: " << digitcount << " digit index : " << mDigitsIndex[digitcount] << " details : hcid=" << mDigits[mDigitsIndex[digitcount]].getHCId()
                    << " calculated hcid=" << (mDigits[mDigitsIndex[digitcount]].getDetector() * 2 + (mDigits[mDigitsIndex[digitcount]].getROB() % 2))
                    << " det=" << mDigits[mDigitsIndex[digitcount]].getDetector()
                    << " mcm=" << mDigits[mDigitsIndex[digitcount]].getMCM()
                    << " rob=" << mDigits[mDigitsIndex[digitcount]].getROB()
                    << " channel=" << mDigits[mDigitsIndex[digitcount]].getChannel()
                    << " col=" << mDigits[mDigitsIndex[digitcount]].getPadRow()
                    << " pad=" << mDigits[mDigitsIndex[digitcount]].getPadCol()
                    << " adcsum=" << mDigits[mDigitsIndex[digitcount]].getADCsum()
                    << " hcid=" << mDigits[mDigitsIndex[digitcount]].getHCId();
        }

      } else {
        LOG(error) << "No Digits for this trigger <----- this should NEVER EVER HAPPEN";
      }
      triggercount++;
    }
    LOG(info) << "end of  pre sort tracklets then digits";
  } // if verbose
}

void Trap2CRU::readTrapData()
{
  // set things up, read the file and then deligate to convertTrapdata to do the conversion.
  //
  if (mVerbosity) {
    LOG(info) << "Trap2CRU::readTrapData";
  }
  // data comes in index by event (triggerrecord)
  // first 15 links go to cru0a, second 15 links go to cru0b, 3rd 15 links go to cru1a ... first 90 links to flp0 and then repeat for 12 flp
  // then do next event

  // request the mapping from CCDB, if not yet available
  if (!mLinkMap) {
    LOG(info) << "Retrieving LinkToHCIDMapping for time stamp " << mTimeStamp;
    auto& ccdbmgr = o2::ccdb::BasicCCDBManager::instance();
    mLinkMap = ccdbmgr.getForTimeStamp<LinkToHCIDMapping>("TRD/Config/LinkToHCIDMapping", mTimeStamp);
  }

  // lets register our links
  std::string prefix = mOutputDir;
  if (!prefix.empty() && prefix.back() != '/') {
    prefix += '/';
  }

  for (int link = 0; link < constants::NHALFCRU; link++) {
    // FeeID *was* 0xFEED, now is indicates the cru Supermodule, side (A/C) and endpoint. See RawData.cxx for details.
    int supermodule = link / 4;
    int endpoint = link % 2;
    int cru = link / 2;
    int side = cru % 2; // A or C, 0 or 1 respectively:
    mFeeID = constructTRDFeeID(supermodule, side, endpoint);
    LOG(info) << "FEEID;" << std::hex << mFeeID;
    mCruID = link / 2;
    mEndPointID = endpoint;

    std::string outFileLink;
    std::string outPrefix = "TRD_";
    outPrefix += "alio2-cr1-flp";
    std::string outSuffix = ".raw";
    // filename structure of trd_cru_[CRU#]_[upper/lower].raw
    auto flpid = trdHWMap[mCruID].flpid;
    auto cruhwid = trdHWMap[mCruID].cruHWID;
    if (mFilePer == "all") {
      // single file for all links
      outFileLink = o2::utils::Str::concat_string(mOutputDir, "/", outPrefix, outSuffix);
    } else if (mFilePer == "sm") {
      // one file per supermodule
      int sm = link / 4;
      std::stringstream ss;
      ss << std::setw(2) << std::setfill('0') << sm;
      std::string supermodule = ss.str();
      outFileLink = o2::utils::Str::concat_string(mOutputDir, "/", outPrefix, "_sm_", supermodule, outSuffix);
    } else if (mFilePer == "fullcru") {
      // one file per CRU (both end points combined)
      outFileLink = o2::utils::Str::concat_string(mOutputDir, "/", outPrefix, std::to_string(flpid), "_cru", std::to_string(cruhwid), outSuffix);
    } else if (mFilePer == "cru") {
      // one file per CRU end point
      outFileLink = o2::utils::Str::concat_string(mOutputDir, "/", outPrefix, std::to_string(flpid), "_cru", std::to_string(cruhwid), "_", std::to_string(mEndPointID), outSuffix);
    } else {
      throw std::runtime_error("invalid option provided for file grouping");
    }
    LOG(info) << "registering links";
    mWriter.registerLink(mFeeID, mCruID, mLinkID, mEndPointID, outFileLink);
  }

  openInputFiles();

  if (mTrackletsTree->GetEntries() != mDigitsTree->GetEntries()) {
    LOG(fatal) << "Entry counts in mTrackletsTree and Digits Tree dont match " << mTrackletsTree->GetEntries() << "!=" << mDigitsTree->GetEntries();
  }
  int nTrackletsTotal = 0;
  int nDigitsTotal = 0;
  int nTriggerRecordsTotal = 0;
  int triggercount = 42; // triggercount is here so that we can span timeframes. The actual number is of no consequence,but must increase.
  for (int entry = 0; entry < mTrackletsTree->GetEntries(); entry++) {
    mTrackletsTree->GetEntry(entry);
    mDigitsTree->GetEntry(entry);
    nTrackletsTotal += mTracklets.size();
    nDigitsTotal += mDigits.size();
    nTriggerRecordsTotal += mTrackletTriggerRecords.size();
    sortDataToLinks();
    // each entry is a timeframe
    for (auto tracklettrigger : mTrackletTriggerRecords) {
      convertTrapData(tracklettrigger, triggercount); // tracklettrigger assumed to be authoritive
      triggercount++;
    }
  }
  LOGF(info, "In the input files there were %u tracklets and %u digits in %u trigger records", nTrackletsTotal, nDigitsTotal, nTriggerRecordsTotal);
  LOGF(info, "Wrote %lu tracklets and %lu digits into the raw data", mTotalTrackletsWritten, mTotalDigitsWritten);
}

uint32_t Trap2CRU::buildHalfCRUHeader(HalfCRUHeader& header, const uint32_t bc, const uint32_t halfcru, bool isCalibTrigger)
{
  int bunchcrossing = bc;
  int stopbits = 0x01; // do we care about this and eventtype in simulations?
  int eventtype = isCalibTrigger ? constants::ETYPECALIBRATIONTRIGGER : constants::ETYPEPHYSICSTRIGGER;
  int crurdhversion = 6;
  int feeid = 0;
  int cruid = 0;
  int endpoint = halfcru % 2 ? 1 : 0;
  //lets first clear it out.
  clearHalfCRUHeader(header);
  //this bunchcrossing is not the same as the bunchcrossing in the rdh, which is the bc coming in the parameter list to this function. See explanation in rawdata.h
  setHalfCRUHeaderFirstWord(header, crurdhversion, bunchcrossing, stopbits, endpoint, eventtype, feeid, cruid);

  return 1;
}

int Trap2CRU::buildDigitRawData(const int digitstartindex, const int digitendindex, const int mcm, const int rob, const uint32_t triggerrecordcount)
{
  //this is not zero suppressed.
  int digitwordswritten = 0;
  int digitswritten = 0;
  //    Digit
  DigitMCMHeader header;
  DigitMCMADCMask adcmask;
  DigitMCMData data;
  header.res = 0xc; //1100
  header.mcm = mcm;
  header.rob = rob;
  header.yearflag = 1; // >10.2007
  header.eventcount = triggerrecordcount;
  memcpy(mRawDataPtr, (char*)&header, sizeof(DigitMCMHeader)); // uint32 -- 4 bytes.
  // DigitMCMHeader* headerptr = (DigitMCMHeader*)mRawDataPtr;
  // LOG(info) << "Digt Header word: 0x" << std::hex << headerptr->word;
  mRawDataPtr += 4;
  digitwordswritten++;
  // we are writing zero suppressed so we need adcmask
  adcmask = constructBlankADCMask();
  memcpy(mRawDataPtr, (char*)&adcmask, sizeof(DigitMCMADCMask));
  DigitMCMADCMask* adcmaskptr = (DigitMCMADCMask*)mRawDataPtr;
  mRawDataPtr += 4;
  digitwordswritten++;
  //LOG(info) << "writing data to digit stream of " << std::hex << header.word;
  for (int digitindex = digitstartindex; digitindex < digitendindex; ++digitindex) {
    Digit* d = &mDigits[mDigitsIndex[digitindex]];
    ArrayADC adcdata = d->getADC();
    int channel = d->getChannel();
    //set adcmask for the channel we currently have.
    incrementADCMask(*adcmaskptr, channel); //adcmaskptr->adcmask |= 1UL << channel;
    for (int timebin = 0; timebin < constants::TIMEBINS; timebin += 3) {
      data.z = adcdata[timebin];
      data.y = adcdata[timebin + 1];
      data.x = adcdata[timebin + 2];
      data.f = (channel % 2 == 0) ? 0x3 : 0x2;                 // 3 for even channel 2 for odd channel
      memcpy(mRawDataPtr, (char*)&data, sizeof(DigitMCMData)); // uint32 -- 4 bytes.
      mRawDataPtr += sizeof(DigitMCMData);
      digitwordswritten++;
    }
    if (mVerbosity) {
      LOG(info) << "DDDD " << d->getDetector() << ":" << d->getROB() << ":" << d->getMCM() << ":" << d->getChannel() << ":" << d->getADCsum() << ":" << d->getADC()[0] << ":" << d->getADC()[1] << ":" << d->getADC()[2] << "::" << d->getADC()[27] << ":" << d->getADC()[28] << ":" << d->getADC()[29];
    }
    digitswritten++;
  }
  // sanityCheckDigitMCMADCMask(*adcmaskptr, digitswritten);
  if (digitswritten != digitendindex - digitstartindex) {
    LOG(error) << " something wrong the number of digitswritten does not correspond to the the loop count";
  }
  mTotalDigitsWritten += digitswritten;
  if (digitwordswritten != (digitswritten * 10 + 2)) {
    LOG(error) << "something wrong with writing the digits the following should be equal " << digitwordswritten << "==" << (digitswritten * 10 + 2) << " with digitswritten=" << digitswritten;
    LOG(error) << "digit start index distance to digit end index :" << digitendindex - digitstartindex;
  }
  return digitwordswritten;
}

int Trap2CRU::buildTrackletRawData(unsigned int trackletIndexStart)
{
  int hcid = mTracklets[trackletIndexStart].getHCID();
  TrackletMCMHeader header;                 // header with common tracklet information and upper 8 bit of PID information for each tracklet
  std::array<TrackletMCMData, 3> tracklets; // the up to three tracklet words

  header.col = mTracklets[trackletIndexStart].getColumn();
  header.padrow = mTracklets[trackletIndexStart].getPadRow();
  header.onea = 1;
  header.oneb = 1;
  header.pid0 = 0xff;
  header.pid1 = 0xff;
  header.pid2 = 0xff;
  int iCurrTracklet = 0;

  while (hcid == mTracklets[trackletIndexStart + iCurrTracklet].getHCID() &&
         header.col == mTracklets[trackletIndexStart + iCurrTracklet].getColumn() &&
         header.padrow == mTracklets[trackletIndexStart + iCurrTracklet].getPadRow()) { // we are still on the same MCM
    unsigned int trackletIndex = trackletIndexStart + iCurrTracklet;
    auto& trackletData = tracklets[iCurrTracklet];
    trackletData.word = 0;
    // slope and position have the 8-th bit flipped each
    trackletData.slope = mTracklets[trackletIndex].getSlope() ^ 0x80;
    trackletData.pos = mTracklets[trackletIndex].getPosition() ^ 0x80;
    trackletData.checkbit = 0;
    int pidHeader = 0;
    bool qDynamicRange = mTracklets[trackletIndex].getFormat() & 0x1;
    if (qDynamicRange) {
      // Dynamic charge range, the tracklet PID contains all 6 bits of q0 and q1
      // TODO add scaling factor
      LOG(warning) << "Trying to add PID information for dynamic charge range, which is not yet verified";
      trackletData.pid = ((mTracklets[trackletIndex].getQ1() & 0x3f) << 6) | (mTracklets[trackletIndex].getQ0() & 0x3f);
      pidHeader = mTracklets[trackletIndex].getQ2() & 0x3f;
    } else {
      // Fixed charge range, the tracklet PID contains all 7 bits of q0 and 5 out of 7 bits for q1
      trackletData.pid = ((mTracklets[trackletIndex].getQ1() & 0x1f) << 7) | (mTracklets[trackletIndex].getQ0() & 0x7f);
      pidHeader = ((mTracklets[trackletIndex].getQ2() & 0x3f) << 2) | ((mTracklets[trackletIndex].getQ1() >> 5) & 0x3);
    }
    if (iCurrTracklet == 0) {
      header.pid0 = pidHeader;
    } else if (iCurrTracklet == 1) {
      header.pid1 = pidHeader;
      if (header.pid0 == 0xff) {
        LOG(error) << "Adding PID info for second tracklet, but first is marked as not available";
      }
    } else if (iCurrTracklet == 2) {
      header.pid2 = pidHeader;
      if (header.pid1 == 0xff || header.pid0 == 0xff) {
        LOG(error) << "Adding PID info for third tracklet, but first or second is marked as not available";
      }
    } else {
      LOG(fatal) << "Cannot have more than 3 tracklets for single trigger and single MCM";
    }
    iCurrTracklet++;
    if (trackletIndexStart + iCurrTracklet >= mTracklets.size()) {
      break;
    }
  }
  // MCM header and MCM data are assembled, write it now
  if (iCurrTracklet == 0) {
    LOG(fatal) << "Not writing any tracklet. This cannot happen, there must be at least one or this function would not be called";
  }
  if (mVerbosity) {
    printTrackletMCMHeader(header);
  }
  memcpy((char*)mRawDataPtr, (char*)&header, sizeof(TrackletMCMHeader));
  mRawDataPtr += sizeof(TrackletMCMHeader);
  for (int i = 0; i < iCurrTracklet; ++i) {
    if (mVerbosity) {
      printTrackletMCMData(tracklets[i]);
    }
    memcpy((char*)mRawDataPtr, (char*)&tracklets[i], sizeof(TrackletMCMData));
    mRawDataPtr += sizeof(TrackletMCMData);
  }
  return iCurrTracklet;
}

void Trap2CRU::writeDigitEndMarkers()
{
  // append 0x00000000 0x00000000
  uint32_t digitendmarker = 0;
  memcpy(mRawDataPtr, (char*)&digitendmarker, 4);
  mRawDataPtr += 4;
  memcpy(mRawDataPtr, (char*)&digitendmarker, 4);
  mRawDataPtr += 4;
}

void Trap2CRU::writeTrackletEndMarkers()
{
  // append 0x10001000 0x10001000
  uint32_t trackletendmarker = constants::TRACKLETENDMARKER;
  memcpy(mRawDataPtr, (char*)&trackletendmarker, 4);
  mRawDataPtr += 4;
  memcpy(mRawDataPtr, (char*)&trackletendmarker, 4);
  mRawDataPtr += 4;
}

void Trap2CRU::writeTrackletHCHeader(int hcid, int eventcount)
{
  // from linkid we can get supermodule, stack, layer, side
  unsigned int chipclock = eventcount * 42; // just has to be a constant increasing number per event for our purposes in sim to raw.
  unsigned int format = 12;
  TrackletHCHeader tracklethcheader;
  constructTrackletHCHeader(tracklethcheader, hcid, chipclock, format);
  if (mVerbosity) {
    printTrackletHCHeader(tracklethcheader);
  }
  memcpy(mRawDataPtr, (char*)&tracklethcheader, sizeof(TrackletHCHeader));
  if (mVerbosity) {
    LOG(info) << "writing tracklethcheader of 0x" << std::hex << tracklethcheader.word;
  }
  mRawDataPtr += 4;
}

void Trap2CRU::writeDigitHCHeaders(const int eventcount, const uint32_t hcId)
{
  // The detector can in theory send up to 8 HCHeaders, but it will always send at least 2.
  // Here, we always only send those two headers
  int detector = hcId / 2;
  DigitHCHeader digitheader;
  DigitHCHeader1 digitheader1;
  digitheader.res = 1;
  digitheader.side = (hcId % 2) ? 1 : 0;
  digitheader.stack = HelperMethods::getStack(detector);
  digitheader.layer = HelperMethods::getLayer(detector);
  digitheader.supermodule = HelperMethods::getSector(detector);
  digitheader.numberHCW = 1; // number of additional words in th header, we are using 2 header words so 1 here.
  digitheader.minor = 42;    // my (shtm) version, not used
  digitheader.major = 0x21;  // zero suppressed and 0x1 to comply with what we see in the raw data
  digitheader.version = 1;   //new version of the header. we only have 1 version
  digitheader1.res = 1;
  digitheader1.ptrigcount = 1;
  digitheader1.ptrigphase = 1;
  digitheader1.bunchcrossing = eventcount; //NB this is not the same as the bunchcrossing the rdh. See RawData.h for explanation
  digitheader1.numtimebins = constants::TIMEBINS;
  memcpy(mRawDataPtr, (char*)&digitheader, sizeof(DigitHCHeader)); // 8 because we are only using the first 2 32bit words of the header, the rest are optional.
  mRawDataPtr += sizeof(DigitHCHeader);
  memcpy(mRawDataPtr, (char*)&digitheader1, sizeof(DigitHCHeader1)); // 8 because we are only using the first 2 32bit words of the header, the rest are optional.
  mRawDataPtr += sizeof(DigitHCHeader1);
}

void Trap2CRU::convertTrapData(o2::trd::TriggerRecord const& triggerrecord, const int& triggercount)
{
  // Create the raw data for this trigger
  // loop over half-CRUs and for each half-CRU we put
  // 1. HalfCRUHeader
  // 2. Tracklet data
  // 3. Two tracklet endmarkers
  // 4. Two digit HC headers (only for calibration events)
  // 5. Digit data (only for calibration events)
  // 6. Two end markers (only for calibration events)

  int rawwords = 0;
  int nLinksWithData = 0;
  char* rawdataptratstart;
  std::vector<char> rawdatavector(1024 * 1024 * 2); // sum of link sizes + padding in units of bytes and some space for the header (512 bytes).
  if (mVerbosity) {
    LOG(info) << "BUNCH CROSSING : " << triggerrecord.getBCData().bc << " with orbit : " << triggerrecord.getBCData().orbit;
  }

  uint64_t endtrackletindex = triggerrecord.getFirstTracklet() + triggerrecord.getNumberOfTracklets();
  uint64_t enddigitindex = triggerrecord.getFirstDigit() + triggerrecord.getNumberOfDigits();
  // with digit downscaling enabled there will be triggers with only tracklets
  bool isCalibTrigger = triggerrecord.getNumberOfDigits() > 0 ? true : false;
  const auto& ctpOffsets = o2::ctp::TriggerOffsetsParam::Instance();
  auto ir = triggerrecord.getBCData();
  ir += ctpOffsets.LM_L0;
  if (ctpOffsets.LM_L0 < 0 && ir.toLong() <= -ctpOffsets.LM_L0) {
    // skip this trigger
    LOG(info) << "Skip writing IR " << triggerrecord.getBCData() << " as after applying LM_L0 shift of " << ctpOffsets.LM_L0 << " bunches the orbit would become negative";
    mCurrentDigit = enddigitindex;
    mCurrentTracklet = endtrackletindex;
    return;
  }
  if (triggerrecord.getNumberOfTracklets() == 0 && triggerrecord.getNumberOfDigits() == 0) {
    LOG(info) << "Skip writing trigger " << triggercount << " as there are neither digits nor tracklets";
    return;
  }

  for (int halfcru = 0; halfcru < constants::NHALFCRU; halfcru++) {
    int halfcruwordswritten = 0;
    int supermodule = halfcru / 4; // 2 cru per supermodule. 72/4, as of writing
    mEndPointID = halfcru % 2;     // 2 pci end points per cru, 15 links each
    //first cru is A second CRU is C , so an flp will be either ACA or CAC A=0 C=1
    int cru = halfcru / 2;
    int side = cru % 2; // first cru is A second is B, 3rd is A etc
    mFeeID = constructTRDFeeID(supermodule, side, mEndPointID);
    mCruID = halfcru / 2;
    mEndPointID = halfcru % 2; // just the upper or lower half of the cru, hence %2 of the the halfcru number.
    // 15 links per half cru or cru end point.
    HalfCRUHeader halfcruheader;
    //now write the cruheader at the head of all the data for this halfcru.
    buildHalfCRUHeader(halfcruheader, ir.bc, halfcru, isCalibTrigger);
    halfcruheader.EndPoint = mEndPointID;
    mRawDataPtr = rawdatavector.data();
    HalfCRUHeader* halfcruheaderptr = (HalfCRUHeader*)mRawDataPtr; // store the ptr to the halfcruheader for later adding the link lengths and possibly simulated errors.
    mRawDataPtr += sizeof(halfcruheader);
    halfcruwordswritten += sizeof(halfcruheader) / 4;
    int totallinklengths = 0;
    rawdataptratstart = mRawDataPtr; // keep track of where we started.
    for (int halfcrulink = 0; halfcrulink < constants::NLINKSPERHALFCRU; halfcrulink++) {
      //links run from 0 to 14, so linkid offset is halfcru*15;
      int linkid = halfcrulink + halfcru * constants::NLINKSPERHALFCRU;
      int hcid = mLinkMap->getHCID(linkid);
      int linkwordswritten = 0; // number of 32 bit words for this link
      int errors = 0;           // put no errors in for now.
      uint32_t crudatasize = 0; // in 256 bit words.
      // loop over tracklets for mcms that match
      int nTrackletsOnLink = 0;
      int nDigitsOnLink = 0;
      bool haveDigitOnLink = false;
      bool haveTrackletOnLink = false;
      if (mCurrentTracklet < mTracklets.size() && mTracklets[mCurrentTracklet].getHCID() == hcid) {
        haveTrackletOnLink = true;
      }
      if (mCurrentDigit < mDigits.size() && mDigits[mDigitsIndex[mCurrentDigit]].getHCId() == hcid) {
        haveDigitOnLink = true;
      }
      if (mVerbosity) {
        LOGF(info, "Link ID(%i), HCID(%i). Tracklets? %i, Digits? %i. Tracklet HCID(%i), mCurrentTracklet(%i), mCurrentDigit(%i)",
             linkid, hcid, haveTrackletOnLink, haveDigitOnLink, mTracklets[mCurrentTracklet].getHCID(), mCurrentTracklet, mCurrentDigit);
      }
      if (haveTrackletOnLink || haveDigitOnLink) {
        nLinksWithData++;
        // we have some data somewhere for this link
        if (mUseTrackletHCHeader > 0) {
          if (haveTrackletOnLink || mUseTrackletHCHeader == 2) {
            // write tracklethcheader if there is tracklet data or if we always want to have tracklethcheader
            // first part of the if statement handles the mUseTrackletHCHeader==1 option
            writeTrackletHCHeader(hcid, triggercount);
            linkwordswritten += 1;
          }
          //else do nothing as we dont want/have tracklethcheader
        }
        while (mCurrentTracklet < endtrackletindex && mTracklets[mCurrentTracklet].getHCID() == hcid) {
          // still on an mcm on this link
          int tracklets = buildTrackletRawData(mCurrentTracklet); // returns # of tracklets for single MCM
          mCurrentTracklet += tracklets;
          nTrackletsOnLink += tracklets;
          mTotalTrackletsWritten += tracklets;
          linkwordswritten += tracklets + 1; // +1 to include the header
        }
        // write 2 tracklet end markers irrespective of there being tracklet data.
        writeTrackletEndMarkers();
        linkwordswritten += 2;

        if (isCalibTrigger) {
          // we write two DigitHCHeaders here
          writeDigitHCHeaders(triggercount, hcid);
          linkwordswritten += 2;
          while (mCurrentDigit < enddigitindex && mDigits[mDigitsIndex[mCurrentDigit]].getHCId() == hcid) {
            // while we are on a single mcm, copy the digits timebins to the array.
            int digitcounter = 0;
            int currentROB = mDigits[mDigitsIndex[mCurrentDigit]].getROB();
            int currentMCM = mDigits[mDigitsIndex[mCurrentDigit]].getMCM();
            int firstDigitMCM = mCurrentDigit;
            while (mDigits[mDigitsIndex[mCurrentDigit]].getMCM() == currentMCM &&
                   mDigits[mDigitsIndex[mCurrentDigit]].getROB() == currentROB &&
                   mDigits[mDigitsIndex[mCurrentDigit]].getHCId() == hcid) {
              mCurrentDigit++;
              digitcounter++;
              if (mCurrentDigit == enddigitindex) {
                break;
              }
            }
            // mcm digits are full, now write it out.
            linkwordswritten += buildDigitRawData(firstDigitMCM, mCurrentDigit, currentMCM, currentROB, triggercount);
            nDigitsOnLink += (mCurrentDigit - firstDigitMCM);
          }

          // write the digit end marker so long as we have any data (digits or tracklets).
          writeDigitEndMarkers();
          linkwordswritten += 2;
        } // end isCalibTrigger

        // pad up to a whole 256 bit word size (paddingsize is number of 32 bit words to pad)
        int paddingsize = (linkwordswritten % 8 == 0) ? 0 : 8 - (linkwordswritten % 8);
        int padword = constants::PADDINGWORD;
        for (int i = 0; i < paddingsize; ++i) {
          memcpy(mRawDataPtr, &padword, 4);
          mRawDataPtr += 4;
          linkwordswritten++;
        }
        rawwords += linkwordswritten;
        crudatasize = linkwordswritten / 8; //convert to 256 bit alignment.
        if ((linkwordswritten % 8) != 0) {
          LOG(error) << "linkwordswritten is not 256 bit aligned: " << linkwordswritten << ". Padding size of= " << paddingsize;
        }
        //set the halfcruheader for the length of this link.
        //but first a sanity check.
        if (crudatasize > constants::MAXDATAPERLINK256) {
          LOG(error) << " linksize is huge : " << crudatasize;
        }
        LOG(debug) << " setting halfcrulink " << halfcrulink << " linksize to : " << crudatasize << " with a linkwordswrittern=" << linkwordswritten;
        setHalfCRUHeaderLinkSizeAndFlags(halfcruheader, halfcrulink, crudatasize, errors);
        uint32_t bytescopied;
        totallinklengths += crudatasize;
        if ((mRawDataPtr - rawdataptratstart) != (totallinklengths * 32)) {
          bytescopied = mRawDataPtr - rawdataptratstart;
          LOGF(error, "Data size missmatch. Written words (%i), bytesCopied(%i), crudatasize(%i)", linkwordswritten, bytescopied, crudatasize);
        }
        LOGF(debug, "Found %i tracklets and %i digits on link %i (HCID=%i)", nTrackletsOnLink, nDigitsOnLink, linkid, hcid);
      } else {
        // no data on this link
        setHalfCRUHeaderLinkSizeAndFlags(halfcruheader, halfcrulink, 0, 0);
        if (mVerbosity) {
          LOG(info) << "linkwordswritten is zero : " << linkwordswritten;
        }
      }
      halfcruwordswritten += linkwordswritten;
    } // end loop over all links for one half CRU
    // write the cruhalfheader now that we know the lengths.
    memcpy((char*)halfcruheaderptr, (char*)&halfcruheader, sizeof(halfcruheader));
    //write halfcru data here.
    std::vector<char> feeidpayload(halfcruwordswritten * 4);
    memcpy(feeidpayload.data(), &rawdatavector[0], halfcruwordswritten * 4);
    assert(halfcruwordswritten % 8 == 0);
    mWriter.addData(mFeeID, mCruID, mLinkID, mEndPointID, ir, feeidpayload, false, triggercount);
    if (mVerbosity) {
      LOGF(info, "Written file for trigger %i, FeeID(%x), CruID(%i), LindID(%i), end point (%i), orbit(%i), bc(%i), payload size (%i)",
           triggercount, mFeeID, mCruID, mLinkID, mEndPointID, ir.orbit, ir.bc, halfcruwordswritten);
      printHalfCRUHeader(halfcruheader);
      LOG(info) << "+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+  ======   end of writing";
    }
  }
  if (mVerbosity) {
    LOG(info) << "Raw data written for all CRUs of this trigger: " << rawwords;
    LOG(info) << "Number of links with data for this trigger: " << nLinksWithData;
  }
}

} // end namespace trd
} // end namespace o2
