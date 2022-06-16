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
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/LinkRecord.h"
#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/Constants.h"
#include "DetectorsRaw/HBFUtils.h"
#include "DetectorsRaw/RawFileWriter.h"
#include "TRDSimulation/Trap2CRU.h"
#include "TRDSimulation/TrapSimulator.h"
#include "CommonUtils/StringUtils.h"
#include "TFile.h"
#include "TTree.h"
#include <TStopwatch.h>
#include <cassert>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <array>
#include <string>
#include <bitset>
#include <vector>
#include <gsl/span>
#include <typeinfo>

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
const TRDCRUMapping trdHWMap[o2::trd::constants::NHALFCRU / 2] =
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
  mSuperPageSizeInB = 1024 * 1024;
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
    mDigitsTree->SetBranchAddress("TRDDigit", &mDigitsPtr);                   // the branch with the actual digits
    mDigitsTree->SetBranchAddress("TriggerRecord", &mDigitTriggerRecordsPtr); // branch with trigger records for digits
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
             int link1=HelperMethods::getLinkIDfromHCID(mDigits[i].getDetector() * 2 + (mDigits[i].getROB() % 2));
             int link2=HelperMethods::getLinkIDfromHCID(mDigits[j].getDetector() * 2 + (mDigits[j].getROB() % 2));
             if(link1!=link2){return link1<link2;}
             if(mDigits[i].getROB() != mDigits[j].getROB()){return (mDigits[i].getROB() < mDigits[j].getROB());}
             if(mDigits[i].getMCM() != mDigits[j].getMCM()){return (mDigits[i].getMCM() < mDigits[j].getMCM());}
             return (mDigits[i].getChannel() < mDigits[j].getChannel()); });
    }
  }

  std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - sortstart;
  if (mVerbosity) {
    LOG(info) << "TRD Digit/Tracklet Sorting took " << duration.count() << " s";
    int triggercount = 0;
    for (auto& trig : mTrackletTriggerRecords) {

      LOG(info) << "Trigger: " << triggercount << " with T" << trig.getBCData().asString();
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

void Trap2CRU::mergetriggerDigitRanges()
{
  // pass through the digit ranges of the incoming tracklet triggers.
  // this most handles the old data.
  // trapsim should now be sending out the trigger with both information.
  bool fixdigitinfo = false;
  for (auto trig : mTrackletTriggerRecords) {
    if (trig.getNumberOfDigits() == 0) {
      fixdigitinfo = true;
    }
  }
  if (fixdigitinfo) {
    int counter = 0;
    for (auto trig : mTrackletTriggerRecords) {
      trig.setDigitRange(mDigitTriggerRecords[counter].getFirstDigit(), mDigitTriggerRecords[counter].getNumberOfDigits());
    }
  }
}

void Trap2CRU::readTrapData()
{
  //set things up, read the file and then deligate to convertTrapdata to do the conversion.
  //
  mRawData.reserve(1024 * 1024); //TODO take out the hardcoded 1MB its supposed to come in from the options
  if (mVerbosity) {
    LOG(info) << "Trap2CRU::readTrapData";
  }
  // data comes in index by event (triggerrecord)
  // first 15 links go to cru0a, second 15 links go to cru0b, 3rd 15 links go to cru1a ... first 90 links to flp0 and then repeat for 12 flp
  // then do next event

  // lets register our links
  std::string prefix = mOutputDir;
  if (!prefix.empty() && prefix.back() != '/') {
    prefix += '/';
  }

  for (int link = 0; link < o2::trd::constants::NHALFCRU; link++) {
    // FeeID *was* 0xFEED, now is indicates the cru Supermodule, side (A/C) and endpoint. See RawData.cxx for details.
    int supermodule = link / 4;
    int endpoint = link % 2;
    int cru = link / 2;
    int side = cru % 2; // A or C, 0 or 1 respectively:
    mFeeID = buildTRDFeeID(supermodule, side, endpoint);
    LOG(info) << "FEEID;" << std::hex << mFeeID;
    mCruID = link / 2;
    mEndPointID = endpoint;
    mLinkID = o2::trd::constants::TRDLINKID;

    std::string outFileLink;
    std::string outPrefix = "TRD_";
    outPrefix += "alio2-cr1-flp";
    std::string outSuffix = ".raw";
    // filename structure of trd_cru_[CRU#]_[upper/lower].raw
    auto flpid = trdHWMap[mCruID].flpid;
    auto cruhwid = trdHWMap[mCruID].cruHWID;
    if (mFilePer == "all") { // single file for all links
      outFileLink = o2::utils::Str::concat_string(mOutputDir, "/", outPrefix, outSuffix);
    } else if (mFilePer == "sm") {
      int sm = link / 4;
      std::stringstream ss;
      ss << std::setw(2) << std::setfill('0') << sm;
      std::string supermodule = ss.str();
      outFileLink = o2::utils::Str::concat_string(mOutputDir, "/", outPrefix, "_sm_", supermodule, outSuffix);
    } else if (mFilePer == "cru") {
      outFileLink = o2::utils::Str::concat_string(mOutputDir, "/", outPrefix, std::to_string(flpid), "_cru", std::to_string(cruhwid), "_", std::to_string(mEndPointID), outSuffix);
    } else if (mFilePer == "halfcru") {
      outFileLink = o2::utils::Str::concat_string(mOutputDir, "/", outPrefix, std::to_string(flpid), "_cru", std::to_string(cruhwid), "_", std::to_string(mEndPointID), outSuffix);
    } else {
      throw std::runtime_error("invalid option provided for file grouping");
    }

    //std::string outputFilelink = o2::utils::Str::concat_string(prefix, "trd_cru_", std::to_string(mCruID), "_", trdside, "_", whichrun, ".raw");
    LOG(info) << "registering links";

    mWriter.registerLink(mFeeID, mCruID, mLinkID, mEndPointID, outFileLink);
  }

  openInputFiles();

  if (mTrackletsTree->GetEntries() != mDigitsTree->GetEntries()) {
    LOG(fatal) << "Entry counts in mTrackletsTree and Digits Tree dont match " << mTrackletsTree->GetEntries() << "!=" << mDigitsTree->GetEntries();
  }
  uint32_t totaltracklets = 0;
  uint32_t totaldigits = 0;
  int triggercount = 42; // triggercount is here so that we can span timeframes. The actual number is of no consequence,but must increase.
  for (int entry = 0; entry < mTrackletsTree->GetEntries(); entry++) {
    mTrackletsTree->GetEntry(entry);
    mDigitsTree->GetEntry(entry);
    totaltracklets += mTracklets.size();
    totaldigits += mDigits.size();
    //migrate digit trigger information into the tracklettrigger (historical)
    sortDataToLinks();
    // each entry is a timeframe
    uint32_t linkcount = 0;
    for (auto tracklettrigger : mTrackletTriggerRecords) {
      convertTrapData(tracklettrigger, triggercount); // tracklettrigger assumed to be authoritive
      triggercount++;
      mEventDigitCount++;
    }
  }
  LOG(info) << " Total digits : " << totaldigits;
  LOG(info) << " Total tracklets : " << totaltracklets;
}

void Trap2CRU::linkSizePadding(uint32_t linksize, uint32_t& crudatasize, uint32_t& padding)
{

  // if zero the whole 256 bit must be padded (empty link)
  // crudatasize is the size to be stored in the cruheader, i.e. units of 256bits.
  // linksize is the incoming link size from the linkrecord,
  // padding is of course the amount of padding in 32bit words.
  uint32_t rem = 0;
  if (linksize != 0) {
    //data, so figure out padding cru word, the other case is simple, full padding if size=0
    rem = linksize % 8;
    if (rem != 0) {
      crudatasize = linksize / 8 + 1;
      padding = 8 - rem;
    } else {

      crudatasize = linksize / 8; // 32 bit word to 256 bit word.
      padding = 0;
    }
    if (mVerbosity) {
      LOG(info) << "We have data with linkdatasize=" << linksize << " with size number in header of:" << crudatasize << " padded with " << padding << " 32bit words";
    }
  } else {
    //linksize is zero so no data, pad fully.
    crudatasize = 1;
    padding = 8;
    if (mVerbosity) {
      LOG(info) << "We have data with linkdatasize=" << linksize << " with size number in header of:" << crudatasize << " padded with " << padding << " 32bit words";
    }
  }
}

uint32_t Trap2CRU::buildHalfCRUHeader(HalfCRUHeader& header, const uint32_t bc, const uint32_t halfcru)
{
  int bunchcrossing = bc;
  int stopbits = 0x01; // do we care about this and eventtype in simulations?
  int eventtype = o2::trd::constants::ETYPECALIBRATIONTRIGGER;
  int crurdhversion = 6;
  int feeid = 0;
  int cruid = 0;
  uint32_t crudatasize = 0; //link size in units of 256 bits.
  int endpoint = halfcru % 2 ? 1 : 0;
  uint32_t padding = 0;
  //lets first clear it out.
  clearHalfCRUHeader(header);
  //this bunchcrossing is not the same as the bunchcrossing in the rdh, which is the bc coming in the parameter list to this function. See explanation in rawdata.h
  setHalfCRUHeader(header, crurdhversion, bunchcrossing, stopbits, endpoint, eventtype, feeid, cruid);

  return 1;
}

bool Trap2CRU::isTrackletOnLink(const int linkid, const int currenttrackletpos)
{
  //hcid is simply the halfcru*15+linkid
  if (currenttrackletpos == mTracklets.size()) {
    return false;
  }
  int link = HelperMethods::getLinkIDfromHCID(mTracklets[currenttrackletpos].getHCID());
  if (linkid == link) {
    // this tracklet is on this link.
    return true;
  }
  return false;
}

bool Trap2CRU::isDigitOnLink(const int linkid, const int currentdigitpos)
{
  if (currentdigitpos >= mDigits.size()) {
    return false;
  }
  Digit* digit = &mDigits[mDigitsIndex[currentdigitpos]];
  int link = HelperMethods::getLinkIDfromHCID(digit->getDetector() * 2 + (digit->getROB() % 2));
  if (link == linkid) {
    return true;
  }
  return false;
}

int Trap2CRU::buildDigitRawData(const int digitstartindex, const int digitendindex, const int mcm, const int rob, const uint32_t triggerrecordcount)
{
  //this is not zero suppressed.
  int digitwordswritten = 0;
  int digitswritten = 0;
  //    Digit
  DigitMCMHeader header;
  DigitMCMData data;
  int startdet = mDigits[mDigitsIndex[digitstartindex]].getDetector();
  int startrob = mDigits[mDigitsIndex[digitstartindex]].getROB();
  int startmcm = mDigits[mDigitsIndex[digitstartindex]].getMCM();
  if (startrob != rob && startmcm != mcm) {
    LOG(error) << "buildDigitRawData but startmcm and rob are not correct : " << startrob << "!=" << rob << " and mcm: " << startmcm << "!=" << mcm;
  }
  int digitcounter = 0;
  header.res = 0xc; //1100
  header.mcm = startmcm;
  header.rob = startrob;
  header.yearflag = 1; // >10.2007
  header.eventcount = triggerrecordcount;
  memcpy(mRawDataPtr, (char*)&header, sizeof(DigitMCMHeader)); // uint32 -- 4 bytes.
  DigitMCMHeader* headerptr = (DigitMCMHeader*)mRawDataPtr;
  //LOG(info) << "Digt Header word: 0x" << std::hex << headerptr->word;
  mRawDataPtr += 4;
  digitwordswritten++;
  //we are writing zero suppressed so
  DigitMCMADCMask adcmask;
  adcmask = buildBlankADCMask();
  memcpy(mRawDataPtr, (char*)&adcmask, sizeof(DigitMCMADCMask));
  DigitMCMADCMask* adcmaskptr = (DigitMCMADCMask*)mRawDataPtr;
  mRawDataPtr += 4;
  digitwordswritten++;
  //LOG(info) << "writing data to digit stream of " << std::hex << header.word;
  for (int digitindex = digitstartindex; digitindex < digitendindex; ++digitindex) {
    Digit* d = &mDigits[mDigitsIndex[digitindex]];
    ArrayADC adcdata = d->getADC();
    //write these 2 now as we only have it now.
    if (startmcm != d->getMCM()) {
      LOG(error) << " we are on the wrong mcm:" << startmcm << "!=" << d->getMCM();
    }
    if (startrob != d->getROB()) {
      LOG(error) << " we are on the wrong rob:" << startrob << "!=" << d->getROB();
    }
    int channel = d->getChannel();
    //set adcmask for the channel we currently have.
    adcmaskptr->adcmask |= 1UL << channel;
    for (int timebin = 0; timebin < o2::trd::constants::TIMEBINS; timebin += 3) {
      data.z = adcdata[timebin];
      data.y = adcdata[timebin + 1];
      data.x = adcdata[timebin + 2];
      data.c = (channel % 2 == 0) ? 0x3 : 0x2;                 // 3 for even channel 2 for odd channel
      memcpy(mRawDataPtr, (char*)&data, sizeof(DigitMCMData)); // uint32 -- 4 bytes.
      mRawDataPtr += sizeof(DigitMCMData);
      digitwordswritten++;
    }
    if (mVerbosity) {
      LOG(info) << "DDDD " << d->getDetector() << ":" << d->getROB() << ":" << d->getMCM() << ":" << d->getChannel() << ":" << d->getADCsum() << ":" << d->getADC()[0] << ":" << d->getADC()[1] << ":" << d->getADC()[2] << "::" << d->getADC()[27] << ":" << d->getADC()[28] << ":" << d->getADC()[29];
    }
    if (d->getMCM() != startmcm) {
      LOG(fatal) << "digit getmcm = " << d->getMCM() << " while startmcm=" << startmcm;
    }
    digitswritten++;
  }
  if (digitswritten != digitendindex - digitstartindex) {
    LOG(error) << " something wrong the number of digitswritten does not correspond to the the loop count";
  }
  if (digitwordswritten != (digitswritten * 10 + 2)) {
    LOG(error) << "something wrong with writing the digits the following should be equal " << digitwordswritten << "==" << (digitswritten * 10 + 2) << " with digitswritten=" << digitswritten;
    LOG(error) << "digit start index distance to digit end index :" << digitendindex - digitstartindex;
  }
  return digitwordswritten;
}

int Trap2CRU::buildTrackletRawData(const int trackletindex, const int linkid)
{
  TrackletMCMHeader header;
  bool destroytracklets = false;
  std::array<TrackletMCMData, 3> trackletdata;

  header.col = mTracklets[trackletindex].getColumn();
  header.padrow = mTracklets[trackletindex].getPadRow();
  header.onea = 1;
  header.oneb = 1;
  header.pid0 = 0xff;
  header.pid1 = 0xff;
  header.pid2 = 0xff;
  unsigned int trackletcounter = 0;
  if (mVerbosity) {
    LOG(info) << "After instantiation header is : 0x" << header.word << "  " << header << " Trackletindex:" << trackletindex << " max tracklet:" << mTracklets.size();
    LOG(info) << "mTracklet:" << mCurrentTracklet << "  ==?? trackletindex" << trackletindex << " max tracklet:" << mTracklets.size();
  }
  while (linkid == HelperMethods::getLinkIDfromHCID(mTracklets[trackletindex + trackletcounter].getHCID()) && header.col == mTracklets[trackletindex + trackletcounter].getColumn() && header.padrow == mTracklets[trackletindex + trackletcounter].getPadRow()) {
    int trackletoffset = trackletindex + trackletcounter;
    buildTrackletMCMData(trackletdata[trackletcounter], mTracklets[trackletoffset].getSlope(),
                         mTracklets[trackletoffset].getPosition(), mTracklets[trackletoffset].getQ0(),
                         mTracklets[trackletoffset].getQ1(), mTracklets[trackletoffset].getQ2());
    unsigned int headerqpart = ((mTracklets[trackletoffset].getQ2() & 0x2f) << 2) + ((mTracklets[trackletoffset].getQ1() >> 6) & 0x3);
    //all 6 bits of Q1 and 2 upper bits of 7bit Q1
    if (mVerbosity) {
      if (mTracklets[trackletoffset].getQ2() > 0x3f) {
        LOGP(warning, "Tracklet Q2 out of range for raw data {0:#x}", mTracklets[trackletoffset].getQ2());
      }
      if (mTracklets[trackletoffset].getQ1() > 0x7f) {
        LOGP(warning, "Tracklet Q1 out of range for raw data {0:#x}", mTracklets[trackletoffset].getQ1());
      }
      if (mTracklets[trackletoffset].getQ0() > 0x7f) {
        LOGP(warning, "Tracklet Q0 out of range for raw data {0:#x}", mTracklets[trackletoffset].getQ0());
      }
    }
    switch (trackletcounter) {
      case 0:
        header.pid0 = headerqpart;
        break;
      case 1:
        header.pid1 = headerqpart;
        if (header.pid0 == 0xff) {
          LOG(warn) << "we are setting pid1 but pid0 is not set, a second tracklet but no first one?";
        }
        break;
      case 2:
        header.pid2 = headerqpart;
        if (header.pid1 == 0xff || header.pid0 == 0xff) {
          LOG(warn) << "we are setting pid2 but pid0/1 is not set, a second tracklet but no first one?" << header.pid0 << " " << header.pid1;
        }
        break;
      default:
        LOG(warn) << ">3 tracklets when building the Tracklet raw data stream for hcid=" << mTracklets[trackletindex + trackletcounter].getHCID() << " col:" << mTracklets[trackletindex + trackletcounter].getColumn() << " padrow:" << mTracklets[trackletindex + trackletcounter].getPadRow();
        destroytracklets = true;
        break;
    }
    trackletcounter++;
    if (trackletcounter + trackletindex >= mTracklets.size()) {
      break;
    }
  }
  //now copy the mcmheader and mcmdata.
  if (!destroytracklets) {
    setNumberOfTrackletsInHeader(header, trackletcounter);
    if (trackletcounter > 0) { // dont write header if there are no tracklets.
      memcpy((char*)mRawDataPtr, (char*)&header, sizeof(TrackletMCMHeader));
      mRawDataPtr += sizeof(TrackletMCMHeader);
      for (int i = 0; i < trackletcounter; ++i) {
        memcpy((char*)mRawDataPtr, (char*)&trackletdata[i], sizeof(TrackletMCMData));
        mRawDataPtr += sizeof(TrackletMCMData);
      }
    }
  } else {
    LOG(warn) << "something wrong with these tracklets, there are too many. You might want to take a closer look. Rejecting for now, and moving on.";
  }
  return trackletcounter;
}

int Trap2CRU::writeDigitEndMarker()
{
  int wordswritten = 0;
  uint32_t digitendmarker = 0;

  memcpy(mRawDataPtr, (char*)&digitendmarker, 4);
  mRawDataPtr += 4;
  wordswritten++;
  memcpy(mRawDataPtr, (char*)&digitendmarker, 4);
  mRawDataPtr += 4;
  wordswritten++;

  return wordswritten;
}

int Trap2CRU::writeTrackletEndMarker()
{
  int wordswritten = 0;
  uint32_t trackletendmarker = 0x10001000;

  memcpy(mRawDataPtr, (char*)&trackletendmarker, 4);
  mRawDataPtr += 4;
  wordswritten++;
  memcpy(mRawDataPtr, (char*)&trackletendmarker, 4);
  mRawDataPtr += 4;
  wordswritten++;
  return wordswritten;
}

int Trap2CRU::writeTrackletHCHeader(const int eventcount)
{
  int wordswritten = 0;
  //from linkid we can get supermodule, stack, layer, side
  int linkid = mTracklets[mCurrentTracklet].getHCID();
  int detector = mTracklets[mCurrentTracklet].getHCID() / 2;
  TrackletHCHeader trackletheader;
  trackletheader.supermodule = linkid / 60;
  trackletheader.stack = (detector % (o2::trd::constants::NLAYER * o2::trd::constants::NSTACK)) / o2::trd::constants::NLAYER;
  trackletheader.layer = (detector % o2::trd::constants::NLAYER);
  trackletheader.one = 1;
  if (mVerbosity) {
    LOG(info) << "Tracklet linkid : " << linkid << ":"
              << " " << trackletheader.supermodule << ":" << trackletheader.stack << ":" << trackletheader.layer << ":" << trackletheader.side;
  }
  trackletheader.side = (linkid % 2) ? 1 : 0;
  trackletheader.MCLK = eventcount * 42; // just has to be a constant increasing number per event for our purposes in sim to raw.
  trackletheader.format = 12;
  if (mUseTrackletHCHeader) { // run 3 we also have a TrackletHalfChamber.
    memcpy(mRawDataPtr, (char*)&trackletheader, sizeof(TrackletHCHeader));
    if (mVerbosity) {
      LOG(info) << "writing tracklethcheader of 0x" << std::hex << trackletheader.word;
    }
    mRawDataPtr += 4;
    wordswritten++;
  }
  return wordswritten;
}

int Trap2CRU::writeDigitHCHeader(const int eventcount, const uint32_t linkid)
{
  // we have 2 HCHeaders defined Tracklet and Digit in Rawdata.h
  int wordswritten = 0;
  //from linkid we can get supermodule, stack, layer, side
  int detector = linkid / 2;

  DigitHCHeader digitheader;
  DigitHCHeader1 digitheader1;
  digitheader.res = 1;
  digitheader.side = (linkid % 2) ? 1 : 0;
  digitheader.stack = (detector % (o2::trd::constants::NLAYER * o2::trd::constants::NSTACK)) / o2::trd::constants::NLAYER;
  digitheader.layer = (detector % o2::trd::constants::NLAYER);
  digitheader.supermodule = linkid / 60;
  digitheader.numberHCW = 1; // number of additional words in th header, we are using 2 header words so 1 here.
  digitheader.minor = 42;    // my (shtm) version, not used
  digitheader.major = 0x21;  // zero suppressed and 0x1 to comply with what we see in the raw data
  digitheader.version = 1;   //new version of the header. we only have 1 version
  digitheader1.res = 1;
  digitheader1.ptrigcount = 1;
  digitheader1.ptrigphase = 1;
  digitheader1.bunchcrossing = eventcount; //NB this is not the same as the bunchcrossing the rdh. See RawData.h for explanation
  digitheader1.numtimebins = 30;
  memcpy(mRawDataPtr, (char*)&digitheader, sizeof(DigitHCHeader)); // 8 because we are only using the first 2 32bit words of the header, the rest are optional.
  mRawDataPtr += sizeof(DigitHCHeader);
  memcpy(mRawDataPtr, (char*)&digitheader1, sizeof(DigitHCHeader1)); // 8 because we are only using the first 2 32bit words of the header, the rest are optional.
  mRawDataPtr += sizeof(DigitHCHeader1);
  wordswritten += 2;
  return wordswritten;
}

void Trap2CRU::convertTrapData(o2::trd::TriggerRecord const& triggerrecord, const int& triggercount)
{
  //build a HalfCRUHeader for this event/cru/endpoint
  //loop over cru's
  //  loop over all half chambers, thankfully they data is sorted.
  //    check if current chamber has a link
  //      if not blank, else fill in data from link records
  //  dump data to rawwriter
  //finished for event. this method is only called per event.
  //    char* traprawdataptr = (char*)&mTrapRawData[0];
  std::array<int64_t, 21> localParsedDigitsindex; // store the index of the digits of an mcm
  int rawwords = 0;
  char* rawdataptratstart;
  std::vector<char> rawdatavector(1024 * 1024 * 2); // sum of link sizes + padding in units of bytes and some space for the header (512 bytes).
  if (mVerbosity) {
    LOG(info) << "BUNCH CROSSING : " << triggerrecord.getBCData().bc << " with orbit : " << triggerrecord.getBCData().orbit;
  }

  //set startdigit and starttracklet relative to the trigger.
  //
  int starttrackletindex = triggerrecord.getFirstTracklet();
  int endtrackletindex = triggerrecord.getFirstTracklet() + triggerrecord.getNumberOfTracklets();
  int64_t startdigitindex = triggerrecord.getFirstDigit();
  int64_t enddigitindex = triggerrecord.getFirstDigit() + triggerrecord.getNumberOfDigits();

  for (int halfcru = 0; halfcru < o2::trd::constants::NHALFCRU; halfcru++) {
    int halfcruwordswritten = 0;
    int supermodule = halfcru / 4; // 2 cru per supermodule. 72/4, as of writing
    mEndPointID = halfcru % 2;     // 2 pci end points per cru, 15 links each
    //first cru is A second CRU is C , so an flp will be either ACA or CAC A=0 C=1
    int cru = halfcru / 2;
    int side = cru % 2; // first cru is A second is B, 3rd is A etc
    mFeeID = buildTRDFeeID(supermodule, side, mEndPointID);
    mCruID = halfcru / 2;
    mLinkID = o2::trd::constants::TRDLINKID;
    mEndPointID = halfcru % 2; // just the upper or lower half of the cru, hence %2 of the the halfcru number.
    std::string sside;
    (side) ? sside = "C" : sside = "A";
    //15 links per half cru or cru end point.
    memset(&mRawData[0], 0, sizeof(mRawData[0]) * mRawData.size()); //   zero the rawdata storage
    HalfCRUHeader halfcruheader;
    //now write the cruheader at the head of all the data for this halfcru.
    buildHalfCRUHeader(halfcruheader, triggerrecord.getBCData().bc, halfcru);
    halfcruheader.EndPoint = mEndPointID;
    mRawDataPtr = rawdatavector.data();
    HalfCRUHeader* halfcruheaderptr = (HalfCRUHeader*)mRawDataPtr; // store the ptr to the halfcruheader for later adding the link lengths and possibly simulated errors.
    mRawDataPtr += sizeof(halfcruheader);
    halfcruwordswritten += sizeof(halfcruheader) / 4;
    int totallinklengths = 0;
    rawdataptratstart = mRawDataPtr; // keep track of where we started.
    for (int halfcrulink = 0; halfcrulink < o2::trd::constants::NLINKSPERHALFCRU; halfcrulink++) {
      //links run from 0 to 14, so linkid offset is halfcru*15;
      int linkid = halfcrulink + halfcru * o2::trd::constants::NLINKSPERHALFCRU;
      if (mVerbosity) {
        LOG(info) << " linkid : " << linkid << " with link " << halfcrulink << "  of halfcru " << halfcru << " tracklet is on link for linkid : " << linkid << " and tracklet index of : " << mCurrentTracklet << " with current digit index : " << mCurrentDigit;
      }
      int linkwordswritten = 0;
      int errors = 0;           // put no errors in for now.
      int size = 0;             // in 32 bit words
      uint32_t paddingsize = 0; // in 32 bit words
      uint32_t crudatasize = 0; // in 256 bit words.
      //loop over tracklets for mcm's that match
      int tracklets = 0;
      int trackletendmarker = 0;
      int adccounter = 0;
      int rawwordsbefore = 0;
      bool isFirstDigit = true;
      int trackletcounter = 0;
      if (mVerbosity) {
        LOG(info) << "tracklet on link : " << linkid << " mcurrenttracklet:" << mCurrentTracklet << " endtrackletindex:" << endtrackletindex << " is on link: " << isTrackletOnLink(linkid, mCurrentTracklet) << " and digits current digit:" << mCurrentDigit << " enddigitindex:" << enddigitindex << "is digit on link:" << isDigitOnLink(linkid, mCurrentDigit);
      }
      if (isTrackletOnLink(linkid, mCurrentTracklet) || isDigitOnLink(linkid, mCurrentDigit)) {
        // we have some data somewhere for this link
        //write tracklet half chamber header irrespective of there being tracklet data
        if (mUseTrackletHCHeader != 0) {
          if (isTrackletOnLink(linkid, mCurrentTracklet) || mUseTrackletHCHeader == 2) {
            //write tracklethcheader if there is tracklet data or if we always have tracklethcheader
            //first part of the if statement handles the mUseTrackletHCHeader==1 option
            int hcheaderwords = writeTrackletHCHeader(triggercount);
            linkwordswritten += hcheaderwords;
            rawwords += hcheaderwords;
          }
          //else do nothing as we dont want/have tracklethcheader
        }
        while (isTrackletOnLink(linkid, mCurrentTracklet) && mCurrentTracklet < endtrackletindex) {
          // still on an mcm on this link
          tracklets = buildTrackletRawData(mCurrentTracklet, linkid); //returns # of 32 bits, header plus trackletdata words that would have come from the mcm.
          mCurrentTracklet += tracklets;
          trackletcounter += tracklets;
          linkwordswritten += tracklets + 1;
          rawwords += tracklets + 1; //1 to include the header
        }
        if (mCurrentTracklet >= mTracklets.size()) {
          LOG(debug) << " finished with tracklets";
        }
        //write tracklet end marker irrespective of their being tracklet data.
        trackletendmarker = writeTrackletEndMarker();
        linkwordswritten += trackletendmarker;
        rawwords += trackletendmarker;
        adccounter = 0;
        rawwordsbefore = rawwords;
        //always write the digit hc header
        int hcheaderwords = 0;
        if (mCurrentDigit >= mDigits.size()) {
          // take care of the case where the digit hc header is written but we have no more digits to write, we then need to get the half chamber header from the tracklet.
          hcheaderwords = writeDigitHCHeader(triggercount, mTracklets[mCurrentTracklet].getHCID());
        } else {
          hcheaderwords = writeDigitHCHeader(triggercount, mDigits[mDigitsIndex[mCurrentDigit]].getHCId());
        }

        linkwordswritten += hcheaderwords;
        rawwords += hcheaderwords;
        //although if there are trackelts there better be some digits unless the digits are switched off.
        if (mCurrentDigit < mDigits.size()) {
          while (isDigitOnLink(linkid, mCurrentDigit) && mCurrentDigit < enddigitindex && mEventDigitCount % mDigitRate == 0) {
            if (mVerbosity) {
              LOG(info) << "at top of digit while loop calc linkid :" << linkid << " : hcid=" << mDigits[mDigitsIndex[mCurrentDigit]].getHCId() << " actual link=" << linkid;
              LOG(info) << "mCurrentDigit :" << mCurrentDigit;
            }
            //while we are on a single mcm, copy the digits timebins to the array.
            int digitcounter = 0;
            int currentROB = mDigits[mDigitsIndex[mCurrentDigit]].getROB();
            int currentMCM = mDigits[mDigitsIndex[mCurrentDigit]].getMCM();
            int currentDetector = mDigits[mDigitsIndex[mCurrentDigit]].getDetector();
            int startmCurrentDigit = mCurrentDigit;
            while (mDigits[mDigitsIndex[mCurrentDigit]].getMCM() == currentMCM &&
                   mDigits[mDigitsIndex[mCurrentDigit]].getROB() == currentROB &&
                   mDigits[mDigitsIndex[mCurrentDigit]].getDetector() == currentDetector) {
              LOG(debug) << " on index of : " << mDigitsIndex[mCurrentDigit] << " wuf channel=" << mDigits[mDigitsIndex[mCurrentDigit]].getChannel();
              mCurrentDigit++;
              digitcounter++;
              adccounter++;
              if (digitcounter > 22) {
                LOG(error) << " we are on the 22nd digit of an mcm ?? This is not possible";
              }
              if (mCurrentDigit == mDigits.size()) {
                break;
              }
            }
            // mcm digits are full, now write it out.
            char* preptr;
            preptr = mRawDataPtr;
            int digitwordswritten = 0;
            digitwordswritten = buildDigitRawData(startmCurrentDigit, mCurrentDigit, currentMCM, currentROB, triggercount);
            linkwordswritten += digitwordswritten;
          }
        }
      }
      if (mVerbosity) {
        LOG(info) << "link:" << linkid << " trackletcounter: " << trackletcounter << "  currenttracklet: " << mCurrentTracklet << " adccounter :" << adccounter << " current digit : " << mCurrentDigit;
      }
      int counter = 0;
      if (adccounter > 0 || trackletcounter > 0) {
        //write the tracklet end marker so long as we have any data (digits or tracklets).
        int digitendmarkerwritten = writeDigitEndMarker();
        linkwordswritten += digitendmarkerwritten;
        rawwords += digitendmarkerwritten;
      }
      //pad up to a whole 256 bit word size
      if (linkwordswritten != 0) {
        crudatasize = linkwordswritten / 8;
        linkSizePadding(linkwordswritten, crudatasize, paddingsize);

        // now pad the data if needed ....
        char* olddataptr = mRawDataPtr; // store the old pointer so we can do some sanity checks for how far we advance.
        //now for padding
        uint16_t padbytes = paddingsize * sizeof(uint32_t);
        uint32_t padword = 0xeeeeeeee;
        for (int i = 0; i < paddingsize; ++i) {
          memcpy(mRawDataPtr, (char*)&padword, 4);
          mRawDataPtr += 4;
          linkwordswritten++;
          rawwords++;
        }
        crudatasize = linkwordswritten / 8; //convert to 256 bit alignment.
        if ((linkwordswritten % 8) != 0) {
          LOG(error) << "linkwordswritten is not 256 bit aligned " << linkwordswritten << " %8 = " << linkwordswritten % 8 << " and a padding size of : " << paddingsize << " or padbytes of : " << padbytes;
        }
        //set the halfcruheader for the length of this link.
        //but first a sanity check.
        if (crudatasize > constants::MAXDATAPERLINK256) {
          LOG(error) << " linksize is huge : " << crudatasize;
        }
        LOG(debug) << " setting halfcrulink " << halfcrulink << " linksize to : " << crudatasize << " with a linkwordswrittern=" << linkwordswritten;
        setHalfCRUHeaderLinkData(halfcruheader, halfcrulink, crudatasize, errors);
        uint32_t bytescopied;
        totallinklengths += crudatasize;
        if ((mRawDataPtr - rawdataptratstart) != (totallinklengths * 32)) {
          bytescopied = mRawDataPtr - rawdataptratstart;
          if (mVerbosity) {
            LOG(info) << "something wrong with data size in cruheader writing"
                      << "linkwordswriten:"
                      << linkwordswritten << " rawwords:" << rawwords << "bytestocopy : "
                      << bytescopied << " crudatasize:" << crudatasize << " sum of links up to now : "
                      << totallinklengths << " mRawDataPtr:0x" << std::hex << (void*)mRawDataPtr
                      << "  start ptr:" << std::hex << (void*)rawdataptratstart;
          }
        } else {
          if (mVerbosity) {
            LOG(debug) << "all fine with data size writing padbytes:" << paddingsize
                       << " linkwordswriten:" << linkwordswritten << " bytestocopy : " << bytescopied
                       << " crudatasize:" << crudatasize << " mRawDataPtr:0x" << std::hex
                       << (void*)mRawDataPtr << "  start ptr:" << std::hex << (void*)rawdataptratstart;
          }
        }
        //sanity check for now:
        if (crudatasize != o2::trd::getlinkdatasize(halfcruheader, halfcrulink)) {
          // we have written the wrong amount of data ....
          LOG(warn) << "crudata is ! = get link data size " << crudatasize << "!=" << o2::trd::getlinkdatasize(halfcruheader, halfcrulink);
        }
      } // if we have data on link
      else {
        setHalfCRUHeaderLinkData(halfcruheader, halfcrulink, 0, 0);
        if (mVerbosity) {
          LOG(info) << "linkwordswritten is zero : " << linkwordswritten;
        }
        if (crudatasize != 0) {
          LOG(warn) << " we should not be here with a crudatasize of " << crudatasize << " as the linkwordswritten is " << linkwordswritten << " with a halfcrulink of : " << halfcrulink;
          LOG(debug) << " ### setting halfcrulink " << halfcrulink << " linksize to : " << crudatasize << " with a linkwordswrittern=" << linkwordswritten;
        }
        // setHalfCRUHeaderLinkData(halfcruheader, halfcrulink, 0,0);
      }
      halfcruwordswritten += linkwordswritten;
    } // if tracklets.size >0
    //write the cruhalfheader now that we know the lengths.
    memcpy((char*)halfcruheaderptr, (char*)&halfcruheader, sizeof(halfcruheader));
    //write halfcru data here.
    std::vector<char> feeidpayload(halfcruwordswritten * 4);
    memcpy(feeidpayload.data(), &rawdatavector[0], halfcruwordswritten * 4);
    assert(halfcruwordswritten % 8 == 0);
    mWriter.addData(mFeeID, mCruID, mLinkID, mEndPointID, triggerrecord.getBCData(), feeidpayload, false, triggercount);
    if (mVerbosity) {
      LOG(info) << "written file for trigger : " << triggercount << " feeid of 0x" << std::hex << mFeeID << " cruid : " << mCruID << " and linkid: " << mLinkID << " and EndPoint: " << mEndPointID << " orbit :0x" << std::hex << triggerrecord.getBCData().orbit << " bc:0x" << std::hex << triggerrecord.getBCData().bc << " and payload size of : " << halfcruwordswritten << " with  a half cru of: ";
      printHalfCRUHeader(halfcruheader);
      HalfCRUHeader* h;
      h = (HalfCRUHeader*)feeidpayload.data();
      HalfCRUHeader h1 = *h;
      printHalfCRUHeader(h1);
      LOG(info) << "+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+!+  ======   end of writing";
    }
  }
}

} // end namespace trd
} // end namespace o2
