// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "TRDBase/CommonParam.h"
#include "TFile.h"
#include "TTree.h"
#include <TStopwatch.h>
#include <cassert>
#include <fstream>
#include <iostream>
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
  if (mDigitsFile != nullptr) {
    mDigitsTree = (TTree*)mDigitsFile->Get("o2sim");
    mDigitsTree->SetBranchAddress("TRDDigit", &mDigitsPtr);                   // the branch with the actual digits
    mDigitsTree->SetBranchAddress("TriggerRecord", &mDigitTriggerRecordsPtr); // branch with trigger records for digits
                                                                              //    LOG(info) << "Digit tree has " << mDigitsTree->GetEntries() << " entries"; //<<
  } else {
    LOG(warn) << " cant open file containing digit tree";
  }
  mTrackletsFile = TFile::Open(mInputTrackletsFileName.data());
  if (mTrackletsFile != nullptr) {
    mTrackletsTree = (TTree*)mTrackletsFile->Get("o2sim");
    mTrackletsTree->SetBranchAddress("Tracklet", &mTrackletsPtr);              // the branch with the actual tracklets.
    mTrackletsTree->SetBranchAddress("TrackTrg", &mTrackletTriggerRecordsPtr); // branch with trigger records for digits
                                                                               //  LOG(info) << "Tracklet tree has " << mTrackletsTree->GetEntries() << " entries";//<<
  } else {
    LOG(fatal) << " cant open file containing tracklet tree";
  }
}

void Trap2CRU::sortDataToLinks()
{
  //sort the digits array TODO refactor this intoa vector index sort and possibly generalise past merely digits.
  auto sortstart = std::chrono::high_resolution_clock::now();
  //build indexes
  // digits first
  mDigitsIndex.resize(mDigits.size());
  std::iota(mDigitsIndex.begin(), mDigitsIndex.end(), 0);

  for (auto& trig : mDigitTriggerRecords) {
    if (trig.getNumberOfDigits() != 0) {
      LOG(debug) << " sorting digits from : " << trig.getFirstDigit() << " till " << trig.getFirstDigit() + trig.getNumberOfDigits();
      std::stable_sort(mDigitsIndex.begin() + trig.getFirstDigit(), mDigitsIndex.begin() + trig.getNumberOfDigits() + trig.getFirstDigit(), //,digitcompare);
                       [this](const uint32_t i, const uint32_t j) { 
             uint32_t hcida = mDigits[i].getDetector() * 2 + (mDigits[i].getROB() % 2);
             uint32_t hcidb = mDigits[j].getDetector() * 2 + (mDigits[j].getROB() % 2);
             if(hcida!=hcidb){return hcida<hcidb;}
             if(mDigits[i].getROB() != mDigits[j].getROB()){return (mDigits[i].getROB() < mDigits[j].getROB());}
             return (mDigits[i].getMCM() < mDigits[j].getMCM()); });
    }
  }
  for (auto& trig : mTrackletTriggerRecords) {
    if (trig.getNumberOfTracklets() > 0) {
      LOG(debug) << " sorting digits from : " << trig.getFirstTracklet() << " till " << trig.getFirstTracklet() + trig.getNumberOfTracklets();
      std::stable_sort(std::begin(mTracklets) + trig.getFirstTracklet(), std::begin(mTracklets) + trig.getNumberOfTracklets() + trig.getFirstTracklet(),
                       [this](auto&& t1, auto&& t2) {
                         if (t1.getHCID() != t2.getHCID()) {
                           return t1.getHCID() < t2.getHCID();
                         }
                         if (t1.getPadRow() != t2.getPadRow()) {
                           return t1.getPadRow() < t2.getPadRow();
                         }
                         return t1.getMCM() < t2.getMCM();
                       });
    }
  }

  std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - sortstart;
  LOG(debug) << "TRD Digit/Tracklet Sorting took " << duration.count() << " s";
}

void Trap2CRU::mergetriggerDigitRanges()
{
  // pass through the digit ranges of the incoming tracklet triggers.
  // this most handles the old data.
  // trapsim should now be sending out the trigger with both information.
  bool fixdigitinfo = false;
  for (auto trig : mTrackletTriggerRecords) {
    if (trig.getNumberOfDigits() != 0) {
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
  LOG(debug) << "Trap2CRU::readTrapData";
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
    int endpoint = link / 2;
    int side = link % 2 ? 1 : 0;
    mFeeID = buildTRDFeeID(supermodule, side, endpoint);
    mCruID = link / 2;
    mEndPointID = link % 2 ? 1 : 0;
    mLinkID = o2::trd::constants::TRDLINKID;

    std::string outFileLink;
    std::string outPrefix = "trd";
    std::string outSuffix = ".raw";
    std::string trdside = mEndPointID ? "_h" : "_l"; // the side of cru lower or higher
    // filename structure of trd_cru_[CRU#]_[upper/lower].raw
    std::string whichrun = mUseTrackletHCHeader ? "run3" : "run2";
    if (mFilePer == "all") { // single file for all links
      outFileLink = o2::utils::Str::concat_string(mOutputDir, "/", outPrefix, outSuffix);
    } else if (mFilePer == "sm") {
      int sm = link / 4;
      std::stringstream ss;
      ss << std::setw(2) << std::setfill('0') << sm;
      std::string supermodule = ss.str();
      outFileLink = o2::utils::Str::concat_string(mOutputDir, "/", outPrefix, "_sm_", supermodule, outSuffix);
    } else if (mFilePer == "cru") {
      outFileLink = o2::utils::Str::concat_string(mOutputDir, "/", outPrefix, "_cru_", std::to_string(mCruID), outSuffix);
    } else if (mFilePer == "halfcru") {
      outFileLink = o2::utils::Str::concat_string(mOutputDir, "/", outPrefix, "_cru_", std::to_string(mCruID), trdside, outSuffix);
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

  int triggercount = 42; // triggercount is here so that we can span timeframes. The actual number is of no consequence,but must increase.
  for (int entry = 0; entry < mTrackletsTree->GetEntries(); entry++) {
    mTrackletsTree->GetEntry(entry);
    mDigitsTree->GetEntry(entry);
    //migrate digit trigger information into the tracklettrigger (historical)
    mergetriggerDigitRanges(); // merge data (if needed) from the digits trigger record to the tracklets trigger record (different files)

    //TODO figure out if want to care about unaligned trees.
    sortDataToLinks();
    // each entry is a timeframe
    uint32_t linkcount = 0;
    for (auto tracklettrigger : mTrackletTriggerRecords) {
      //      for (auto digitstrigger : mDigitTriggerRecords) {
      //        //get the event limits from TriggerRecord;
      //        if (digitstrigger.getBCData() == tracklettrigger.getBCData()) {
      //          convertTrapData(tracklettrigger, digitstrigger, triggercount);
      //        }
      convertTrapData(tracklettrigger, triggercount); // tracklettrigger assumed to be authoritive, TODO check for blanks digits range
      //      }
      triggercount++;
    }
  }
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
    LOG(debug) << "We have data with linkdatasize=" << linksize << " with size number in header of:" << crudatasize << " padded with " << padding << " 32bit words";
  } else {
    //linksize is zero so no data, pad fully.
    crudatasize = 1;
    padding = 8;
    LOG(debug) << "We have data with linkdatasize=" << linksize << " with size number in header of:" << crudatasize << " padded with " << padding << " 32bit words";
  }
  LOG(debug) << __func__ << " leaving linkSizePadding : CRUDATASIZE : " << crudatasize;
}

uint32_t Trap2CRU::buildHalfCRUHeader(HalfCRUHeader& header, const uint32_t bc, const uint32_t halfcru)
{
  int bunchcrossing = bc;
  int stopbits = 0x01; // do we care about this and eventtype in simulations?
  int eventtype = 0x01;
  int crurdhversion = 6;
  int feeid = 0;
  int cruid = 0;
  uint32_t crudatasize = 0; //link size in units of 256 bits.
  int endpoint = halfcru % 2 ? 1 : 0;
  uint32_t padding = 0;
  //TODO this bunchcrossing is not the same as the bunchcrossing in the rdh, which is effectively the bc coming in the parameter list to this function.
  setHalfCRUHeader(header, crurdhversion, bunchcrossing, stopbits, endpoint, eventtype, feeid, cruid); //TODO come back and pull this from somewhere.

  return 1;
}

bool Trap2CRU::isTrackletOnLink(const int linkid, const int currenttrackletpos)
{
  //hcid is simply the halfcru*15+linkid
  int hcid = mTracklets[currenttrackletpos].getHCID();
  if (linkid == hcid) {
    // this tracklet is on this link.
    return true;
  }
  return false;
}

bool Trap2CRU::isDigitOnLink(const int linkid, const int currentdigitpos)
{
  Digit* digit = &mDigits[currentdigitpos];
  if ((digit->getDetector() * 2 + (digit->getROB() % 2)) == linkid) {
    return true;
  }
  return false;
}

int Trap2CRU::buildDigitRawData(const int digitindex, const std::array<int64_t, o2::trd::constants::NADCMCM>& localParsedDigits, const uint32_t triggerrecordcount)
{
  LOG(debug) << __func__ << " digitindex:" << digitindex << " and triggercount of" << triggerrecordcount;
  //this is not zero suppressed.
  int digitswritten = 0;
  //    Digit
  DigitMCMHeader header;
  DigitMCMData data;
  int startdet = mDigits[digitindex].getDetector();
  int startrob = mDigits[digitindex].getROB();
  int startmcm = mDigits[digitindex].getMCM();
  int digitcounter = 0;
  header.res = 12; //1100
  header.mcm = startmcm;
  header.rob = startrob;
  header.yearflag = 1; // >2007
  header.eventcount = triggerrecordcount;
  header.yearflag = 1;
  memcpy(mRawDataPtr, (char*)&header, 4); // 32 bits -- 4 bytes.
  LOG(debug) << "writing data to digit stream of " << std::hex << header.word;
  mRawDataPtr += 4;
  for (auto digitindex : localParsedDigits) {
    LOG(debug) << "digit indexof : " << digitindex; //<<
    if (digitindex != -1) {
      Digit* d = &mDigits[digitindex];
      ArrayADC adcdata = d->getADC();
      //write these 2 now as we only have it now.
      header.mcm = d->getMCM();
      header.rob = d->getROB();

      for (int timebin = 0; timebin < o2::trd::constants::TIMEBINS; timebin += 3) {
        LOG(debug) << "ADC values  : " << timebin << " = " << adcdata[timebin] << ":" << adcdata[timebin + 1] << ":" << adcdata[timebin + 2];
        data.x = adcdata[timebin];
        data.y = adcdata[timebin + 1];
        data.z = adcdata[timebin + 2];
        data.c = 0;
        LOG(debug) << "writing data to digit stream of " << std::hex << data.word;
        memcpy(mRawDataPtr, (char*)&data, 4); // 32 bits -- 4 bytes.
        mRawDataPtr += 4;
      }
      digitswritten++;
    } else {
      //blank digit;//10 words of 0 for the 30 timebins, its done this way incase we want to or need to change c.
      for (int timebin = 0; timebin < o2::trd::constants::TIMEBINS; timebin += 3) {
        data.x = 0;
        data.y = 0;
        data.z = 0;
        data.c = 0;
        LOG(debug) << "writing data to digit stream of " << std::hex << data.word;
        memcpy(mRawDataPtr, (char*)&data, 4); // 32 bits -- 4 bytes.
        mRawDataPtr += 4;
      }
      digitswritten++;
    }
  }
  LOG(debug) << __func__ << " leaving with digitswritten of: " << digitswritten << " for det:rob:mcm of " << startdet << ":" << startrob << ":" << startmcm;
  return digitswritten;
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
  int trackletcounter = 0;
  LOG(debug) << header;
  while (linkid == mTracklets[trackletindex + trackletcounter].getHCID() && header.col == mTracklets[trackletindex + trackletcounter].getColumn() && header.padrow == mTracklets[trackletindex + trackletcounter].getPadRow()) {
    buildTrackletMCMData(trackletdata[trackletcounter], mTracklets[trackletindex + trackletcounter].getSlope(),
                         mTracklets[trackletindex + trackletcounter].getPosition(), mTracklets[trackletindex + trackletcounter].getQ0(),
                         mTracklets[trackletindex + trackletcounter].getQ1(), mTracklets[trackletindex + trackletcounter].getQ2());
    int headerqpart = ((mTracklets[trackletindex + trackletcounter].getQ2()) << 2) + ((mTracklets[trackletindex + trackletcounter].getQ1()) >> 5);
    switch (trackletcounter) {
      case 0:
        header.pid0 = headerqpart;
        break;
      case 1:
        header.pid1 = headerqpart;
        break;
      case 2:
        header.pid2 = headerqpart;
        break;
      default:
        LOG(warn) << ">3 tracklets when building the Tracklet raw data stream for hcid=" << mTracklets[trackletindex + trackletcounter].getHCID() << " col:" << mTracklets[trackletindex + trackletcounter].getColumn() << " padrow:" << mTracklets[trackletindex + trackletcounter].getPadRow();
        destroytracklets = true;
        break;
    }
    trackletcounter++;
  }
  //now copy the mcmheader and mcmdata.
  if (!destroytracklets) {
    memcpy((char*)mRawDataPtr, (char*)&header, sizeof(TrackletMCMHeader));
    mRawDataPtr += sizeof(TrackletMCMHeader);
    for (int i = 0; i < trackletcounter; i++) {
      memcpy((char*)mRawDataPtr, (char*)&trackletdata[i], sizeof(TrackletMCMData));
      mRawDataPtr += sizeof(TrackletMCMData);
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

int Trap2CRU::writeHCHeader(const int eventcount, const uint32_t linkid)
{ // we have 2 HCHeaders defined Tracklet and Digit in Rawdata.h
  //TODO is it a case of run2 and run3 ? for digithcheader and tracklethcheader?
  //The parsing I am doing of CRU raw data only has a DigitHCHeader in it after the TrackletEndMarker ... confusion?
  int wordswritten = 0;
  //from linkid we can get supermodule, stack, layer, side
  int detector = linkid / 2;
  TrackletHCHeader trackletheader;
  trackletheader.supermodule = linkid / 60;
  trackletheader.stack = linkid;
  trackletheader.layer = linkid;
  trackletheader.one = 1;
  trackletheader.side = (linkid % 2) ? 1 : 0;
  trackletheader.MCLK = eventcount * 42; // just has to be a consistant increasing number per event.
  trackletheader.format = 12;

  DigitHCHeader digitheader;
  digitheader.res0 = 1;
  digitheader.side = (linkid % 2) ? 1 : 0;
  digitheader.stack = (detector % (o2::trd::constants::NLAYER * o2::trd::constants::NSTACK)) / o2::trd::constants::NLAYER;
  digitheader.layer = (detector % o2::trd::constants::NLAYER);
  digitheader.supermodule = linkid / 60;
  digitheader.numberHCW = 1; //TODO put something more real in here?
  digitheader.minor = 42;    //TODO put something more real in here?
  digitheader.major = 42;    //TODO put something more real in here?
  digitheader.version = 1;   //TODO put something more real in here?
  digitheader.res1 = 1;
  digitheader.ptrigcount = 1;             //TODO put something more real in here?
  digitheader.ptrigphase = 1;             //TODO put something more real in here?
  digitheader.bunchcrossing = eventcount; //NB this is not the same as the bunchcrossing the rdh.
  digitheader.numtimebins = 30;
  if (mUseTrackletHCHeader) { // run 3 we also have a TrackletHalfChamber, that comes after thetracklet endmarker.
    memcpy(mRawDataPtr, (char*)&trackletheader, sizeof(TrackletHCHeader));
    mRawDataPtr += 4;
    wordswritten++;
  }
  memcpy(mRawDataPtr, (char*)&digitheader, 8); // 8 because we are only using the first 2 32bit words of the header, the rest are optional.
  mRawDataPtr += 8;
  wordswritten += 2;
  return wordswritten;
}

void Trap2CRU::convertTrapData(o2::trd::TriggerRecord const& triggerrecord, const int& triggercount)
{
  //  LOG(info) << "starting :" << __func__ ; //<<
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
  LOG(debug) << "BUNCH CROSSING : " << triggerrecord.getBCData().bc << " with orbit : " << triggerrecord.getBCData().orbit;
  for (int halfcru = 0; halfcru < o2::trd::constants::NHALFCRU; halfcru++) {
    int halfcruwordswritten = 0;
    int supermodule = halfcru / 4; // 2 cru per supermodule.
    int endpoint = halfcru / 2;    // 2 pci end points per cru, 15 links each
    int side = halfcru % 2 ? 1 : 0;
    mFeeID = buildTRDFeeID(supermodule, side, endpoint);
    mCruID = halfcru / 2;
    mLinkID = o2::trd::constants::TRDLINKID;
    mEndPointID = halfcru % 2 ? 1 : 0;
    //15 links per half cru or cru end point.
    memset(&mRawData[0], 0, sizeof(mRawData[0]) * mRawData.size()); //   zero the rawdata storage
    int numberofdetectors = o2::trd::constants::MAXCHAMBER;
    HalfCRUHeader halfcruheader;
    //now write the cruheader at the head of all the data for this halfcru.
    buildHalfCRUHeader(halfcruheader, triggerrecord.getBCData().bc, halfcru);
    halfcruheader.EndPoint = mEndPointID;
    mRawDataPtr = rawdatavector.data();
    memcpy(mRawDataPtr, (char*)&halfcruheader, sizeof(halfcruheader));
    mRawDataPtr += sizeof(halfcruheader);
    halfcruwordswritten += sizeof(halfcruheader) / 4;
    int totallinklengths = 0;
    rawdataptratstart = mRawDataPtr; // keep track of where we started.
    for (int halfcrulink = 0; halfcrulink < o2::trd::constants::NLINKSPERHALFCRU; halfcrulink++) {
      //links run from 0 to 14, so linkid offset is halfcru*15;
      int linkid = halfcrulink + halfcru * o2::trd::constants::NLINKSPERHALFCRU;
      LOG(debug) << __func__ << " " << __LINE__ << " linkid : " << linkid << " with link " << halfcrulink << "  of halfcru " << halfcru;
      int linkwordswritten = 0;
      int errors = 0;           // put no errors in for now.
      int size = 0;             // in 32 bit words
      uint32_t paddingsize = 0; // in 32 bit words
      uint32_t crudatasize = 0; // in 256 bit words.
      //loop over tracklets for mcm's that match
      if (isTrackletOnLink(linkid, mCurrentTracklet) || isDigitOnLink(linkid, mCurrentDigit)) {
        //we have some data on this link
        int trackletcounter = 0;
        while (isTrackletOnLink(linkid, mCurrentTracklet)) {
          // still on an mcm on this link
          int tracklets = buildTrackletRawData(mCurrentTracklet, linkid); //returns # of 32 bits, header plus trackletdata words that would have come from the mcm.
          mCurrentTracklet += tracklets;
          trackletcounter += tracklets;
          linkwordswritten += tracklets + 1;
          rawwords += tracklets + 1; //1 to include the header
        }
        //write tracklet end marker
        int trackletendmarker = writeTrackletEndMarker();
        linkwordswritten += trackletendmarker;
        rawwords += trackletendmarker;
        int hcheaderwords = writeHCHeader(triggercount, linkid);
        linkwordswritten += hcheaderwords;
        rawwords += hcheaderwords;
        while (isDigitOnLink(linkid, mDigitsIndex[mCurrentDigit])) {
          //while we are on a single mcm, copy the digits timebins to the array.
          int digitcounter = 0;
          int currentROB = mDigits[mDigitsIndex[mCurrentDigit]].getROB();
          int currentMCM = mDigits[mDigitsIndex[mCurrentDigit]].getMCM();
          int currentDetector = mDigits[mDigitsIndex[mCurrentDigit]].getDetector();
          int startmCurrentDigit = mCurrentDigit;
          localParsedDigitsindex.fill(-1); // clear the digit index array
          while (mDigits[mDigitsIndex[mCurrentDigit]].getMCM() == currentMCM &&
                 mDigits[mDigitsIndex[mCurrentDigit]].getROB() == currentROB &&
                 mDigits[mDigitsIndex[mCurrentDigit]].getDetector() == currentDetector) {
            localParsedDigitsindex[mDigits[mDigitsIndex[mCurrentDigit]].getChannel()] = mDigitsIndex[mCurrentDigit];
            mCurrentDigit++;
            digitcounter++;
            if (digitcounter > 22) {
              LOG(error) << " we are on the 22nd digit of an mcm ?? This is not possible";
            }
          }
          // mcm digits are full, now write it out.
          char* preptr;
          preptr = mRawDataPtr;
          int digits = buildDigitRawData(mDigitsIndex[mCurrentDigit], localParsedDigitsindex, triggercount);
          //mCurrentDigit += digits;
          //due to not being zero suppressed, digits returned from buildDigitRawData should *always* be 21.
          if (digits != 21) {
            LOG(error) << "We are writing non zero suppressed digits yet we dont have 21 digits"; //<<
          }
          rawwords += digits * 10 + 1; //10 for the tiembins and 1 for the header.
          linkwordswritten += digits * 10 + 1;
        }
        int digitendmarkerwritten = writeDigitEndMarker();
        linkwordswritten += digitendmarkerwritten;
        rawwords += digitendmarkerwritten;

      } else {
        LOG(debug) << "no data on link : " << linkid;
      }
      //write digit end marker.
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
        //fix the halfcruheader for the length of this link.
        setHalfCRUHeaderLinkData(halfcruheader, halfcrulink, crudatasize, errors);
        uint32_t bytescopied;
        totallinklengths += crudatasize;
        if ((mRawDataPtr - rawdataptratstart) != (totallinklengths * 32)) {
          bytescopied = mRawDataPtr - rawdataptratstart;
          LOG(debug) << "something wrong with data size in cruheader writing"
                     << "linkwordswriten:" << linkwordswritten << " rawwords:" << rawwords << "bytestocopy : " << bytescopied << " crudatasize:" << crudatasize << " sum of links up to now : " << totallinklengths << " mRawDataPtr:0x" << std::hex << (void*)mRawDataPtr << "  start ptr:" << std::hex << (void*)rawdataptratstart;
          //something wrong with data size writing padbytes:81 bytestocopy : 3488 crudatasize:81 mRawDataPtr:0x0x7f669acdedf0  start ptr:0x7f669acde050

        } else {
          LOG(debug) << "all fine with data size writing padbytes:" << paddingsize << " linkwordswriten:" << linkwordswritten << " bytestocopy : " << bytescopied << " crudatasize:" << crudatasize << " mRawDataPtr:0x" << std::hex << (void*)mRawDataPtr << "  start ptr:" << std::hex << (void*)rawdataptratstart;
        }
        //sanity check for now:
        if (crudatasize != o2::trd::getlinkdatasize(halfcruheader, halfcrulink)) {
          // we have written the wrong amount of data ....
          LOG(debug) << "crudata is ! = get link data size " << crudatasize << "!=" << o2::trd::getlinkdatasize(halfcruheader, halfcrulink);
        }
        LOG(debug) << "Link words to be written : " << linkwordswritten * 4;
      } // if we have data on link
      halfcruwordswritten += linkwordswritten;
    }
    //write halfcru data here.
    std::vector<char> feeidpayload(halfcruwordswritten * 4);
    memcpy(feeidpayload.data(), &rawdatavector[0], halfcruwordswritten * 4);
    assert(halfcruwordswritten % 8 == 0);
    mWriter.addData(mFeeID, mCruID, mLinkID, mEndPointID, triggerrecord.getBCData(), feeidpayload);
    LOG(debug) << "written file for feeid of 0x" << std::hex << mFeeID << " and payload size of : " << halfcruwordswritten;
  }
}

} // end namespace trd
} // end namespace o2
