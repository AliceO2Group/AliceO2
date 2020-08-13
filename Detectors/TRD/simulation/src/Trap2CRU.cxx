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
#include "CommonUtils/StringUtils.h"
#include "TRDBase/TRDCommonParam.h"
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

using namespace o2::raw;

namespace o2
{
namespace trd
{

Trap2CRU::Trap2CRU(const std::string& outputDir, const std::string& inputFilename)
{
  readTrapData(outputDir, inputFilename, 1024 * 1024);
}

void Trap2CRU::readTrapData(const std::string& otuputDir, const std::string& inputFilename, int superPageSizeInB)
{
  //set things up, read the file and then deligate to convertTrapdata to do the conversion.
  //
  mRawData.reserve(1024 * 1024); //TODO take out the hardcoded 1MB its supposed to come in from the options
  LOG(info) << "Trap2CRU::readTrapData";
  // data comes in index by event (triggerrecord) and link (linke record) both sequentially.
  // first 15 links go to cru0a, second 15 links go to cru0b, 3rd 15 links go to cru1a ... first 90 links to flp0 and then repate for 12 flp
  // then do next event

  // lets register our links
  for (int link = 0; link < NumberOfCRU; link++) {
    mFeeID = link;
    mCruID = link;
    mEndPointID = link * 2;
    mLinkID = link;
    std::string outputFilelink = o2::utils::concat_string("trd_cru_", std::to_string(link), "_a.raw");
    mWriter.registerLink(mFeeID, mCruID, mLinkID, mEndPointID, outputFilelink);
    outputFilelink = o2::utils::concat_string("trd_cru_", std::to_string(link), "_b.raw");
    mEndPointID++;
    mWriter.registerLink(mFeeID, mCruID, mLinkID, mEndPointID, outputFilelink);
  }
  mTrapRawFile = TFile::Open(inputFilename.data());
  assert(mTrapRawFile != nullptr);
  LOG(info) << "Trap Raw file open " << inputFilename;
  mTrapRawTree = (TTree*)mTrapRawFile->Get("o2sim");

  mTrapRawTree->SetBranchAddress("TrapLinkRecord", &mLinkRecordsPtr);      // branch with the link records
  mTrapRawTree->SetBranchAddress("RawTriggerRecord", &mTriggerRecordsPtr); // branch with the trigger records
  mTrapRawTree->SetBranchAddress("TrapRaw", &mTrapRawDataPtr);             // branch with the actual incoming data.

  for (int entry = 0; entry < mTrapRawTree->GetEntries(); entry++) {
    mTrapRawTree->GetEntry(entry);
    LOG(debug) << "Before Trigger Reord loop Event starts at:" << mTriggerRecords[0].getFirstEntry() << " and has " << mTriggerRecords[0].getNumberOfObjects() << " entries";
    uint32_t linkcount = 0;
    for (auto trigger : mTriggerRecords) {
      //get the event limits from TriggerRecord;
      uint32_t eventstart = trigger.getFirstEntry();
      uint32_t eventend = trigger.getFirstEntry() + trigger.getNumberOfObjects();
      LOG(info) << "Event starts at:" << eventstart << " and ends at :" << eventend;
      convertTrapData(trigger);
    }
  }
}

void Trap2CRU::buildCRUPayLoad()
{
  // go through the data for the event in question, produce the raw stream for each cru.
  // i.e. 30 link per cru, 3cru per flp.
  // 30x [HalfCRUHeader, TrackletHCHeader0, [MCMHeader TrackletMCMData. .....] TrackletHCHeader1 ..... TrackletHCHeader30 ...]
  //
  // data must be padded into blocks of 256bit so on average 4 padding 32 bit words.
}

void Trap2CRU::linkSizePadding(uint32_t linksize, uint32_t& crudatasize, uint32_t& padding)
{
  // all data must be 256 bit aligned (8x64bit).
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
  LOG(debug) << "linkSizePadding : CRUDATASIZE : " << crudatasize;
}

uint32_t Trap2CRU::buildCRUHeader(HalfCRUHeader& header, uint32_t bc, uint32_t halfcru, int startlinkrecord)
{
  int bunchcrossing = bc;
  int stopbits = 0x01; // do we care about this and eventtype in simulations?
  int eventtype = 0x01;
  int crurdhversion = 6;
  int feeid = 0;                      //TODO must be in cruheader down the road, inside the reserved somewhere ...
  int cruid = 0;                      //TODO """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
  uint32_t crudatasize = 0;           //link size in units of 256 bits.
  int endpoint = halfcru % 2 ? 1 : 0; //TODO figure out a value ... endpoint needs a rebase to PR4106
  uint32_t padding = 0;
  //setHalfCRUHeader(halfcruheader, rdhversion, bunchcrossing, stopbits, endpoint, eventtype); //TODO come back and pull this from somewhere.
  setHalfCRUHeader(header, crurdhversion, bunchcrossing, stopbits, endpoint, eventtype, feeid, cruid); //TODO come back and pull this from somewhere.
                                                                                                       //  memset(&tmpLinkInfo[0],0,sizeof(tmpLinkInfo[0])*tmpLinkInfo.size());

  // halfcruheader from the relevant mLinkRecords.
  int linkrecord = startlinkrecord;
  int totallinkdatasize = 0; //in units of 256bits
  for (int link = 0; link < NLinksPerHalfCRU; link++) {
    int hcid = link + halfcru * NLinksPerHalfCRU; // TODO this might have to change to a lut I dont think the mapping is linear.
    int errors = 0;
    int linksize = 0; // linkSizePadding will convert it to 1 for the no data case.
    if (mLinkRecords[linkrecord].getLinkHCID() == hcid) {
      linksize = mLinkRecords[linkrecord].getNumberOfObjects();
      // this can be done differently by keeping a pointer to halfcruheader and setting it after reading it all in and going back per link to set the size.
      // LOG(info) << "setting CRU HEADER for halfcru : " << halfcru << "and link : " << link << " contents" << halfcruheader << ":" << link << ":" << linksize << ":" << errors;
      linkrecord++; // increment locally for walking through linkrecords.
    }
    linkSizePadding(linksize, crudatasize, padding);
    setHalfCRUHeaderLinkData(header, link, crudatasize, errors); // write one padding block for empty links.
    totallinkdatasize += crudatasize;
  }
  return totallinkdatasize;
}

void Trap2CRU::convertTrapData(o2::trd::TriggerRecord const& TrigRecord)
{

  //build a HalfCRUHeader for this event/cru/endpoint
  //loop over cru's
  //  loop over all half chambers, thankfully they data is sorted.
  //    check if current chamber has a link
  //      if not blank, else fill in data from link records
  //  dump data to rawwriter
  //finished for event. this method is only called per event.
  int currentlinkrecord = 0;
  char* traprawdataptr = (char*)&mTrapRawData[0];
  for (int halfcru = 0; halfcru < NumberOfHalfCRU; halfcru++) {     //TODO come back and replace 72 with something.
                                                                    //   TrackletHC
    memset(&mRawData[0], 0, sizeof(mRawData[0]) * mRawData.size()); //   zero the rawdata storage
    int numberofdetectors = o2::trd::constants::MAXCHAMBER;
    HalfCRUHeader halfcruheader;
    //now write the cruheader at the head of all the data for this halfcru.
    LOG(debug) << "cru before building cruheader for halfcru index : " << halfcru << " with contents \n"
               << halfcruheader;
    uint32_t totalhalfcrudatasize = buildCRUHeader(halfcruheader, TrigRecord.getBCData().bc, halfcru, currentlinkrecord);

    std::vector<char> rawdatavector(totalhalfcrudatasize * 32 + sizeof(halfcruheader)); // sum of link sizes + padding in units of bytes and some space for the header (512 bytes).
    char* rawdataptr = rawdatavector.data();
    LOG(info) << "before writing halfcruheader pionter is sitting at " << std::hex << static_cast<void*>(rawdataptr);
    dumpHalfCRUHeader(halfcruheader);
    memcpy(rawdataptr, (char*)&halfcruheader, sizeof(halfcruheader));
    std::array<uint64_t, 8> raw{};
    memcpy((char*)&raw[0], rawdataptr, sizeof(halfcruheader));
    for (int i = 0; i < 2; i++) {
      int index = 4 * i;
      LOGF(debug, "[1/2rawdaptr %d] 0x%08x 0x%08x 0x%08x 0x%08x", i, raw[index + 3], raw[index + 2], raw[index + 1], raw[index + 0]);
    }
    rawdataptr += sizeof(halfcruheader);
    LOG(info) << "For writing halfcruheader pionter advanced by " << std::dec << sizeof(halfcruheader) << " ptr is now at:" << std::hex << static_cast<void*>(rawdataptr);
    LOG(debug) << "Just wrote cruheader for halfcru index : " << halfcru << " with contents \n"
               << halfcruheader;
    LOG(debug) << "end of halfcruheader";

    int linkdatasize = 0; // in 32 bit words
    int link = halfcru / 2;
    int endpoint = halfcru;
    for (int halfcrulink = 0; halfcrulink < NLinksPerHalfCRU; halfcrulink++) {
      //links run from 0 to 14, so hcid offset is halfcru*15;
      int hcid = halfcrulink + halfcru * NLinksPerHalfCRU; // TODO this might have to change to a lut I dont think the mapping is linear.
      LOG(info) << "Currently checking for data on hcid : " << hcid << " from halfcru=" << halfcru << " and halfcrulink:" << halfcrulink << " ?? " << hcid << "==" << mLinkRecords[currentlinkrecord].getLinkHCID();
      int errors = 0;           // put no errors in for now.
      int size = 0;             // in 32 bit words
      int datastart = 0;        // in 32 bit words
      int dataend = 0;          // in 32 bit words
      uint32_t paddingsize = 0; // in 32 bit words
      uint32_t crudatasize = 0; // in 256 bit words.
      if (mLinkRecords[currentlinkrecord].getLinkHCID() == hcid) {
        //this link has data in the stream.
        LOG(info) << "+++ We have data on hcid = " << hcid << " halfcrulink : " << halfcrulink;
        linkdatasize = mLinkRecords[currentlinkrecord].getNumberOfObjects();
        datastart = mLinkRecords[currentlinkrecord].getFirstEntry();
        dataend = datastart + size;
        LOG(info) << "We have data on hcid = " << hcid << " and linksize : " << linkdatasize << " so :" << linkdatasize / 8 << " 256 bit words";
        currentlinkrecord++;
      } else {
        assert(mLinkRecords[currentlinkrecord].getLinkId() < hcid);
        LOG(info) << "---We do not have data on hcid = " << hcid << " halfcrulink : " << halfcrulink;
        //blank data for this link??? what do i do?
        // put in a 1 256 bit word of data for the link and padd with 0xeeee x 8
        //     tmpLinkInfo[halfcrulink] = -1;
        linkdatasize = 0;
        paddingsize = 8;
      }
      // now copy data to rawdata, padding as and where needed.
      //
      linkSizePadding(linkdatasize, crudatasize, paddingsize); //TODO this can come out as we have already called it, but previously we have lost the #padding words, solve to remove.

      LOG(info) << "WRITING " << crudatasize << " 256 bit data words to output stream";
      LOG(info) << "setting CRY HEADER for " << halfcruheader << ":" << halfcrulink << ":" << crudatasize << ":" << errors;
      // now pad ....
      LOG(info) << " now to pump data into the stream with : " << linkdatasize << " crudatasize:" << crudatasize << " paddingsize: " << paddingsize << " and rem:" << linkdatasize % 8;
      char* olddataptr = rawdataptr; // store the old pointer so we can do some sanity checks for how far we advance.
      //linkdatasize is the #of 32 bit words coming from the incoming tree.
      //paddingsize is the number of padding words to add 0xeeee
      uint32_t bytestocopy = linkdatasize * (sizeof(uint32_t));
      LOG(info) << "copying " << bytestocopy << " bytes of link tracklet data at pos:" << std::hex << static_cast<void*>(rawdataptr);
      memcpy(rawdataptr, traprawdataptr, bytestocopy);
      //increment pointer
      rawdataptr += bytestocopy;
      traprawdataptr += bytestocopy;
      //now for padding
      uint16_t padbytes = paddingsize * sizeof(uint32_t);
      LOG(info) << "writing " << padbytes << " bytes of padding data  at pos:" << std::hex << static_cast<void*>(rawdataptr);
      memset(rawdataptr, 0xee, padbytes);
      //increment pointer.
      rawdataptr += padbytes;
      if (padbytes + bytestocopy != crudatasize * 32) {
        LOG(info) << "something wrong with data size writing padbytes:" << padbytes << " bytestocopy : " << bytestocopy << " crudatasize:" << crudatasize;
      }
      LOG(debug3) << std::hex << " rawdataptr:" << static_cast<void*>(rawdataptr) << " traprawdataptr " << static_cast<void*>(traprawdataptr);
      // printf(std::cout << std::hex << " rawdataptr:0x"<<&rawdataptr << " traprawdataptr 0x"<<&traprawdataptr << std::endl;
      //sanity check for now:
      if (((char*)rawdataptr - (char*)olddataptr) != crudatasize * 32) { // cru words are 8 uint32 and comparison is in bytes.
        LOG(debug) << "according to pointer arithmatic we have added " << rawdataptr - olddataptr << "bytes from " << static_cast<void*>(rawdataptr) << "-" << static_cast<void*>(olddataptr) << " when we should have added  " << crudatasize * 8 * 4 << " because crudatasize=" << crudatasize;
      }
      if (crudatasize != o2::trd::getlinkdatasize(halfcruheader, halfcrulink)) {
        // we have written the wrong amount of data ....
        LOG(debug) << "crudata is ! = get link data size " << crudatasize << "!=" << o2::trd::getlinkdatasize(halfcruheader, halfcrulink);
      }
      //    if(crudatasize ==0)LOG(info) << "CRUDATASIZ EIS ZERO" ;
      //      if(crudatasize>0){
      //      std::ofstream out1("crutestdumphalfcrurawdata");
      //      std::ofstream out2("crutestdumprawdatavector");
      //      LOG(info) << "writing out " << halfcrurawdata.size() << " and"  << crudatasize*32;
      //      out1.write(halfcrurawdata.data(),halfcrurawdata.size());
      //      out2.write(rawdatavector.data(),crudatasize*32);
      //      out1.flush();
      //      out2.flush();
      //      exit(1);
      //      }
      LOG(info) << "copied " << crudatasize * 32 << "bytes to halfcrurawdata which now has  size of " << rawdatavector.size() << " for " << link << ":" << endpoint;
    }
    // std::vector<char> halfcrurawdata(crudatasize * 32); // vector of 256bit words * 32 to get to 8 bit words

    mWriter.addData(link, link, link, endpoint, TrigRecord.getBCData(), rawdatavector);
    std::ofstream out2("crutestdumprawdatavector");
    out2.write(rawdatavector.data(), rawdatavector.size());
    out2.flush();
    //    halfcru = NumberOfHalfCRU; // exit loop after 1 half cru for now.
  }
}

} // end namespace trd
} // end namespace o2
