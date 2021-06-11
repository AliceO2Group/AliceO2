// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   CruRawReader.h
/// @brief  TRD raw data translator

#include "DetectorsRaw/RDHUtils.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "Headers/RDHAny.h"
#include "TRDReconstruction/CruRawReader.h"
#include "TRDBase/FeeParam.h"
#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "TRDReconstruction/DigitsParser.h"
#include "TRDReconstruction/TrackletsParser.h"
#include "DataFormatsTRD/Constants.h"

#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <iostream>
#include <numeric>
#include <iostream>

namespace o2::trd
{

bool CruRawReader::skipRDH()
{
  // check rdh for being empty or only padding words.
  if (o2::raw::RDHUtils::getMemorySize(mOpenRDH) == o2::raw::RDHUtils::getHeaderSize(mOpenRDH)) {
    //empty rdh so we want to avoid parsing it for cru data.
    LOG(info) << " skipping rdh (empty) with packetcounter of: " << std::hex << o2::raw::RDHUtils::getPacketCounter(mOpenRDH);
    return true;
  } else {

    if (mHBFPayload[0] == o2::trd::constants::CRUPADDING32 && mHBFPayload[0] == o2::trd::constants::CRUPADDING32) {
      //event only contains paddings words.
      LOG(info) << " skipping rdh (padding) with packetcounter of: " << std::hex << o2::raw::RDHUtils::getPacketCounter(mOpenRDH);
      // mDataPointer+= o2::raw::RDHUtils::getOffsetToNext()/4;
      auto rdh = reinterpret_cast<const o2::header::RDHAny*>(mDataPointer);
      mDataPointer += o2::raw::RDHUtils::getOffsetToNext(rdh) / 4;
      //mDataPointer=reinterpret_cast<const uint32_t*>(reinterpret_cast<const char*>(rdh) + o2::raw::RDHUtils::getOffsetToNext(rdh));
      return true;
      return true;
    } else {
      return false;
    }
  }
}

bool CruRawReader::processHBFs(int datasizealreadyread, bool verbose)
{
  if (mVerbose) {
    LOG(info) << "PROCESS HBF starting at " << std::hex << (void*)mDataPointer;
    LOG(info) << "PROCESS HBF from payload starting at " << std::hex << (void*)&mHBFPayload[0];
  }
  mDataRDH = reinterpret_cast<const o2::header::RDHAny*>(mDataPointer);
  mOpenRDH = reinterpret_cast<o2::header::RDHAny*>((char*)mDataPointer);
  auto rdh = mDataRDH;
  auto preceedingrdh = rdh;
  uint64_t totaldataread = 0;
  mState = CRUStateHalfCRUHeader;
  uint32_t currentsaveddatacount = 0;
  mTotalHBFPayLoad = 0;
  int loopcount = 0;
  // loop until RDH stop header
  while (!o2::raw::RDHUtils::getStop(rdh)) { // carry on till the end of the event.
    if (mVerbose) {
      LOG(info) << "--- RDH open/continue detected loopcount :" << loopcount;
      LOG(info) << " rdh first word 0x" << std::hex << (uint32_t)*mDataPointer;

      //      o2::raw::RDHUtils::printRDH(rdh);
      for (int i = 0; i < 64; ++i) {
        LOG(info) << std::hex << " 0x" << *(mDataPointer + i);
      }
      LOG(info) << "---------------------- parsing that rdh";
    }
    preceedingrdh = rdh;
    auto headerSize = o2::raw::RDHUtils::getHeaderSize(rdh);
    auto memorySize = o2::raw::RDHUtils::getMemorySize(rdh);
    auto offsetToNext = o2::raw::RDHUtils::getOffsetToNext(rdh);
    auto rdhpayload = memorySize - headerSize;
    mFEEID = o2::raw::RDHUtils::getFEEID(rdh);            //TODO change this and just carry around the curreht RDH
    mCRUEndpoint = o2::raw::RDHUtils::getEndPointID(rdh); // the upper or lower half of the currently parsed cru 0-14 or 15-29
    mCRUID = o2::raw::RDHUtils::getCRUID(rdh);
    auto packetCount = o2::raw::RDHUtils::getPacketCounter(rdh);
    o2::InteractionRecord a = o2::raw::RDHUtils::getTriggerIR(rdh);
    mIR = a;
    //mDataPointer += headerSize/4;
    mDataEndPointer = (const uint32_t*)((char*)rdh + offsetToNext);
    // copy the contents of the current rdh into the buffer to be parsed
    std::memcpy((char*)&mHBFPayload[0] + currentsaveddatacount, reinterpret_cast<const char*>(rdh) + headerSize, rdhpayload);
    mTotalHBFPayLoad += rdhpayload;
    currentsaveddatacount += rdhpayload;
    totaldataread += offsetToNext;
    // move to next rdh
    rdh = reinterpret_cast<const o2::header::RDHAny*>(reinterpret_cast<const char*>(rdh) + offsetToNext);
    if ((char*)(rdh) < (char*)&mHBFPayload[0] + mDataBufferSize) {
      //if (reinterpret_cast<const o2::header::RDHAny*>(rdh) < (char*)&mHBFPayload[0] + mDataBufferSize) {
      if (mVerbose) {
        LOG(info) << __func__ << " " << __LINE__;
        LOG(info) << "rdh position is still inside the buffer";
        LOG(info) << __func__ << " " << __LINE__;
        LOG(info) << "0x;" << std::hex << (void*)rdh;
        LOG(info) << "0x;" << std::hex << (void*)rdh;
        LOG(info) << "0x;" << std::hex << (void*)rdh;
        LOG(info) << "0x;" << std::hex << (void*)rdh;
        LOG(info) << "0x;" << std::hex << (void*)rdh;
        LOG(info) << "0x;" << std::hex << (void*)rdh;
        LOG(info) << "0x;" << std::hex << (void*)rdh;
        //        o2::raw::RDHUtils::printRDH(rdh);
        LOG(info) << __func__ << " " << __LINE__;
      }
      // we can still copy into this buffer.
    } else {
      LOG(warn) << "next rdh exceeds the bounds of the cru payload buffer";
      if (mVerbose) {
        LOG(info) << "rdh position  is out of bounds of the buffer";
        o2::raw::RDHUtils::printRDH(rdh);
      }
      return false; //-1;
    }
  }
  //increment the data pointer by the size of the stop rdh.
  mDataPointer = reinterpret_cast<const uint32_t*>(reinterpret_cast<const char*>(rdh) + o2::raw::RDHUtils::getOffsetToNext(rdh)); //rdh->offsetToNext);//o2::raw::RDHUtils::getOffsetToNext(rdh); // jump over the stop rdh that kicked us out of the loop
  // at this point the entire HBF data payload is sitting in mHBFPayload and the total data count is mTotalHBFPayLoad
  int counthalfcru = 0;
  mHBFoffset32 = 0;

  while ((mHBFoffset32 < ((mTotalHBFPayLoad) / 4))) { //} && mTotalHBFPayLoad>0) { // need at least a complete half cru header else we are in an error condition any case.
    if (mVerbose) {
      LOG(info) << "Looping over cruheaders in HBF, loop count " << counthalfcru << " current offset is" << mHBFoffset32 << " total payload is " << mTotalHBFPayLoad / 4 << "  raw :" << mTotalHBFPayLoad;
    }
    int halfcruprocess = processHalfCRU(mHBFoffset32);
    if (mVerbose) {
      switch (halfcruprocess) {
        case -1:
          LOG(info) << "ignored rdh event ";
          break;
        case 0:
          LOG(fatal) << "figure out what now";
          break;
        case 1:
          LOG(info) << "all good parsing half cru";
          break;
      }
    }
    counthalfcru++;
  } // loop of halfcru's while there is still data in the heart beat frame.
  mDatareadfromhbf = totaldataread;
  return true; //totaldataread;
}

int CruRawReader::processHalfCRU(int cruhbfstartoffset)
{
  if (mVerbose) {
    LOG(info) << "************************ parsing HALFCRU starting at " << cruhbfstartoffset;
  }
  //It will clean this code up *alot*
  // process a halfcru
  // or continue with the remainder of an rdh o2 payload if we got to the end of cru
  // or continue with a new rdh payload if we are not finished with the last cru payload.
  // TODO the above 2 lines are not possible.
  uint32_t currentlinkindex = 0;
  uint32_t currentlinkoffset = 0;
  uint32_t currentlinksize = 0;
  uint32_t currentlinksize32 = 0;
  uint32_t linksizeAccum32 = 0;
  uint32_t sumtrackletwords = 0;
  uint32_t sumdigitwords = 0;
  uint32_t sumlinklengths = 0;
  int trackletwordsread = 0; // this will read up to the tracnklet end marker.
  int digitwordsread = 0;
  uint32_t cruwordsread = 9;
  //reject halfcru if it starts with padding words.
  //this should only hit that instance where the cru payload is a "blank event" of o2::trd::constants::CRUPADDING32
  if (mHBFPayload[cruhbfstartoffset] == o2::trd::constants::CRUPADDING32 && mHBFPayload[cruhbfstartoffset + 1] == o2::trd::constants::CRUPADDING32) {
    //        LOG(info) << "A###############################################################################################################";
    return -1;
  }
  if (mTotalHBFPayLoad == 0) {
    //empty payload
    return -1;
  }
  // well then read the halfcruheader.
  memcpy((char*)&mCurrentHalfCRUHeader, (void*)(&mHBFPayload[cruhbfstartoffset]), sizeof(mCurrentHalfCRUHeader)); //TODO remove the copy just use pointer dereferencing, doubt it will improve the speed much though.

  //check the bunch crossings match .... they dont!
  //if (mCurrentHalfCRUHeader.BunchCrossing != mIR.bc) {
  //  LOG(warn) << " BC mismatch rdh!=cru1/2header : " << mIR.bc << " != " << mCurrentHalfCRUHeader.BunchCrossing;
  //  printHalfCRUHeader(mCurrentHalfCRUHeader);
  // }
  o2::trd::getlinkdatasizes(mCurrentHalfCRUHeader, mCurrentHalfCRULinkLengths);
  o2::trd::getlinkerrorflags(mCurrentHalfCRUHeader, mCurrentHalfCRULinkErrorFlags);
  mTotalHalfCRUDataLength256 = std::accumulate(mCurrentHalfCRULinkLengths.begin(),
                                               mCurrentHalfCRULinkLengths.end(),
                                               decltype(mCurrentHalfCRULinkLengths)::value_type(0));
  mTotalHalfCRUDataLength = mTotalHalfCRUDataLength256 * 32; //convert to bytes.
  std::array<uint32_t, 1024>::iterator currentlinkstart = mHBFPayload.begin() + cruhbfstartoffset;
  std::array<uint32_t, 1024>::iterator linkstart, linkend;
  int dataoffsetstart32 = sizeof(mCurrentHalfCRUHeader) / 4 + cruhbfstartoffset; // in uint32
  //CHECK 1 does rdh endpoint match cru header end point.
  if (mCRUEndpoint != mCurrentHalfCRUHeader.EndPoint) {
    LOG(warn) << " Endpoint mismatch : CRU Half chamber header endpoint = " << mCurrentHalfCRUHeader.EndPoint << " rdh end point = " << mCRUEndpoint;
    //TODO increment histogram bin.
    if (mVerbose) {
      LOG(info) << "******* LINK # " << currentlinkindex;
    }
  }

  if (mDataVerbose) {
    printHalfCRUHeader(mCurrentHalfCRUHeader);
    for (int i = 0; i < 16; ++i) {
      LOG(info) << std::hex << " 0x" << mHBFPayload[mHBFoffset32 + i];
    }
    for (auto t : mCurrentHalfCRULinkLengths) {
      if (t > 100) {
        LOG(info) << " errorin lengthof cru link";
      }
    }
    LOG(info) << "end halfcrudump";
  }
  // verify cru header vs rdh header
  //FEEID has supermodule/layer/stack/side in it.
  //CRU has
  mHBFoffset32 += sizeof(mCurrentHalfCRUHeader) / 4;
  linkstart = mHBFPayload.begin() + dataoffsetstart32;
  linkend = mHBFPayload.begin() + dataoffsetstart32;
  //loop over links
  for (currentlinkindex = 0; currentlinkindex < constants::NLINKSPERHALFCRU; currentlinkindex++) {
    currentlinksize = mCurrentHalfCRULinkLengths[currentlinkindex];
    currentlinksize32 = currentlinksize * 8; //x8 to go from 256 bits to 32 bit;
    linkstart = mHBFPayload.begin() + dataoffsetstart32 + linksizeAccum32;
    linkend = linkstart + currentlinksize32;
    linksizeAccum32 += currentlinksize32;
    int supermodule = ((TRDFeeID*)&mFEEID)->supermodule;
    int endpoint = ((TRDFeeID*)&mFEEID)->endpoint;
    int side = ((TRDFeeID*)&mFEEID)->side;
    //stack layer and side map to ori
    int stack, layer, oriside;
    int oriindex = currentlinkindex + constants::NLINKSPERHALFCRU * endpoint; // side denotes the pci side, upper or lower for the pair of 15 fibres.
    FeeParam::unpackORI(oriindex, side, stack, layer, oriside);
    int currentdetector = stack + layer + oriside;
    int ori = 1;
    if (mVerbose) {
      LOG(info) << "******* LINK # " << currentlinkindex << " and ORI:" << ori;
    }
    // tracklet first then digit ??
    // tracklets end with tracklet end marker(0x10001000 0x10001000), digits end with digit endmarker (0x0 0x0)
    if (linkstart != linkend) { // if link is not empty
      //        LOG(info) << "linkstart != linkend";

      if (mVerbose) {
        LOG(info) << "parse tracklets ";
      }
      bool cleardigits = false; //linkstart and linkend already have the multiple cruheaderoffsets built in
      if (mVerbose) {
        LOG(info) << "mem copy with offset of : " << cruhbfstartoffset << " parsing tracklets with linkstart: " << linkstart << " ending at : " << linkend;
      }
      trackletwordsread = mTrackletsParser.Parse(&mHBFPayload, linkstart, linkend, mFEEID, oriside, currentdetector, stack, layer, cleardigits, mByteSwap, mVerbose, mHeaderVerbose, mDataVerbose); // this will read up to the tracnklet end marker.
      if (mVerbose) {
        LOG(info) << "trackletwordsread:" << trackletwordsread << "  mem copy with offset of : " << cruhbfstartoffset << " parsing with linkstart: " << linkstart << " ending at : " << linkend;
      }
      linkstart += trackletwordsread;
      //now we have a tracklethcheader and a digithcheader.
      mHBFoffset32 += trackletwordsread;
      mTotalTrackletsFound += mTrackletsParser.getTrackletsFound();
      digitwordsread = 0;
      if (mVerbose) {
        LOG(info) << "parse digits";
      }
      //linkstart and linkend already have the multiple cruheaderoffsets built in
      if (mVerbose) {
        LOG(info) << "mem copy with offset of : " << cruhbfstartoffset << " parsing digits with linkstart: " << linkstart << " ending at : " << linkend;
      }
      digitwordsread = mDigitsParser.Parse(&mHBFPayload, linkstart, linkend, currentdetector, cleardigits, mByteSwap, mVerbose, mHeaderVerbose, mDataVerbose);
      mTotalDigitsFound += mDigitsParser.getDigitsFound();
      if (mVerbose) {
        LOG(info) << "digitwordsread : " << digitwordsread << " mem copy with offset of : " << cruhbfstartoffset << " parsing digits with linkstart: " << linkstart << " ending at : " << linkend;
      }
      sumlinklengths += mCurrentHalfCRULinkLengths[currentlinkindex];
      sumtrackletwords = trackletwordsread;
      sumdigitwords = digitwordsread;
      mHBFoffset32 += digitwordsread; // all 3 in 32bit units
      if (mDataVerbose) {
        LOG(info) << "After parsing digits digitwordsread:" << digitwordsread << " trackletwordsread:" << trackletwordsread;
        LOG(info) << " pointer content is :0x" << std::hex << mHBFPayload[mHBFoffset32 - 5];
        LOG(info) << " data pointer content is :0x" << std::hex << mHBFPayload[mHBFoffset32 - 4];
        LOG(info) << " data pointer content is :0x" << std::hex << mHBFPayload[mHBFoffset32 - 3];
        LOG(info) << " data pointer content is :0x" << std::hex << mHBFPayload[mHBFoffset32 - 2];
        LOG(info) << " data pointer content is :0x" << std::hex << mHBFPayload[mHBFoffset32 - 1];
        LOG(info) << "Current data pointer after coming back from digit parsing has content :0x" << std::hex << mHBFPayload[mHBFoffset32];
        LOG(info) << " data pointer content is :0x" << std::hex << mHBFPayload[mHBFoffset32 + 1];
        LOG(info) << " data pointer content is :0x" << std::hex << mHBFPayload[mHBFoffset32 + 2];
        LOG(info) << " data pointer content is :0x" << std::hex << mHBFPayload[mHBFoffset32 + 3];
        LOG(info) << "After parsing digits sumdigitwords:" << sumdigitwords << " sumtrackletwords:" << sumtrackletwords;
      }
    } else {
      if (mVerbose) {
        LOG(info) << "link start and end are the same, link appears to be empty for link currentlinkdex";
      }
    }
  } //for loop over link index.
  // we have read in all the digits and tracklets for this event.
  //digits and tracklets are sitting inside the parsing classes.
  //extract the vectors and copy them to tracklets and digits here, building the indexing(triggerrecords)
  //as this is for a single cru half chamber header all the tracklets and digits are for the same trigger defined by the bc and orbit in the rdh which we hold in mIR
  mIR.bc = mCurrentHalfCRUHeader.BunchCrossing; // correct mIR to have the physics trigger bunchcrossing *NOT* the heartbeat trigger bunch crossing.

  std::vector<Tracklet64> ta;
  ta = mTrackletsParser.getTracklets();

  //LOG(info) << "adding tracklets from trackletparser with a size of:" << mTrackletsParser.getTracklets().size();
  //mEventRecords.addTracklets(mIR, mTrackletsParser.getTrackletsbegin(), mTrackletsParser.getTrackletsend());
  //mEventRecords.addTracklets(mIR, std::begin(mTrackletsParser.getTracklets()), std::end(mTrackletsParser.getTracklets()));
  mEventRecords.addTracklets(mIR, mTrackletsParser.getTracklets());
  if (mVerbose) {
    LOG(info) << "inserting tracklets from parser of size : " << mTrackletsParser.getTracklets().size() << " mEventRecordsTracklets is now :" << mEventRecords.sumTracklets();
  }
  //LOG(info) << "added tracklets from trackletparser with a size of:" << mTrackletsParser.getTracklets().size();
  mTrackletsParser.clear();
  mEventRecords.addDigits(mIR, std::begin(mDigitsParser.getDigits()), std::end(mDigitsParser.getDigits()));
  if (mVerbose) {
    LOG(info) << "inserting digits from parser of size : " << mDigitsParser.getDigits().size();
  }
  mDigitsParser.clear();
  if (mVerbose) {
    LOG(info) << "Event digits after eventi # : " << mEventRecords.sumDigits() << " having added : via sum=" << mDigitsParser.getDigits().size() << " digitsfound is " << mDigitsParser.getDigitsFound();
  }
  int lasttrigger = 0, lastdigit = 0, lasttracklet = 0;
  /*if (mEventTriggers.size() != 0) {
    lasttrigger = mEventTriggers.size() - 1;
    lastdigit = mEventTriggers[lasttrigger].getFirstDigit() + mEventTriggers[lasttrigger].getNumberOfDigits();
    lasttracklet = mEventTriggers[lasttrigger].getFirstTracklet() + mEventTriggers[lasttrigger].getNumberOfTracklets();
  }
  LOG(info) << mIR << " digits : " << mEventDigits.size() << " tracklets : " << mEventTracklets.size();
  mEventTriggers.emplace_back(mIR, lastdigit, mEventDigits.size(), lasttracklet, mEventTracklets.size());
  */
  // now handled internall in mEventRecords
  //if we get here all is ok.
  return 1;
}

bool CruRawReader::buildCRUPayLoad()
{
  // copy data for the current half cru, and when we eventually get to the end of the payload return 1
  // to say we are done.
  int cruid = 0;
  int additionalBytes = -1;
  int crudatasize = -1;
  LOG(info) << "--- Build CRU Payload, added " << additionalBytes << " bytes to CRU "
            << cruid << " with new size " << crudatasize;
  return true;
}

bool CruRawReader::processCRULink()
{
  /* process a CRU Link 15 per half cru */
  //  checkFeeID(); // check the link we are working with corresponds with the FeeID we have in the current rdh.
  //  uint32_t slotId = GET_TRMDATAHEADER_SLOTID(*mDataPointer);
  return false;
}

void CruRawReader::resetCounters()
{
  mEventCounter = 0;
  mFatalCounter = 0;
  mErrorCounter = 0;
}

void CruRawReader::checkSummary()
{
  char chname[2] = {'a', 'b'};

  LOG(info) << "--- SUMMARY COUNTERS: " << mEventCounter << " events "
            << " | " << mFatalCounter << " decode fatals "
            << " | " << mErrorCounter << " decode errors ";
}

bool CruRawReader::run()
{
  uint32_t dowhilecount = 0;
  uint64_t totaldataread = 0;
  rewind();
  uint32_t* bufferptr;
  bufferptr = (uint32_t*)mDataBuffer;
  do {
    if (mDataVerbose) {
      LOG(info) << " mDataBuffer :" << (void*)mDataBuffer << " and offset to start on is :" << totaldataread;
    }
    //int mDatareadfromhbf = processHBFs(totaldataread, mVerbose);
    mDatareadfromhbf = 0;
    processHBFs(totaldataread, mVerbose);
    totaldataread += mDatareadfromhbf;
    if (mDataVerbose) {
      LOG(info) << "end with " << mDatareadfromhbf << " total data read : " << totaldataread;
      LOG(info) << " about to end do while with " << (void*)mDataPointer << "-" << (void*)bufferptr << " < " << mDataBufferSize;
    }
  } while (((char*)mDataPointer - mDataBuffer) < mDataBufferSize);
  return false;
};

void CruRawReader::getParsedObjects(std::vector<Tracklet64>& tracklets, std::vector<Digit>& digits, std::vector<TriggerRecord>& triggers)
{
  int digitcountsum = 0;
  int trackletcountsum = 0;
  mEventRecords.unpackDataForSending(triggers, tracklets, digits);
  /*for(auto eventrecord: mEventRecords)//loop over triggers incase they have already been done.
  {
  int digitcount=0;
  int trackletcount=0;
    int start,end;
    LOG(info) << __func__ << " " << tracklets.size() << " "
              << cdigits.size()<< " trackletv size:"<< mEventRecords.getTracklets(ir.getBCData());
    for(auto trackletv: mEventStores.getTracklets(ir.getBCData())){
      //loop through the vector of ranges
      start=trackletv.getFirstEntry();
      end= start+trackletv.getEntries();
      LOG(info) << "insert tracklets from " << start<< "  " << end;
      tracklets.insert(tracklets.end(),mEventTracklets.begin()+start, mEventTracklets.begin()+end);
      trackletcount+=trackletv.getEntries();
    }
    for(auto digitv: mEventStores.getDigits(ir.getBCData())){
      start=digitv.getFirstEntry();
      end= start+digitv.getEntries();
      LOG(info) << "insert digits from " << start<< "  " << end;
      cdigits.insert(cdigits.end(),mEventCompressedDigits.begin()+start , mEventCompressedDigits.begin()+end);
      digitcount+=digitv.getEntries();
    }
    triggers.emplace_back(ir.getBCData(),digitcountsum,digitcount,trackletcountsum,trackletcount);
    digitcountsum+=digitcount;
    trackletcountsum+=trackletcount;
  }*/
}
} // namespace o2::trd
