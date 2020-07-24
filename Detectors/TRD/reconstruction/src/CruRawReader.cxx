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
#include "Headers/RDHAny.h"
#include "TRDReconstruction/CruRawReader.h"
#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "TRDReconstruction/DigitsParser.h"
#include "TRDReconstruction/TrackletsParser.h"
#include "DataFormatsTRD/Constants.h"
//#include "DataFormatsTRD/CompressedDigit.h"

#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <iostream>
#include <numeric>
#include <iostream>

namespace o2
{
namespace trd
{


bool CruRawReader::skipRDH()
{
// check rdh for being empty or only padding words.
  if(o2::raw::RDHUtils::getMemorySize(mOpenRDH)== o2::raw::RDHUtils::getHeaderSize(mOpenRDH)) {
      //empty rdh so we want to avoid parsing it for cru data.
      LOG(info) << " skipping rdh (empty) with packetcounter of: " <<std::hex <<  o2::raw::RDHUtils::getPacketCounter(mOpenRDH);
      return true;
  }
  else {

      if(mCRUPayLoad[0]==o2::trd::constants::CRUPADDING32 && mCRUPayLoad[0]==o2::trd::constants::CRUPADDING32)  {
          //event only contains paddings words.
      LOG(info) << " skipping rdh (padding) with packetcounter of: " << std::hex <<o2::raw::RDHUtils::getPacketCounter(mOpenRDH);
     // mDataPointer+= o2::raw::RDHUtils::getOffsetToNext()/4;
      auto rdh = reinterpret_cast<const o2::header::RDHAny*>(mDataPointer);
      mDataPointer+= o2::raw::RDHUtils::getOffsetToNext(rdh)/4;
      //mDataPointer=reinterpret_cast<const uint32_t*>(reinterpret_cast<const char*>(rdh) + o2::raw::RDHUtils::getOffsetToNext(rdh));
      return true;
          return true;
      }
      else return false;
  }
}

bool CruRawReader::processHBFs(int datasizealreadyread, bool verbose)
{
    LOG(info) << "PROCESS HBF starting at " << std::hex << (void*)mDataPointer;

    mDataRDH = reinterpret_cast<const o2::header::RDHAny*>(mDataPointer);
    mOpenRDH = reinterpret_cast<o2::header::RDHAny*>((char*)mDataPointer);
    auto rdh = mDataRDH;
    auto preceedingrdh=rdh;
    uint64_t totaldataread = 0;
    mState = CRUStateHalfCRUHeader;
    uint32_t currentsaveddatacount=0;
    mTotalCRUPayLoad=0;
    // loop until RDH stop header 
    while (!o2::raw::RDHUtils::getStop(rdh)) { // carry on till the end of the event.
        LOG(info) << "--- RDH open/continue detected";
        o2::raw::RDHUtils::printRDH(rdh);
        LOG(info) << "--- parsing that rdh";
        preceedingrdh=rdh;
        auto headerSize = o2::raw::RDHUtils::getHeaderSize(rdh);
        auto memorySize = o2::raw::RDHUtils::getMemorySize(rdh);
        auto offsetToNext = o2::raw::RDHUtils::getOffsetToNext(rdh);
        auto cruPayLoad = memorySize - headerSize;
        mFEEID = o2::raw::RDHUtils::getFEEID(rdh);            //TODO change this and just carry around the curreht RDH
        mCRUEndpoint = o2::raw::RDHUtils::getEndPointID(rdh); // the upper or lower half of the currently parsed cru 0-14 or 15-29
        mCRUID = o2::raw::RDHUtils::getCRUID(rdh);
        auto packetCount = o2::raw::RDHUtils::getPacketCounter(rdh);
        //mDataPointer += headerSize/4;
        mDataEndPointer = (const uint32_t*)((char*)rdh + offsetToNext);
        // copy the contents of the current rdh into the buffer to be parsed
        std::memcpy((char*)&mCRUPayLoad[0]+currentsaveddatacount, reinterpret_cast<const char*>(rdh)+headerSize,cruPayLoad);
        mTotalCRUPayLoad+=cruPayLoad;
        currentsaveddatacount+= cruPayLoad;
        totaldataread+=offsetToNext;
        // move to next rdh
        rdh = reinterpret_cast<const o2::header::RDHAny*>(reinterpret_cast<const char*>(rdh) + offsetToNext);
        o2::raw::RDHUtils::printRDH(rdh);
        if(reinterpret_cast<const o2::header::RDHAny*>(rdh) < (void*)&mCRUPayLoad[0] + mDataBufferSize){
            ;//LOG(info) << "E***************************";
            // we can still copy into this buffer. 
        }
        else {
            LOG(warn) << "next rdh exceeds the bounds of the cru payload buffer";
            return false;//-1;
        }


    }
    //increment the data pointer by the size of the stop rdh.
    mDataPointer=reinterpret_cast<const uint32_t*>(reinterpret_cast<const char*>(rdh) + o2::raw::RDHUtils::getOffsetToNext(rdh));//rdh->offsetToNext);//o2::raw::RDHUtils::getOffsetToNext(rdh); // jump over the stop rdh that kicked us out of the loop
    int halfcruprocess=processHalfCRU();
    if(mVerbose){
        switch(halfcruprocess) {
            case -1 : LOG(info) << "ignored rdh event ";break;
            case 0 : LOG(fatal) << "figure out what now";break;
            case 1 : LOG(info) << "all good parsing half cru";break;
        }
    }
    datareadfromhbf=totaldataread;
    return true;//totaldataread;
}

int CruRawReader::processHalfCRU()
{
    LOG(info)   <<"************************ HALFCRU with a payload of :" << mTotalCRUPayLoad;
    //TODO this should be done external to the getStop loop as getStop loop (line 46) will end up with a singular event buffer.
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
    uint32_t sumtrackletwords=0;
    uint32_t sumdigitwords=0;
    uint32_t sumlinklengths=0;
    int trackletwordsread=0; // this will read up to the tracnklet end marker.
    int digitwordsread = 0;
    //reject halfcru if it starts with padding words.
    //this should only hit that instance where the cru payload is a "blank event" of o2::trd::constants::CRUPADDING32
    if(mCRUPayLoad[0] == o2::trd::constants::CRUPADDING32 && mCRUPayLoad[1]== o2::trd::constants::CRUPADDING32){
        //        LOG(info) << "A###############################################################################################################";
        return -1;
    }
    if(mTotalCRUPayLoad==0){
        //empty payload
        return -1;
    }
    // well then read the halfcruheader.
    memcpy((char*)&mCurrentHalfCRUHeader, (void*)(&mCRUPayLoad[0]), sizeof(mCurrentHalfCRUHeader)); //TODO remove the copy just use pointer dereferencing, doubt it will improve the speed much though.
    o2::trd::getlinkdatasizes(mCurrentHalfCRUHeader, mCurrentHalfCRULinkLengths);
    o2::trd::getlinkerrorflags(mCurrentHalfCRUHeader, mCurrentHalfCRULinkErrorFlags);
    mTotalHalfCRUDataLength256 = std::accumulate(mCurrentHalfCRULinkLengths.begin(),
            mCurrentHalfCRULinkLengths.end(),
            decltype(mCurrentHalfCRULinkLengths)::value_type(0));
    mTotalHalfCRUDataLength = mTotalHalfCRUDataLength256 * 32; //convert to bytes.
    std::array<uint32_t, 1024>::iterator currentlinkstart = mCRUPayLoad.begin();
    std::array<uint32_t, 1024>::iterator linkstart, linkend;
    int dataoffsetstart32=sizeof(mCurrentHalfCRUHeader)/4; // in uint32
    linkstart = mCRUPayLoad.begin() + dataoffsetstart32;
    linkend = mCRUPayLoad.begin()+ dataoffsetstart32;
    //loop over links
    for (currentlinkindex = 0; currentlinkindex < 15; currentlinkindex++) {
        if(mVerbose)LOG(info) << "******* LINK # " << currentlinkindex;
        currentlinksize = mCurrentHalfCRULinkLengths[currentlinkindex];
        currentlinksize32 = currentlinksize * 8; //x8 to go from 256 bits to 32 bit units;
        linkstart = mCRUPayLoad.begin() + dataoffsetstart32+ linksizeAccum32;
        linkend = linkstart + currentlinksize;
        linksizeAccum32 += currentlinksize32;
        int currentdetector = 1; // TODO fix this based on the above data.

        // tracklet first then digit ??
        // tracklets end with tracklet end marker(0x10001000 0x10001000), digits end with digit endmarker (0x0 0x0)
        if (linkstart != linkend) { // if link is not empty
            //        LOG(info) << "linkstart != linkend";
            mTrackletsParser.setVerbose(mVerbose);
            LOG(info) << "parse tracklets ";
            trackletwordsread = mTrackletsParser.Parse(&mCRUPayLoad,linkstart,linkend,currentdetector); // this will read up to the tracnklet end marker.
            linkstart += trackletwordsread;

            digitwordsread = 0;
            mDigitsParser.setVerbose(mVerbose);
            LOG(info) << "parse digits";
            digitwordsread = mDigitsParser.Parse(&mCRUPayLoad, linkstart, linkend, currentdetector);
        } else {
            LOG(info) << "link start and end are the same, link appears to be empty";
        }
        sumlinklengths+=mCurrentHalfCRULinkLengths[currentlinkindex];
        sumtrackletwords=trackletwordsread;
        sumdigitwords=digitwordsread;
    } //for loop over link index.
    // we have read in all the digits and tracklets for this event.
    //digits and tracklets are sitting inside the parsing classes.
    //extract the vectors and copy them to tracklets and digits here, building the indexing(triggerrecords)
    //TODO version 2 remove the tracklet and digit class and write directly the binary format.
    mEventTracklets.insert(std::end(mEventTracklets), std::begin(mTrackletsParser.getTracklets()), std::end(mTrackletsParser.getTracklets()));
    mEventDigits.insert(std::end(mEventDigits), std::begin(mDigitsParser.getDigits()), std::end(mDigitsParser.getDigits()));
    if (mVerbose)   LOG(info) << "Event digits after eventi # : " << mEventDigits.size() << " having added : " << mDigitsParser.getDigits().size();
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
    LOG(info) << "And away we go, run method of Translator";
    uint32_t dowhilecount = 0;
    uint64_t totaldataread = 0;
    rewind();
    //std::string filenameroot=std::mkstemp(nullptr);
    //std::ofstream bufferdump("dumpfile");
    //bufferdump.write((char*)mDataBuffer,mDataBufferSize);
    //bufferdump.close();
    uint32_t *bufferptr;
    bufferptr=(uint32_t*)mDataBuffer;
    do {
        //    LOG(info) << "do while loop count " << dowhilecount++;
        //      LOG(info) << " data readin : " << mDataReadIn;
        //      LOG(info) << " mDataBuffer :" << (void*)mDataBuffer << " and offset to start on is :"<< totaldataread;
        //int datareadfromhbf = processHBFs(totaldataread, mVerbose);
        datareadfromhbf=0;
        processHBFs(totaldataread, mVerbose);
        //       LOG(info) << "end with " << datareadfromhbf;
        //      LOG(info) << " about to end do while with " << mDataPointer << " < " << mDataBufferSize;
        //      LOG(info) << " about to end do while having read in " << mDataPointer-bufferptr << " < " << mDataBufferSize;
        //      LOG(info) << " about to end do while with databuffer+databuffersize > datapointer ... " << std::hex << (void*)mDataBuffer+mDataBufferSize << " > " <<std::hex <<  mDataPointer;

    } while (((char*)mDataPointer-mDataBuffer) < mDataBufferSize);

    return false;
};

void checkSummary();

} // namespace trd
} // namespace o2
