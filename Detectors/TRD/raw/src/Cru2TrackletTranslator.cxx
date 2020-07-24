// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   Cru2TrackletTranslator.h
/// @brief  TRD raw data translator

#include "TRDRaw/Cru2TrackletTranslator.h"
#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DetectorsRaw/RDHUtils.h"
//#include "Headers/RAWDataHeader.h"
#include "Headers/RDHAny.h"

#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <iostream>
#include <numeric>

namespace o2
{
namespace trd
{

uint32_t Cru2TrackletTranslator::processHBFs()
{

  LOG(debug) << "PROCESS HBF starting at " << (void*)mDataPointer;

  mDataRDH = reinterpret_cast<const o2::header::RDHAny*>(mDataPointer);
  //mEncoderRDH = reinterpret_cast<o2::header::RDHAny*>(mEncoderPointer);
  auto rdh = mDataRDH;
  /* loop until RDH stop header */
  while (!o2::raw::RDHUtils::getStop(rdh)) { // carry on till the end of the event.
    LOG(debug) << "--- RDH open/continue detected";
    o2::raw::RDHUtils::printRDH(rdh);

    auto headerSize = o2::raw::RDHUtils::getHeaderSize(rdh);
    auto memorySize = o2::raw::RDHUtils::getMemorySize(rdh);
    auto offsetToNext = o2::raw::RDHUtils::getOffsetToNext(rdh);
    auto dataPayload = memorySize - headerSize;
    mFEEID = o2::raw::RDHUtils::getFEEID(rdh);            //TODO change this and just carry around the curreht RDH
    mCRUEndpoint = o2::raw::RDHUtils::getEndPointID(rdh); // the upper or lower half of the currently parsed cru 0-14 or 15-29
    mCRUID = o2::raw::RDHUtils::getCRUID(rdh);
    int packetCount = o2::raw::RDHUtils::getPacketCounter(rdh);
    LOG(debug) << "FEEID : " << mFEEID << " Packet: " << packetCount << " sizes : header" << headerSize << " memorysize:" << memorySize << " offsettonext:" << offsetToNext << " datapayload:" << dataPayload;
    /* copy CRU payload to save buffer TODO again why the copy ??*/
    //   std::memcpy(mSaveBuffer + mSaveBufferDataSize, (char*)(rdh) + headerSize, drmPayload);
    //   mSaveBufferDataSize += drmPayload;
    //
    //   we will "simply" parse on the fly with a basic state machine.
    LOGF(debug, "rdh ptr is %p\n", (void*)rdh);
    mDataPointer = (uint32_t*)((char*)rdh + headerSize);
    LOGF(debug, "mDataPointer is %p  ?= %p", (void*)mDataPointer, (void*)rdh);
    mDataEndPointer = (const uint32_t*)((char*)rdh + offsetToNext);
    LOGF(debug, "mDataEndPointer is %p\n ", (void*)mDataEndPointer);
    while ((void*)mDataPointer < (void*)mDataEndPointer) { // loop to handle the case where a halfcru ends/begins mid rdh data block
      mEventCounter++;
      if (processHalfCRU()) { // at this point the entire payload is in mSaveBuffer, TODO parse this incrementally, less mem foot print.
        LOG(warn) << "processHalfCRU return flase";
        break; // end of CRU
      }
      mState = CRUStateHalfChamber;
      //buildCRUPayLoad(); // the rest of the hbf for the subsequent cruhalfchamber payload.
      // TODO is this even possible if hbfs upto the stop  is for one event and each cru header is for 1 event?
    }
    /* move to next RDH */
    rdh = (o2::header::RDHAny*)((char*)(rdh) + offsetToNext);
    LOG(debug) << " rdh is now at 0x" << (void*)rdh << " offset to next : " << offsetToNext;
  }

  if (o2::raw::RDHUtils::getStop(rdh)) {
    if (mDataPointer != (const uint32_t*)((char*)rdh + o2::raw::RDHUtils::getOffsetToNext(rdh))) {
      LOG(warn) << " at end of parsing loop and mDataPointer is on next rdh";
    }
    mDataPointer = (const uint32_t*)((char*)rdh + o2::raw::RDHUtils::getOffsetToNext(rdh));
    // make sure mDataPointer is in the correct place.
  } else
    mDataPointer = (const uint32_t*)((char*)rdh);
  LOGF(debug, " at exiting processHBF after advancing to next rdh mDataPointer is %p  ?= %p", (void*)mDataPointer, (void*)mDataPointer);
  o2::raw::RDHUtils::printRDH(rdh);

  LOG(debug) << "--- RDH close detected";

  LOG(debug) << "--- END PROCESS HBF";

  /* move to next RDH */
  // mDataPointer = (uint32_t*)((char*)(rdh) + o2::raw::RDHUtils::getOffsetToNext(rdh));

  /* otherwise return */

  return mDataEndPointer - mDataPointer;
}

bool Cru2TrackletTranslator::buildCRUPayLoad()
{
  // copy data for the current half cru, and when we eventually get to the end of the payload return 1
  // to say we are done.
  int cruid = 0;
  int additionalBytes = -1;
  int crudatasize = -1;
  LOG(debug) << "--- Build CRU Payload, added " << additionalBytes << " bytes to CRU "
             << cruid << " with new size " << crudatasize;
  return true;
}

bool Cru2TrackletTranslator::processHalfCRU()
{
  /* process a FeeID/halfcru, 15 links */
  LOG(debug) << "--- PROCESS HalfCRU FeeID:" << mFEEID;
  mCurrentLinkDataPosition = 0;
  if (mState == CRUStateHalfCRUHeader) {
    // well then read the halfcruheader.
    memcpy(&mCurrentHalfCRUHeader, (void*)(mDataPointer), sizeof(mCurrentHalfCRUHeader));
    //mEncoderRDH = reinterpret_cast<o2::header::RDHAny*>(mEncoderPointer);)
    mCurrentLink = 0;
    o2::trd::getlinkdatasizes(mCurrentHalfCRUHeader, mCurrentHalfCRULinkLengths);
    o2::trd::getlinkerrorflags(mCurrentHalfCRUHeader, mCurrentHalfCRULinkErrorFlags);
    mTotalHalfCRUDataLength = std::accumulate(mCurrentHalfCRULinkLengths.begin(),
                                              mCurrentHalfCRULinkLengths.end(),
                                              decltype(mCurrentHalfCRULinkLengths)::value_type(0));
    // we will always have at least a length of 1 fully padded for each link.
    LOG(debug) << "Found  a HalfCRUHeader : ";
    LOG(debug) << mCurrentHalfCRUHeader;
    mState = CRUStateHalfChamber; // we expect a halfchamber header now
                                  //TODO maybe change name to something more generic, this is will have to change to handle other data types config/adcdata.
  }
  while (mDataPointer != mDataEndPointer && mCurrentLinkDataPosition != mTotalHalfCRUDataLength * 16) { // while we are stil in the rdh block and with in the current link
    LOG(debug) << "in while loop with state of :" << mState;
    if (mState == CRUStateHalfChamber) {
      // read in the halfchamber header.
      LOGF(debug, "mTrackletHCHeader is at %p had value 0x%08x", (void*)mDataPointer, mDataPointer[0]);
      mTrackletHCHeader = (TrackletHCHeader*)mDataPointer;
      mDataPointer += 16; //sizeof(mTrackletHCHeader)/4;
      mHCID = mTrackletHCHeader->HCID;
      //     LOGF(info,"mDataPointer after advancing past TrackletHCHeader is at %p has value 0x%08x",(void*)mDataPointer,mDataPointer[0]);
      //if(debugparsing){
      //     printHalfChamber(*mTrackletHCHeader);
      // }
      mState = CRUStateTrackletMCMHeader; //now we expect a TrackletMCMHeader or some padding.
    }
    if (mState == CRUStateTrackletMCMHeader) {
      LOGF(debug, "mTrackletMCMHeader is at %p had value 0x%08x", (void*)mDataPointer, mDataPointer[0]);
      if (debugparsing) {
        //           LOG(debug) << " state is : " << mState << " about to read TrackletMCMHeader";
      }
      //read the header OR padding of 0xeeee;
      if (mDataPointer[0] != 0xeeeeeeee) {
        //we actually have an header word.
        mTrackletHCHeader = (TrackletHCHeader*)mDataPointer;
        LOG(debug) << "state mcmheader and word : 0x" << std::hex << mDataPointer[0];
        mDataPointer++;
        mCurrentLinkDataPosition++;
        if (debugparsing) {
          //       printTrackletMCMHeader(*mTrackletHCHeader);
        }
        mState = CRUStateTrackletMCMData;
      } else { // this is the case of a first padding word for a "noncomplete" tracklet i.e. not all 3 tracklets.
               //        LOG(debug) << "C";
        mState = CRUStatePadding;
        mDataPointer++;
        mCurrentLinkDataPosition++;
        TRDStatCounters.LinkPadWordCounts[mHCID]++; // keep track off all the padding words.
        if (debugparsing) {
          //       printTrackletMCMHeader(*mTrackletHCHeader);
        }
      }
    }
    if (mState == CRUStatePadding) {
      LOGF(debug, "Padding is at %p had value 0x%08x", (void*)mDataPointer, mDataPointer[0]);
      LOG(debug) << "state padding and word : 0x" << std::hex << mDataPointer[0];
      if (mDataPointer[0] == 0xeeeeeeee) {
        //another pointer with padding.
        mDataPointer++;
        mCurrentLinkDataPosition++;
        TRDStatCounters.LinkPadWordCounts[mHCID]++; // keep track off all the padding words.
        if (mDataPointer[0] & 0x1) {
          //mcmheader
          //        LOG(debug) << "changing state from padding to mcmheader as next datais 0x" << std::hex << mDataPointer[0];
          mState = CRUStateTrackletMCMHeader;
        } else if (mDataPointer[0] != 0xeeeeeeee) {
          //        LOG(debug) << "changing statefrom padding to mcmdata as next datais 0x" << std::hex << mDataPointer[0];
          mState = CRUStateTrackletMCMData;
        }
      } else {
        LOG(debug) << "some went wrong we are in state padding, but not a pad word. 0x" << (void*)mDataPointer;
      }
    }
    if (mState == CRUStateTrackletMCMData) {
      LOGF(debug, "mTrackletMCMData is at %p had value 0x%08x", (void*)mDataPointer, mDataPointer[0]);
      //tracklet data;
      // build tracklet.
      //for the case of on flp build a vector of tracklets, then pack them into a data stream with a header.
      //for dpl build a vector and connect it with a triggerrecord.
      mTrackletMCMData = (TrackletMCMData*)mDataPointer;
      mDataPointer++;
      mCurrentLinkDataPosition++;
      if (mDataPointer[0] == 0xeeeeeeee) {
        mState = CRUStatePadding;
        //  LOG(debug) <<"changing to padding from mcmdata" ;
      } else {
        if (mDataPointer[0] & 0x1) {
          mState = CRUStateTrackletMCMHeader; // we have more tracklet data;
          LOG(debug) << "changing from MCMData to MCMHeader";
        } else {
          mState = CRUStateTrackletMCMData;
          LOG(debug) << "continuing with mcmdata";
        }
      }
      // Tracklet64 trackletsetQ0(o2::trd::getTrackletQ0());
    }
    //accounting ....
    // mCurrentLinkDataPosition256++;
    // mCurrentHalfCRUDataPosition256++;
    // mTotalHalfCRUDataLength++;
    LOG(debug) << mDataPointer << ":" << mDataEndPointer << " &&  " << mCurrentLinkDataPosition << " != " << mTotalHalfCRUDataLength * 16;
  }
  //end of data so
  /* init decoder */
  mDataNextWord = 1;
  mError = false;
  mFatal = false;

  /* check TRD Data Header */

  LOG(debug) << "--- END PROCESS HalfCRU with state: " << mState;

  return true;
}

bool Cru2TrackletTranslator::processCRULink()
{
  /* process a CRU Link 15 per half cru */
  //  checkFeeID(); // check the link we are working with corresponds with the FeeID we have in the current rdh.
  //  uint32_t slotId = GET_TRMDATAHEADER_SLOTID(*mDataPointer);
  return false;
}

bool Cru2TrackletTranslator::checkerCheck()
{
  /* checker check */

  LOG(debug) << "--- CHECK EVENT";

  return false;
}

void Cru2TrackletTranslator::checkerCheckRDH()
{
  /* check orbit */
  //   LOGF(info," --- Checking HalfCRU/RDH orbit: %08x/%08x \n", orbit, getOrbit(mDatardh));
  //  if (orbit != mDatardh->orbit) {
  //      LOGF(info," HalfCRU/RDH orbit mismatch: %08x/%08x \n", orbit, getOrbit(mDatardh));
  //  }

  /* check FEE id */
  //   LOGF(info, " --- Checking CRU/RDH FEE id: %d/%d \n", mcruFeeID, getFEEID(mDatardh) & 0xffff);
  //  if (mcruFeeID != (mDatardh->feeId & 0xFF)) {
  //      LOGF(info, " HalfCRU/RDH FEE id mismatch: %d/%d \n", mcruFeeID, getFEEID(mDatardh) & 0xffff);
  //  }
}

void Cru2TrackletTranslator::resetCounters()
{
  mEventCounter = 0;
  mFatalCounter = 0;
  mErrorCounter = 0;
}

void Cru2TrackletTranslator::checkSummary()
{
  char chname[2] = {'a', 'b'};

  LOG(info) << "--- SUMMARY COUNTERS: " << mEventCounter << " events "
            << " | " << mFatalCounter << " decode fatals "
            << " | " << mErrorCounter << " decode errors ";
}

} // namespace trd
} // namespace o2
