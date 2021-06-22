// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   CompressedRawReader.h
/// @brief  TRD raw data translator

#include "TRDReconstruction/CompressedRawReader.h"
#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/CompressedHeader.h"
#include "DataFormatsTRD/CompressedDigit.h"
#include "DataFormatsTRD/Constants.h"
#include "DetectorsRaw/RDHUtils.h"
//#include "Headers/RAWDataHeader.h"
#include "Headers/RDHAny.h"

#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <iostream>
#include <numeric>

namespace o2::trd
{

uint32_t CompressedRawReader::processHBFs()
{

  LOG(debug) << "PROCESS HBF starting at " << (void*)mDataPointer;

  mDataRDH = reinterpret_cast<const o2::header::RDHAny*>(mDataPointer);
  //mEncoderRDH = reinterpret_cast<o2::header::RDHAny*>(mEncoderPointer);
  auto rdh = mDataRDH;
  /* loop until RDH stop header */
  while (!o2::raw::RDHUtils::getStop(rdh)) { // carry on till the end of the event.
    auto headerSize = o2::raw::RDHUtils::getHeaderSize(rdh);
    auto memorySize = o2::raw::RDHUtils::getMemorySize(rdh);
    auto offsetToNext = o2::raw::RDHUtils::getOffsetToNext(rdh);
    auto dataPayload = memorySize - headerSize;
    mFEEID = o2::raw::RDHUtils::getFEEID(rdh);            //TODO change this and just carry around the curreht RDH
    mCRUEndpoint = o2::raw::RDHUtils::getEndPointID(rdh); // the upper or lower half of the currently parsed cru 0-14 or 15-29
    mCRUID = o2::raw::RDHUtils::getCRUID(rdh);
    int packetCount = o2::raw::RDHUtils::getPacketCounter(rdh);
    if (mHeaderVerbose) {
      LOG(info) << "--- RDH open/continue detected";
      LOG(info) << "FEEID : " << mFEEID << " Packet: " << packetCount << " sizes : header" << headerSize << " memorysize:" << memorySize << " offsettonext:" << offsetToNext << " datapayload:" << dataPayload;
      o2::raw::RDHUtils::printRDH(rdh);
    }
    /* copy CRU payload to save buffer */
    //   std::memcpy(mSaveBuffer + mSaveBufferDataSize, (char*)(rdh) + headerSize, drmPayload);
    //   mSaveBufferDataSize += drmPayload;
    //
    //   we will "simply" parse on the fly with a basic state machine.
    mDataPointer = (uint32_t*)((char*)rdh + headerSize);
    mDataEndPointer = (const uint32_t*)((char*)rdh + offsetToNext);
    mIR = o2::raw::RDHUtils::getTriggerIR(rdh);
    while ((void*)mDataPointer < (void*)mDataEndPointer) { // loop to handle the case where a halfcru ends/begins mid rdh data block
      mEventCounter++;
      if (processBlock()) { // at this point the entire payload is in mSaveBuffer, TODO parse this incrementally, less mem foot print.
        LOG(warn) << "processBlock return flase";
        if (mVerbose) {
          LOG(info) << "processBlock return flase";
        }
        break; // end of CRU
      }
      //buildCRUPayLoad(); // the rest of the hbf for the subsequent cruhalfchamber payload.
      // TODO is this even possible if hbfs upto the stop  is for one event and each cru header is for 1 event?
    }
    /* move to next RDH */
    rdh = (o2::header::RDHAny*)((char*)(rdh) + offsetToNext);
    if (mVerbose) {
      LOG(info) << " rdh is now at 0x" << (void*)rdh << " offset to next : " << offsetToNext;
    }
  }

  if (o2::raw::RDHUtils::getStop(rdh)) {
    if (mDataPointer != (const uint32_t*)((char*)rdh + o2::raw::RDHUtils::getOffsetToNext(rdh))) {
      LOG(warn) << " at end of parsing loop and mDataPointer is on next rdh";
    }
    mDataPointer = (const uint32_t*)((char*)rdh + o2::raw::RDHUtils::getOffsetToNext(rdh));
    // make sure mDataPointer is in the correct place.
  } else {
    mDataPointer = (const uint32_t*)((char*)rdh);
  }
  if (mVerbose) {
    LOGF(debug, " at exiting processHBF after advancing to next rdh mDataPointer is %p  ?= %p", (void*)mDataPointer, (void*)mDataEndPointer);
  }
  o2::raw::RDHUtils::printRDH(rdh);

  if (mVerbose) {
    LOG(info) << "--- RDH close detected";
    LOG(info) << "--- END PROCESS HBF";
  }
  /* move to next RDH */
  // mDataPointer = (uint32_t*)((char*)(rdh) + o2::raw::RDHUtils::getOffsetToNext(rdh));

  /* otherwise return */

  return mDataEndPointer - mDataPointer;
}

bool CompressedRawReader::processBlock()
{
  /* process a FeeID/halfcru, 15 links */
  //  LOG(debug) << "--- PROCESS BLOCK FeeID:" << mFEEID;
  mCurrentLinkDataPosition = 0;

  if (mVerbose) {
    LOG(info) << "--- END PROCESS HalfCRU with state: " << mState;
  }
  //this is essentially the inverse function of CruCompressorTask::buildOutput
  //tracklet headers.
  CompressedRawHeader header;
  memcpy((char*)&header, &mDataBuffer, sizeof(CompressedRawHeader));
  mDataPointer += sizeof(CompressedRawHeader); //bytes
  mDataReadIn += sizeof(CompressedRawHeader);
  //tracklets
  int numberoftracklets = header.size;
  o2::InteractionRecord ir(header.bc, header.orbit);
  Tracklet64* trackletptr = (Tracklet64*)mDataPointer; //payload is supposed to be a block of tracklet64 objects.
  std::copy(trackletptr, trackletptr + numberoftracklets, std::back_inserter(mEventTracklets));
  mDataPointer += numberoftracklets * sizeof(Tracklet64); //bytes
  mDataReadIn += numberoftracklets * sizeof(Tracklet64);

  //digits headers.
  CompressedRawTrackletDigitSeperator tracklettrailer;
  memcpy((char*)&tracklettrailer, &mDataBuffer, sizeof(CompressedRawTrackletDigitSeperator));
  mDataBuffer += sizeof(CompressedRawTrackletDigitSeperator);
  mDataReadIn += sizeof(CompressedRawTrackletDigitSeperator);

  //digits
  int numberofdigits = tracklettrailer.digitcount;
  if (mHeaderVerbose || mVerbose) { //sanity check
    bool badheader = false;
    //check compressedrawtrackletdigitseperator for the pre and postfix 0xe's
    if ((tracklettrailer.pad1 & 0xffffff) == 0xeeeeee) {
      badheader = true;
    }
    if ((tracklettrailer.pad2 & 0xffffff) == 0xeeeeee) {
      badheader = true;
    }
    if (badheader) {
      LOG(info) << "Bad compressed raw header seperator digitsize :" << tracklettrailer.digitcount << " padding1 : " << std::hex << tracklettrailer.pad1 << " padding2:" << std::hex << tracklettrailer.pad2;
    }
    if (tracklettrailer.digitcount > 1000) { // TODO probably a better value, but this isfine for now, its for a single half cru count of digits
      LOG(info) << "Digit count seems unusually high : " << tracklettrailer.digitcount;
    }
  }

  CompressedDigit* digitptr = (CompressedDigit*)mDataPointer;
  std::copy(digitptr, digitptr + numberofdigits, std::back_inserter(mCompressedEventDigits));
  mDataPointer += numberofdigits + sizeof(CompressedDigit);
  mDataReadIn += numberofdigits + sizeof(CompressedDigit);
  //convert compresed digits to proper digits TODO put this in a copy operator of Compressed Digit class.
  for (int digitcounter = 0; digitcounter < numberofdigits; ++digitcounter) {
    ArrayADC timebins;
    //TODO This already pre supposes o2::trd::constants::TIMEBINS is 30 from other places, figure something out.
    for (int adc = 0; adc < o2::trd::constants::TIMEBINS; adc++) {
      timebins[adc] = mCompressedEventDigits[digitcounter][adc];
    }
    mEventDigits.emplace_back(mCompressedEventDigits[digitcounter].getDetector(), mCompressedEventDigits[digitcounter].getROB(),
                              mCompressedEventDigits[digitcounter].getMCM(), mCompressedEventDigits[digitcounter].getChannel(), timebins);
  }
  // now we *should* have a CompressedRawDigitEndMarker or more commonly known as a lot of 0xe
  if ((uint32_t)*mDataPointer != 0xeeeeeeee) {
    LOG(warn) << std::hex << *mDataPointer << "  We should be seeing CompressedRawDigitEndMarker which is the same as a o2::trd::constants::CRUPADDING32";
  }
  mDataPointer += 4;
  mDataReadIn += 4;
  if (mDataReadIn % 8 != 0) {
    // we have an extra padding word to make it to a full 64bit data buffer.
    if ((uint32_t)*mDataPointer != 0xeeeeeeee) {
      LOG(warn) << std::hex << *mDataPointer << "  We should be seeing CompressedRawDigitEndMarker which is the same as a o2::trd::constants::CRUPADDING32";
    }
    mDataPointer += 4;
    mDataReadIn += 4;
  }
  auto lasttrigger = mEventTriggers.size() - 1;
  int lastdigit = mEventTriggers[lasttrigger].getFirstDigit() + mEventTriggers[lasttrigger].getNumberOfDigits();
  int lasttracklet = mEventTriggers[lasttrigger].getFirstTracklet() + mEventTriggers[lasttrigger].getNumberOfTracklets();
  mEventTriggers.emplace_back(mIR, lastdigit, numberofdigits, lasttracklet, numberoftracklets);

  // either 1 or more depending on padding requirements.
  if (mVerbose) {
    LOG(info) << "Tracklets in block : " << numberoftracklets << " vector has size:" << mEventTracklets.size();
    LOG(info) << "Digits in block : " << numberofdigits << " vector has size:" << mEventDigits.size();
  }
  return true;
}

void CompressedRawReader::resetCounters()
{
  mEventCounter = 0;
  mFatalCounter = 0;
  mErrorCounter = 0;
}

void CompressedRawReader::checkSummary()
{
  char chname[2] = {'a', 'b'};

  LOG(info) << "--- SUMMARY COUNTERS: " << mEventCounter << " events "
            << " | " << mFatalCounter << " decode fatals "
            << " | " << mErrorCounter << " decode errors ";
}

} // namespace o2::trd
