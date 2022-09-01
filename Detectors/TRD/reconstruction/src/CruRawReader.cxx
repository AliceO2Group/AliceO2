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

/// @file   CruRawReader.h
/// @brief  TRD raw data translator

#include "DetectorsRaw/RDHUtils.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "Headers/RDHAny.h"
#include "TRDReconstruction/CruRawReader.h"
#include "TRDBase/FeeParam.h"
#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/HelperMethods.h"

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/RawDeviceService.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/Output.h"
#include "Framework/InputRecordWalker.h"
#include "DataFormatsCTP/TriggerOffsetsParam.h"

#include <string>
#include <numeric>
#include <iomanip>

using namespace o2::trd::constants;

namespace o2::trd
{

void CruRawReader::configure(int tracklethcheader, int halfchamberwords, int halfchambermajor, std::bitset<16> options)
{
  mTrackletHCHeaderState = tracklethcheader;
  mHalfChamberWords = halfchamberwords;
  mHalfChamberMajor = halfchambermajor;
  mOptions = options;
}

void CruRawReader::incrementErrors(int error, int hcid, std::string message)
{
  mEventRecords.incParsingError(error, hcid);
  if (mOptions[TRDVerboseErrorsBit]) {
    std::string logMessage = "Detected PE " + ParsingErrorsString.at(error);
    if (hcid >= 0) {
      logMessage += " HCID " + std::to_string(hcid);
    }
    if (!message.empty()) {
      logMessage += " Message: " + message;
    }
    LOG(info) << logMessage;
  }
}

bool CruRawReader::checkRDH(const o2::header::RDHAny* rdh)
{
  // first check for FEEID from unconfigured CRU
  TRDFeeID feeid;
  feeid.word = o2::raw::RDHUtils::getFEEID(rdh);
  if (((feeid.word) >> 4) == 0xfff) { // error condition is 0xfff? as the end point is known to the cru, but the rest is configured.
    if (mMaxErrsPrinted > 0) {
      LOG(error) << "RDH check failed due to 0xfff. FLP not configured, call TRD on call. Whole feeid = " << std::hex << (unsigned int)feeid.word;
      checkNoErr();
    }
    incrementErrors(FEEIDIsFFFF, -1, fmt::format("failed due to 0xfff? : {} whole feeid: {} ", feeid.word, (unsigned int)feeid.word));
    return false;
  }
  if (feeid.supermodule > 17) {
    if (mMaxWarnPrinted > 0) {
      LOG(warn) << "Wrong supermodule number " << std::dec << (int)feeid.supermodule << " detected in RDH. Whole feeid : " << std::hex << (unsigned int)feeid.word;
      checkNoWarn();
    }
    incrementErrors(FEEIDBadSector, -1, fmt::format("Wrong supermodule number {} detected in RDH whole feeid: {}", (int)feeid.supermodule, (unsigned int)feeid.word));
    return false;
  }
  if (o2::raw::RDHUtils::getMemorySize(rdh) <= 0) {
    if (mMaxWarnPrinted > 0) {
      LOG(warn) << "Received RDH header with invalid memory size (<= 0) ";
      checkNoWarn();
    }
    incrementErrors(BadRDHMemSize, -1, fmt::format("Received RDH header with invalid memory size (<= 0) "));
    return false;
  }
  return true;
}

bool CruRawReader::compareRDH(const o2::header::RDHAny* rdhPrev, const o2::header::RDHAny* rdhCurr)
{
  if (o2::raw::RDHUtils::getFEEID(rdhPrev) != o2::raw::RDHUtils::getFEEID(rdhCurr)) {
    if (mMaxWarnPrinted > 0) {
      LOG(warn) << "ERDH FEEID are not identical in rdh.";
      checkNoWarn();
    }
    incrementErrors(BadRDHFEEID);
    return false;
  }
  if (o2::raw::RDHUtils::getEndPointID(rdhPrev) != o2::raw::RDHUtils::getEndPointID(rdhCurr)) {
    if (mMaxWarnPrinted > 0) {
      LOG(warn) << "ERDH  EndPointID are not identical in rdh.";
      checkNoWarn();
    }
    if (mOptions[TRDVerboseErrorsBit]) {
      LOG(error) << "ERDH  EndPointID are not identical in rdh.";
    }
    incrementErrors(BadRDHEndPoint);
    return false;
  }
  if (o2::raw::RDHUtils::getTriggerOrbit(rdhPrev) != o2::raw::RDHUtils::getTriggerOrbit(rdhCurr)) {
    if (mMaxWarnPrinted > 0) {
      LOG(warn) << "ERDH  Orbit are not identical in rdh.";
      checkNoWarn();
    }
    incrementErrors(BadRDHOrbit);
    return false;
  }
  if (o2::raw::RDHUtils::getCRUID(rdhPrev) != o2::raw::RDHUtils::getCRUID(rdhCurr)) {
    if (mMaxWarnPrinted > 0) {
      LOG(warn) << "ERDH  CRUID are not identical in rdh.";
      checkNoWarn();
    }
    incrementErrors(BadRDHCRUID);
    return false;
  }
  if (o2::raw::RDHUtils::getPacketCounter(rdhPrev) == o2::raw::RDHUtils::getPacketCounter(rdhCurr)) {
    if (mMaxWarnPrinted > 0) {
      LOG(warn) << "ERDH  PacketCounters are not sequential in rdh.";
      checkNoWarn();
    }
    incrementErrors(BadRDHPacketCounter);
    return false;
  }
  return true;
}

int CruRawReader::processHBFs()
{
  const o2::header::RDHAny* rdh = reinterpret_cast<const o2::header::RDHAny*>(mCurrRdhPtr);
  auto rdhPrevious = rdh;
  bool firstRdh = true;
  uint32_t totalDataInputSize = 0;
  mTotalHBFPayLoad = 0;

  // loop until RDH stop header
  while (!o2::raw::RDHUtils::getStop(rdh)) { // carry on till the end of the event.
    if (mOptions[TRDVerboseBit]) {
      LOG(info) << "Current RDH is as follows:";
      o2::raw::RDHUtils::printRDH(rdh);
    }
    if (!checkRDH(rdh)) {
      return -1;
    }
    if (!firstRdh && !compareRDH(rdhPrevious, rdh)) {
      // previous and current RDHs are inconsistent, this should not happen
      return -1;
    }
    rdhPrevious = rdh;
    firstRdh = false;
    auto headerSize = o2::raw::RDHUtils::getHeaderSize(rdh);
    auto memorySize = o2::raw::RDHUtils::getMemorySize(rdh);
    auto offsetToNext = o2::raw::RDHUtils::getOffsetToNext(rdh);
    auto rdhpayload = memorySize - headerSize;
    mFEEID.word = o2::raw::RDHUtils::getFEEID(rdh);
    mCRUEndpoint = o2::raw::RDHUtils::getEndPointID(rdh); // the upper or lower half of the currently parsed cru 0-14 or 15-29
    mCRUID = o2::raw::RDHUtils::getCRUID(rdh);
    mIR = o2::raw::RDHUtils::getTriggerIR(rdh); // the orbit counter is taken from the RDH here, the bc is overwritten later from the HalfCRUHeader

    // copy the contents of the current RDH into the buffer to be parsed, RDH payload is memory size minus header size
    std::memcpy((char*)&mHBFPayload[0] + mTotalHBFPayLoad, ((char*)rdh) + headerSize, rdhpayload);
    // copy the contents of the current rdh into the buffer to be parsed
    mTotalHBFPayLoad += rdhpayload;
    totalDataInputSize += offsetToNext;
    // move to next rdh
    rdh = reinterpret_cast<const o2::header::RDHAny*>(reinterpret_cast<const char*>(rdh) + offsetToNext);
    // increment the data pointer by the size of the next RDH.
    mCurrRdhPtr = reinterpret_cast<const char*>(rdh) + offsetToNext;

    if (mOptions[TRDVerboseBit]) {
      LOG(info) << "Next RDH is as follows:";
      o2::raw::RDHUtils::printRDH(rdh);
    }

    if (!o2::raw::RDHUtils::getStop(rdh) && offsetToNext >= mCurrRdhPtr - mDataBufferPtr) {
      // we can still copy into this buffer.
      if (mMaxWarnPrinted > 0) {
        LOGP(warn, "RDH offsetToNext = {} is larger than it can possibly be. Remaining data in the buffer = {}", offsetToNext, mCurrRdhPtr - mDataBufferPtr);
        checkNoWarn();
      }
      return -1;
    }
  }

  // at this point the entire HBF data payload is sitting in mHBFPayload and the total data count is mTotalHBFPayLoad
  int iteration = 0;
  mHBFoffset32 = 0;
  while (mHBFoffset32 < (mTotalHBFPayLoad / 4)) {
    if (mOptions[TRDVerboseBit]) {
      LOGP(info, "Current half-CRU iteration {}, current offset in the HBF payload {}, total payload in number of 32-bit words {}", iteration, mHBFoffset32, mTotalHBFPayLoad / 4);
    }
    if (!processHalfCRU(iteration)) {
      //dump rest of this rdh payload, something screwed up.
      break;
    }
    iteration++;
  } // loop of halfcru's while there is still data in the heart beat frame.

  return totalDataInputSize;
}

void CruRawReader::checkDigitHCHeader(int halfChamberIdRef)
{
  // compare the half chamber ID from the digit HC header with the reference one obtained from the link ID
  int halfChamberIdHeader = mDigitHCHeader.supermodule * NHCPERSEC + mDigitHCHeader.stack * NLAYER * 2 + mDigitHCHeader.layer * 2 + mDigitHCHeader.side;
  if (halfChamberIdRef != halfChamberIdHeader) {
    incrementErrors(DigitHCHeaderMismatch, halfChamberIdRef, fmt::format("HCID mismatch detected. HCID from DigitHCHeader: {}, HCID from RDH: {}", halfChamberIdHeader, halfChamberIdRef));
    if (mMaxWarnPrinted > 0) {
      LOGF(alarm, "HCID mismatch in DigitHCHeader detected for ref HCID %i. DigitHCHeader says HCID is %i", halfChamberIdRef, halfChamberIdHeader);
      checkNoWarn();
    }
  }
}

bool CruRawReader::parseDigitHCHeaders(int hcid)
{
  // mHBFoffset32 is the current offset into the current buffer,
  //
  mDigitHCHeader.word = mHBFPayload[mHBFoffset32++];
  if (mOptions[TRDByteSwapBit]) {
    // byte swap if needed.
    o2::trd::HelperMethods::swapByteOrder(mDigitHCHeader.word);
  }

  // a hack used to make old data readable (e.g. Kr from 2021)
  if (mDigitHCHeader.major == 0 && mDigitHCHeader.minor == 0 && mDigitHCHeader.numberHCW == 0) {
    mDigitHCHeader.major = mHalfChamberMajor;
    mDigitHCHeader.minor = 42;
    mDigitHCHeader.numberHCW = mHalfChamberWords;
    if (mHalfChamberWords == 0 || mHalfChamberMajor == 0) {
      if (mMaxWarnPrinted > 0) {
        LOG(alarm) << "DigitHCHeader is corrupted and using a hack as workaround is not configured";
        checkNoWarn();
      }
      return false;
    }
  }

  int additionalHeaderWords = mDigitHCHeader.numberHCW;
  if (additionalHeaderWords >= 3) {
    incrementErrors(DigitHeaderCountGT3, hcid);
    if (mMaxWarnPrinted > 0) {
      LOGF(warn, "Found too many additional words (%i) in DigitHCHeader 0x%08x", additionalHeaderWords, mDigitHCHeader.word);
      checkNoWarn();
    }
    return false;
  }
  std::bitset<3> headersfound;
  std::array<uint32_t, 3> headers{0};

  for (int headerwordcount = 0; headerwordcount < additionalHeaderWords; ++headerwordcount) {
    headers[headerwordcount] = mHBFPayload[mHBFoffset32++];
    if (mOptions[TRDByteSwapBit]) {
      // byte swap if needed.
      o2::trd::HelperMethods::swapByteOrder(headers[headerwordcount]);
    }
    switch (getDigitHCHeaderWordType(headers[headerwordcount])) {

      case 1: // header header1;
        if (headersfound.test(0)) {
          // we have a problem, we already have a Digit HC Header1, we are lost.
          if (mOptions[TRDVerboseErrorsBit]) {
            LOGF(warn, "We have more than one DigitHCHeader of type 1. Current word in hex %08x", headers[headerwordcount]);
            printDigitHCHeader(mDigitHCHeader, headers.data());
          }
          incrementErrors(DigitHCHeader1Problem, hcid);
        }
        DigitHCHeader1 header1;
        header1.word = headers[headerwordcount];
        headersfound.set(0);
        if ((header1.numtimebins > TIMEBINS) || (header1.numtimebins < 3)) {
          if (mMaxWarnPrinted > 0) {
            LOGF(warn, "According to Digit HC Header 1 there are %i time bins configured", (int)header1.numtimebins);
            checkNoWarn();
          }
          return false;
        }
        mTimeBins = header1.numtimebins;
        break;

      case 2: // header header2;
        if (headersfound.test(1)) {
          // we have a problem, we already have a Digit HC Header2, we are hereby lost.
          if (mOptions[TRDVerboseErrorsBit]) {
            LOGF(warn, "We have more than one DigitHCHeader of type 2. Current word in hex %08x", headers[headerwordcount]);
            printDigitHCHeader(mDigitHCHeader, headers.data());
          }
          incrementErrors(DigitHCHeader2Problem, hcid);
        }
        DigitHCHeader2 header2;
        header2.word = headers[headerwordcount];
        headersfound.set(1);
        break;

      case 3: // header header3;
        if (headersfound.test(2)) {
          // we have a problem, we already have a Digit HC Header3, we are hereby lost.
          if (mOptions[TRDVerboseErrorsBit]) {
            LOG(info) << "We have a >1 Digit HC Header 2  : " << std::hex << " raw: 0x" << headers[headerwordcount];
            printDigitHCHeader(mDigitHCHeader, headers.data());
          }
          incrementErrors(DigitHCHeader3Problem, hcid);
        }
        DigitHCHeader3 header3;
        header3.word = headers[headerwordcount];
        headersfound.set(2);
        if (mHaveSeenDigitHCHeader3) {
          if (header3.svnver != mPreviousDigitHCHeadersvnver || header3.svnrver != mPreviousDigitHCHeadersvnrver) {
            if (mOptions[TRDVerboseErrorsBit]) {
              LOG(warning) << "Conflicting SVN in DigitHCHeader3";
              printDigitHCHeader(mDigitHCHeader, headers.data());
            }
            incrementErrors(DigitHCHeaderSVNMismatch, hcid);
            return false;
          }
        } else {
          mPreviousDigitHCHeadersvnver = header3.svnver;
          mPreviousDigitHCHeadersvnrver = header3.svnrver;
          mHaveSeenDigitHCHeader3 = true;
        }
        break;

      default:
        incrementErrors(DigitHeaderWrongType, hcid, fmt::format("Failed to determine DigitHCHeader type for {:#010x}", headers[headerwordcount]));
        return false;
    }
  }
  if (mOptions[TRDVerboseBit]) {
    printDigitHCHeader(mDigitHCHeader, &headers[0]);
  }

  return true;
}

bool CruRawReader::processHalfCRU(int iteration)
{
  // process data from one half-CRU
  // iteration corresponds to the trigger number within the HBF

  // this should only hit that instance where the cru payload is a "blank event" of CRUPADDING32
  if (mHBFPayload[mHBFoffset32] == CRUPADDING32) {
    if (mOptions[TRDVerboseBit]) {
      LOG(info) << "blank rdh payload data at " << mHBFoffset32 << ": 0x " << std::hex << mHBFPayload[mHBFoffset32] << " and 0x" << mHBFPayload[mHBFoffset32 + 1];
    }
    int loopcount = 0;
    while (mHBFPayload[mHBFoffset32] == CRUPADDING32 && loopcount < 8) { // can only ever be an entire 256 bit word hence a limit of 8 here.
      // TODO: check with Guido if it could not actually be more padding words
      mHBFoffset32++;
      loopcount++;
    }
    return true;
  }

  auto crustart = std::chrono::high_resolution_clock::now();

  memcpy(&mCurrentHalfCRUHeader, &(mHBFPayload[mHBFoffset32]), sizeof(HalfCRUHeader));
  mHBFoffset32 += sizeof(HalfCRUHeader) / 4; // advance past the header.
  if (mOptions[TRDVerboseBit]) {
    //output the cru half chamber header : raw/parsed
    printHalfCRUHeader(mCurrentHalfCRUHeader);
  }
  if (!halfCRUHeaderSanityCheck(mCurrentHalfCRUHeader)) {
    incrementErrors(HalfCRUCorrupt, -1, fmt::format("HalfCRU header failed sanity check for FEEID with {:#x} ", (unsigned int)mFEEID.word));
    mWordsRejected += (mTotalHBFPayLoad / 4) - mHBFoffset32 + sizeof(HalfCRUHeader) / 4;
    return false; // not recoverable, since we don't know when the next half-cru header would start
  }

  o2::trd::getHalfCRULinkDataSizes(mCurrentHalfCRUHeader, mCurrentHalfCRULinkLengths);
  o2::trd::getHalfCRULinkErrorFlags(mCurrentHalfCRUHeader, mCurrentHalfCRULinkErrorFlags);
  uint32_t totalHalfCRUDataLength256 = std::accumulate(mCurrentHalfCRULinkLengths.begin(),
                                                       mCurrentHalfCRULinkLengths.end(),
                                                       0U);
  uint32_t totalHalfCRUDataLength32 = totalHalfCRUDataLength256 * 8; // convert to 32-bit words

  if (mOptions[TRDOnlyCalibrationTriggerBit] && mCurrentHalfCRUHeader.EventType == o2::trd::constants::ETYPEPHYSICSTRIGGER) {
    // skip triggers without digits
    mWordsRejected += totalHalfCRUDataLength32;
    mHBFoffset32 += totalHalfCRUDataLength32;
    return true;
  }
  if (mCRUEndpoint != mCurrentHalfCRUHeader.EndPoint) {
    if (mMaxWarnPrinted > 0) {
      LOGF(warn, "End point mismatch detected. HalfCRUHeader says %i, RDH says %i", mCurrentHalfCRUHeader.EndPoint, mCRUEndpoint);
      checkNoWarn();
    }
    // try next trigger (HalfCRUHeader)
    mHBFoffset32 += totalHalfCRUDataLength32;
    mWordsRejected += totalHalfCRUDataLength32;
    return true;
  }

  if (iteration > 0) {
    // for the second trigger (and thus second HalfCRUHeader we see) we can do some more sanity checks
    // in case one check fails, we try to go to the next HalfCRUHeader
    if (mCurrentHalfCRUHeader.EndPoint != mPreviousHalfCRUHeader.EndPoint) {
      if (mMaxWarnPrinted > 0) {
        LOGF(warn, "For current half-CRU index %i we have end point %i, while the previous end point was %i", iteration, mCurrentHalfCRUHeader.EndPoint, mPreviousHalfCRUHeader.EndPoint);
        checkNoWarn();
      }
      incrementErrors(HalfCRUCorrupt);
      mWordsRejected += totalHalfCRUDataLength32;
      mHBFoffset32 += totalHalfCRUDataLength32;
      return true;
    }
    if (mCurrentHalfCRUHeader.StopBit != mPreviousHalfCRUHeader.StopBit) {
      if (mMaxWarnPrinted > 0) {
        LOGF(warn, "For current half-CRU index %i we have stop bit %i, while the previous stop bit was %i", iteration, mCurrentHalfCRUHeader.StopBit, mPreviousHalfCRUHeader.StopBit);
        checkNoWarn();
      }
      incrementErrors(HalfCRUCorrupt);
      mWordsRejected += totalHalfCRUDataLength32;
      mHBFoffset32 += totalHalfCRUDataLength32;
      return true;
    }
  }
  mPreviousHalfCRUHeader = mCurrentHalfCRUHeader;

  //can this half cru length fit into the available space of the rdh accumulated payload
  if (totalHalfCRUDataLength32 > (mTotalHBFPayLoad / 4) - mHBFoffset32) {
    if (mMaxWarnPrinted > 0) {
      LOGP(warn, "HalfCRU header says it contains more data ({} 32-bit words) than is remaining in the payload ({} 32-bit words)", totalHalfCRUDataLength32, ((mTotalHBFPayLoad / 4) - mHBFoffset32));
      checkNoWarn();
    }
    incrementErrors(HalfCRUSumLength);
    mWordsRejected += (mTotalHBFPayLoad / 4) - mHBFoffset32;
    return false; // not recoverable, since we don't know when the next half-cru header would start
  }

  mIR.bc = mCurrentHalfCRUHeader.BunchCrossing; // TRD BC is obtained from the HalfCRUHeader
  if (o2::ctp::TriggerOffsetsParam::Instance().LM_L0 > (int)mIR.bc) {
    // applying the configured BC shift would lead to negative BC, hence we reject this trigger
    // dump to the end of this cruhalfchamberheader
    // data to dump is totalHalfCRUDataLength32
    mHBFoffset32 += totalHalfCRUDataLength32;   // go to the end of this halfcruheader and payload.
    mWordsRejected += totalHalfCRUDataLength32; // add the rejected data to the accounting;
    incrementErrors(HalfCRUBadBC, -1, fmt::format("Trigger rejected, since BC shift would move the BC to a negative value. LM_L0: {} bc: {}", o2::ctp::TriggerOffsetsParam::Instance().LM_L0, (int)mIR.bc));
    return true; // nothing particularly wrong with the data, we just dont want it, as a trigger problem
  } else {
    // apply CTP offset shift
    mIR.bc -= o2::ctp::TriggerOffsetsParam::Instance().LM_L0;
  }
  mEventRecords.setCurrentEventRecord(mIR);

  //loop over links
  uint32_t linksizeAccum32 = 0;     // accumulated size of all links in 32-bit words
  auto hbfOffsetTmp = mHBFoffset32; // store current position at the beginning of the half-CRU payload data
  for (int currentlinkindex = 0; currentlinkindex < NLINKSPERHALFCRU; currentlinkindex++) {
    int cruIdx = mFEEID.supermodule * 2 + mFEEID.side;                    // 2 CRUs per SM, side defining A/C-side CRU
    int halfCruIdx = cruIdx * 2 + mFEEID.endpoint;                        // endpoint (0 or 1) defines half-CRU
    int linkIdxGlobal = halfCruIdx * NLINKSPERHALFCRU + currentlinkindex; // global link ID [0..1079]
    int halfChamberId = HelperMethods::getHCIDFromLinkID(linkIdxGlobal);
    // int halfChamberId = mLinkMap->getHCID(linkIdxGlobal); FIXME: uncomment, when object available in CCDB
    // TODO we keep detector, stack, layer, sector, side for now to be compatible to the current code state,
    // but halfChamberId contains everything we need to know... More cleanup to be done in second step
    int detectorId = halfChamberId / 2;
    int stack = HelperMethods::getStack(detectorId);
    int layer = HelperMethods::getLayer(detectorId);
    int sector = HelperMethods::getSector(detectorId);
    int side = halfChamberId % 2;
    int stack_layer = stack * NLAYER + layer; // similarly this is also only for graphing so just use the rdh ones for now.
    mEventRecords.incLinkErrorFlags(mFEEID.supermodule, side, stack_layer, mCurrentHalfCRULinkErrorFlags[currentlinkindex]);
    int currentlinksize32 = mCurrentHalfCRULinkLengths[currentlinkindex] * 8; // x8 to go from 256 bits to 32 bit;
    int endOfCurrentLink = mHBFoffset32 + currentlinksize32;

    linksizeAccum32 += currentlinksize32;
    if (currentlinksize32 == 0) {
      mEventRecords.incLinkNoData(detectorId, side, stack_layer);
    }
    if (mOptions[TRDVerboseBit]) {
      if (currentlinksize32 > 0) {
        LOGF(info, "Half-CRU link %i raw dump before parsing starts:", currentlinkindex);
        for (uint32_t dumpoffset = mHBFoffset32; dumpoffset < mHBFoffset32 + currentlinksize32; dumpoffset += 8) {
          LOGF(info, "0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x", mHBFPayload[dumpoffset], mHBFPayload[dumpoffset + 1], mHBFPayload[dumpoffset + 2], mHBFPayload[dumpoffset + 3], mHBFPayload[dumpoffset + 4], mHBFPayload[dumpoffset + 5], mHBFPayload[dumpoffset + 6], mHBFPayload[dumpoffset + 7]);
        }
      } else {
        LOGF(info, "Half-CRU link %i has zero link size", currentlinkindex);
      }
    }
    if (currentlinksize32 > 0) { // if link is not empty
      auto trackletparsingstart = std::chrono::high_resolution_clock::now();
      if (mOptions[TRDVerboseBit]) {
        LOGF(info, "Tracklet parser starting at offset %u and processing up to %u words", mHBFoffset32, currentlinksize32);
      }
      int trackletWordsRejected = 0;
      // LOGF(warn, "Starting tracklet parsing for HCID %i", halfChamberId);
      int trackletWordsRead = parseTrackletLinkData(currentlinksize32, halfChamberId, trackletWordsRejected);
      if (trackletWordsRead == -1) {
        //something went wrong bailout of here.
        if (mMaxWarnPrinted > 0) {
          LOG(warn) << "Tracklet parser returned -1 for link " << currentlinkindex;
          checkNoWarn();
        }
        mHBFoffset32 = hbfOffsetTmp + linksizeAccum32;
        incrementErrors(TrackletsReturnedMinusOne, halfChamberId);
        continue; // move to next link of this half-CRU
      }
      std::chrono::duration<double, std::micro> trackletparsingtime = std::chrono::high_resolution_clock::now() - trackletparsingstart;
      mEventRecords.getCurrentEventRecord().incTrackletTime((double)std::chrono::duration_cast<std::chrono::microseconds>(trackletparsingtime).count());
      if (mOptions[TRDVerboseBit]) {
        LOGF(info, "Read %i tracklet words and rejected %i words", trackletWordsRead, trackletWordsRejected);
      }
      mHBFoffset32 += trackletWordsRead;
      mTrackletWordsRejected += trackletWordsRejected;
      mTrackletWordsRead += trackletWordsRead;
      mEventRecords.getCurrentEventRecord().incWordsRead(trackletWordsRead);
      mEventRecords.getCurrentEventRecord().incWordsRejected(trackletWordsRejected);
      mEventRecords.incLinkWordsRead(mFEEID.supermodule, side, stack_layer, trackletWordsRead);
      mEventRecords.incLinkWordsRejected(mFEEID.supermodule, side, stack_layer, trackletWordsRejected);

      /****************
      ** DIGITS NOW ***
      *****************/
      // Check if we have a calibration trigger ergo we do actually have digits data. check if we are now at the end of the data due to bugs, i.e. if trackletparsing read padding words.
      if (mHBFoffset32 != endOfCurrentLink &&
          (mCurrentHalfCRUHeader.EventType == ETYPECALIBRATIONTRIGGER || mOptions[TRDIgnore2StageTrigger]) &&
          (mHBFPayload[mHBFoffset32] != CRUPADDING32)) {
        // we still have data on this link, we have a calibration trigger (or ignore the event type) and we are not reading a padding word

        uint32_t offsetBeforeDigitParsing = mHBFoffset32;
        // the digit HC headers come first
        if (!parseDigitHCHeaders(halfChamberId)) {
          mHBFoffset32 = hbfOffsetTmp + linksizeAccum32;
          continue; // move to next link of this half-CRU
        }
        checkDigitHCHeader(halfChamberId);
        if (mHBFoffset32 - offsetBeforeDigitParsing != 1U + mDigitHCHeader.numberHCW) {
          if (mMaxErrsPrinted > 0) {
            LOGF(error, "Did not read as many digit headers (%i) as expected (%i)",
                 mHBFoffset32 - offsetBeforeDigitParsing, mDigitHCHeader.numberHCW + 1);
            checkNoErr();
          }
          mHBFoffset32 = hbfOffsetTmp + linksizeAccum32;
          continue; // move to next link of this half-CRU
        }

        mEventRecords.incMajorVersion(mDigitHCHeader.major); // 127 is max histogram goes to 256

        if (mDigitHCHeader.major == 0x47) {
          // config event so ignore for now and bail out of parsing.
          //advance data pointers to the end;
          mHBFoffset32 = hbfOffsetTmp + currentlinksize32;
          mDigitWordsRejected += hbfOffsetTmp + currentlinksize32; // count full link as rejected
          LOG(info) << "Configuration event  ";
        } else {
          auto digitsparsingstart = std::chrono::high_resolution_clock::now();
          int digitWordsRejected = 0;
          int digitWordsRead = parseDigitLinkData(endOfCurrentLink - mHBFoffset32, halfChamberId, digitWordsRejected);
          std::chrono::duration<double, std::micro> digitsparsingtime = std::chrono::high_resolution_clock::now() - digitsparsingstart;
          mEventRecords.getCurrentEventRecord().incDigitTime((double)std::chrono::duration_cast<std::chrono::microseconds>(digitsparsingtime).count());
          mEventRecords.getCurrentEventRecord().incWordsRead(digitWordsRead);
          mEventRecords.getCurrentEventRecord().incWordsRejected(digitWordsRejected);
          mEventRecords.incLinkWordsRead(mFEEID.supermodule, side, stack_layer, digitWordsRead);
          mEventRecords.incLinkWordsRejected(mFEEID.supermodule, side, stack_layer, digitWordsRejected);

          if (mOptions[TRDVerboseBit]) {
            LOGF(info, "Read %i digit words and rejected %i words", digitWordsRead, digitWordsRejected);
          }

          mHBFoffset32 += digitWordsRead + digitWordsRejected; // all 3 in 32bit units
          mDigitWordsRead += digitWordsRead;
          mDigitWordsRejected += digitWordsRejected;
        }
      }
    } else {
      // OS: link is empty, what does this block mean???
      if (mCurrentHalfCRUHeader.EventType == ETYPEPHYSICSTRIGGER) {
        mEventRecords.incMajorVersion(128); // 127 is max histogram goes to 256
      }
    }
    if (mHBFoffset32 != hbfOffsetTmp + linksizeAccum32) {
      // if only tracklet data is available the tracklet parser advances to the tracklet end marker, but there can still be padding words on the link
      LOGF(debug, "After processing link %i the payload offset of %u is not the expected %u + %u", currentlinkindex, mHBFoffset32, hbfOffsetTmp, linksizeAccum32);
      mHBFoffset32 = hbfOffsetTmp + linksizeAccum32;
    }
  } // for loop over link index.
  if (mHBFoffset32 != hbfOffsetTmp + totalHalfCRUDataLength32) {
    LOGF(debug, "After processing half-CRU data the offset (%u) is not pointing to the expected position (%u + %u = %u)", mHBFoffset32, hbfOffsetTmp, totalHalfCRUDataLength32, totalHalfCRUDataLength32 + hbfOffsetTmp);
    mHBFoffset32 = hbfOffsetTmp + totalHalfCRUDataLength32;
  }
  // we have read in all the digits and tracklets for this event.
  //digits and tracklets are sitting inside the parsing classes.
  //extract the vectors and copy them to tracklets and digits here, building the indexing(triggerrecords)
  //as this is for a single cru half chamber header all the tracklets and digits are for the same trigger defined by the bc and orbit in the rdh which we hold in mIR

  std::chrono::duration<double, std::milli> cruparsingtime = std::chrono::high_resolution_clock::now() - crustart;
  mEventRecords.getCurrentEventRecord().incTime(cruparsingtime.count());

  //if we get here all is ok.
  return true;
}

bool CruRawReader::isTrackletHCHeaderOK(const TrackletHCHeader& header, int& hcid)
{
  if (header.one != 1) {
    return false;
  }
  int detHeader = HelperMethods::getDetector(((~header.supermodule) & 0x1f), ((~header.stack) & 0x7), ((~header.layer) & 0x7));
  int hcidHeader = (detHeader * 2 + ((~header.side) & 0x1));

  if (mOptions[TRDIgnoreBogusTrackletHCHeaders]) {
    // in the current synthetic data sample the tracklet HC headers are screwed up
    // this option should be removed when we have new synthetic data samples available
    // with fixed headers
    return true;
  }

  if (hcid != hcidHeader) {
    /* FIXME currently would be flooded by these messages, as long as CCDB object not available
    if (mMaxWarnPrinted > 0) {
      LOGF(alarm, "RDH HCID %i, TrackletHCHeader HCID %i. Taking the TrackletHCHedaer as authority", hcid, hcidHeader);
      checkNoWarn();
    }
    */
    hcid = hcidHeader;
  }
  return (hcid == hcidHeader);
}

int CruRawReader::parseDigitLinkData(int maxWords32, int hcid, int& wordsRejected)
{
  int wordsRead = 0;
  int numberOfDigitsFound = 0;
  int state = StateDigitMCMHeader;
  DigitMCMHeader mcmHeader;
  DigitMCMADCMask adcMask;

  // data is expected to be ordered
  int previousMcm = -1;
  int previousRob = -1;
  // TODO add check for event counter of DigitMCMHeader?
  // are the counters expected to be the same for all MCMs for one trigger?

  while (wordsRead < maxWords32 && state != StateFinished) {
    uint32_t currWord = mHBFPayload[mHBFoffset32 + wordsRead];

    if (state == StateDigitMCMHeader) {
      ++wordsRead;
      if (currWord == DIGITENDMARKER) {
        state = StateSecondEndmarker;
        continue;
      }
      mcmHeader.word = currWord;
      if (!sanityCheckDigitMCMHeader(mcmHeader)) {
        incrementErrors(DigitMCMHeaderSanityCheckFailure, hcid, fmt::format("DigitMCMHeader {:#010x} failed sanity check", currWord));
        ++wordsRejected;
        state = StateMoveToEndMarker; // give up for this link and try to find end markers
        continue;
      }
      // check correct ordering of link data
      if (previousMcm >= 0) {
        // first DigitMCMHeader we see for this link
        if (previousRob > mcmHeader.rob || (mcmHeader.rob == previousRob && mcmHeader.mcm < previousMcm)) {
          incrementErrors(DigitMCMNotIncreasing, hcid, fmt::format("Current rob/mcm = {}/{}, previous rob/mcm = {}/{}", (int)mcmHeader.rob, (int)mcmHeader.mcm, previousRob, previousMcm));
        } else if (previousRob == mcmHeader.rob && previousMcm == mcmHeader.mcm) {
          incrementErrors(DigitMCMDuplicate, hcid, fmt::format("Second MCM header {:#010x} for rob/mcm = {}/{}", currWord, previousRob, previousMcm));
        }
      } else {
        previousMcm = mcmHeader.mcm;
        previousRob = mcmHeader.rob;
      }
      if (mDigitHCHeader.major & 0x20) {
        // zero suppression (ZS) is ON, we expect ADC mask next
        state = StateDigitADCMask;
      } else {
        // full readout, expect ADC data next
        state = StateDigitMCMData;
      }
      LOGF(debug, "Got DigitMCMHeader 0x%08x", currWord);
      continue;
    }

    else if (state == StateDigitADCMask) {
      ++wordsRead;
      adcMask.word = currWord;
      if (!sanityCheckDigitMCMADCMask(adcMask)) {
        incrementErrors(DigitADCMaskInvalid, hcid, fmt::format("DigitADCMask {:#010x} failed sanity check", currWord));
        ++wordsRejected;
        state = StateMoveToEndMarker; // give up for this link and try to find end markers
        continue;
      }
      state = StateDigitMCMData;
      LOGF(debug, "Got DigitADCMask 0x%08x", currWord);
      continue;
    }

    else if (state == StateDigitMCMData) {
      std::array<uint16_t, TIMEBINS> adcValues;
      bool exitChannelLoop = false;
      state = StateDigitMCMHeader; // after we are done reading the ADC data, by default we expect another MCM header
      for (int iChannel = 0; iChannel < NADCMCM; ++iChannel) {
        if (!(mDigitHCHeader.major & 0x20) || adcMask.adcmask & (1UL << iChannel)) {
          // either ZS is OFF, or the adcMask has iChannel flagged as active
          DigitMCMData data;
          int timebin = 0;
          while (timebin < mTimeBins) {
            if (currWord == DIGITENDMARKER) {
              incrementErrors(DigitEndMarkerWrongState, hcid, "Expected Digit ADC data, but found end marker instead");
              exitChannelLoop = true;
              state = StateSecondEndmarker;
              ++wordsRead;
              wordsRejected += timebin / 3; // we are rejecting all ADC data we have already read for this channel
              break;
            }
            data.word = currWord;
            if ((((iChannel % 2) == 0) && (data.f != 0x3)) || ((iChannel % 2) && (data.f != 0x2))) {
              incrementErrors(DigitSanityCheck, hcid, fmt::format("Current channel {}, check bits {}. Word {:#010x}", iChannel, (int)data.f, currWord));
              exitChannelLoop = true;
              state = StateMoveToEndMarker;
              ++wordsRead;
              ++wordsRejected;
              break;
            }
            adcValues[timebin++] = data.z;
            adcValues[timebin++] = data.y;
            adcValues[timebin++] = data.x;
            ++wordsRead;
            currWord = mHBFPayload[mHBFoffset32 + wordsRead];
          } // end time bin loop
          if (exitChannelLoop) {
            break;
          }
          mEventRecords.getCurrentEventRecord().addDigit(Digit(hcid / 2, (int)mcmHeader.rob, (int)mcmHeader.mcm, iChannel, adcValues));
          mEventRecords.getCurrentEventRecord().incDigitsFound(1);
          ++mDigitsFound;
        } // end active channel
      }   // end channel loop
      continue;
    }

    else if (state == StateMoveToEndMarker) {
      ++wordsRead;
      if (currWord == DIGITENDMARKER) {
        state = StateSecondEndmarker;
      } else {
        ++wordsRejected;
      }
      continue;
    } // StateMoveToEndMarker

    else if (state == StateSecondEndmarker) {
      ++wordsRead;
      if (currWord != DIGITENDMARKER) {
        if (mMaxWarnPrinted > 0) {
          LOGF(warn, "Expected second end marker, but found 0x%08x instead", currWord);
          checkNoWarn();
        }
        return -1;
      }
      state = StateFinished;
      continue;
    } // StateSecondEndmarker
  }

  if (state == StateFinished) {
    // all good, we exited the state machine in the expected state
    return wordsRead;
  } else {
    // not good, something went wrong with digit parsing
    // e.g. we tried to move to the end marker but reached the link size
    //      without finding one.
    if (mMaxWarnPrinted > 0) {
      LOG(warn) << "We exited the digit parser state machine, but we are not in the state finished";
      checkNoWarn();
    }
    incrementErrors(DigitParsingExitInWrongState, hcid, "Done with digit parsing but state is not StateFinished");
    return -1;
  }
}

// Returns number of words read (>=0) or error state (<0)
int CruRawReader::parseTrackletLinkData(int linkSize32, int& hcid, int& wordsRejected)
{
  int wordsRead = 0;                 // count the number of words which were parsed (both successful and not successful)
  int numberOfTrackletsFound = 0;    // count the number of found tracklets for this link
  int state = StateTrackletHCHeader; // we expect to always see a TrackletHCHeader at the beginning of the link
  // tracklet data for one link is expected to arrive ordered
  // first MCM in row=0, column=0, then row=0, column=1, ... row=1, column=0, ...
  int previousColumn = -1;
  int previousRow = -1;
  TrackletHCHeader hcHeader;
  TrackletMCMHeader mcmHeader;

  // main loop we exit only when we reached the end of the link or have seen two tracklet end markers
  while (wordsRead < linkSize32 && state != StateFinished) {
    uint32_t currWord = mHBFPayload[mHBFoffset32 + wordsRead];

    if (state == StateTrackletHCHeader) {
      ++wordsRead;
      if (mTrackletHCHeaderState == 0) {
        LOG(error) << "It's not allowed anymore to have the TrackletHCHeader never present";
        return -1;
      } else if (mTrackletHCHeaderState == 1) {
        // in case tracklet data is present we have a TrackletHCHeader, otherwise we should
        // see a DigitHCHeader
        hcHeader.word = currWord;
        if (!isTrackletHCHeaderOK(hcHeader, hcid)) {
          // either the TrackletHCHeader is corrupt or we have only digits on this link
          return 0;
        }
        state = StateTrackletMCMHeader; // we do expect tracklets following
      } else {
        // we always expect a TrackletHCHeader as first word for a link (default)
        hcHeader.word = currWord;
        if (!isTrackletHCHeaderOK(hcHeader, hcid)) {
          // we have a corrupt TrackletHCHeader
          incrementErrors(TrackletHCHeaderFailure, hcid, fmt::format("Invalid word {:#010x} for the expected TrackletHCHeader", currWord));
          state = StateMoveToEndMarker;
          ++wordsRejected;
          continue;
        }
        state = StateTrackletMCMHeader; // we might have tracklets following or tracklet end markers
      }
    } // StateTrackletHCHeader

    else if (state == StateTrackletMCMHeader) {
      if (currWord == TRACKLETENDMARKER) {
        state = StateSecondEndmarker; // we expect a second tracklet end marker to follow
      } else {
        mcmHeader.word = currWord;
        if (!sanityCheckTrackletMCMHeader(mcmHeader)) {
          incrementErrors(TrackletMCMHeaderSanityCheckFailure, hcid, fmt::format("Invalid word {:#010x} for the expected TrackletMCMHeader", currWord));
          state = StateMoveToEndMarker; // invalid MCM header, no chance to interpret the following MCM data
          ++wordsRead;
          ++wordsRejected;
          continue;
        }
        // check ordering by MCM index
        if (previousColumn >= 0) {
          if (mcmHeader.padrow < previousRow || (mcmHeader.padrow == previousRow && mcmHeader.col < previousColumn)) {
            incrementErrors(TrackletDataWrongOrdering, hcid, fmt::format("Current padrow/column = {}/{}, previous padrow/column = {}/{}", (int)mcmHeader.padrow, (int)mcmHeader.col, previousRow, previousColumn));
          } else if (mcmHeader.padrow == previousRow && mcmHeader.col == previousColumn) {
            incrementErrors(TrackletDataDuplicateMCM, hcid, fmt::format("Second MCM header {:#010x} for padrow/column = {}/{}", currWord, previousRow, previousColumn));
          }
        } else {
          previousColumn = mcmHeader.col;
          previousRow = mcmHeader.padrow;
        }
        state = StateTrackletMCMData; // tracklet words must be following, unless the HC header format indicates sending of empty MCM headers
      }
      ++wordsRead;
    } // StateTrackletMCMHeader

    else if (state == StateTrackletMCMData) {
      bool addedTracklet = false; // flag whether we actually found a tracklet
      for (int iCpu = 0; iCpu < 3; ++iCpu) {
        if (((mcmHeader.word >> (1 + iCpu * 8)) & 0xff) == 0xff) {
          // iCpu did not produce a tracklet.
          // Since no empty tracklet words are sent, we don't need to move to the next word.
          // Instead, we check if the next CPU is supposed to have sent a tracklet
          continue;
        }
        if (currWord == TRACKLETENDMARKER) {
          if ((hcHeader.format & 0x2) == 0x2) {
            // format indicates no empty MCM headers are sent, so we should not see an end marker here
            // TODO add error message/counter for QC
            LOGF(warn, "For the MCM Header 0x%08x we expected a tracklet from CPU %i, but got an endmarker instead", mcmHeader.word, iCpu);
          }
          state = StateSecondEndmarker; // we expect a second tracklet end marker to follow
          break;
        }
        if ((currWord & 0x1) == 0x1) {
          // the reserved bit of the trackler MCM data is set
          // TODO add error message/counter for QC
          LOGF(warn, "Invalid word 0x%08x for the expected TrackletMCMData", currWord);
          ++wordsRejected;
        }
        TrackletMCMData mcmData;
        mcmData.word = currWord;
        mEventRecords.getCurrentEventRecord().addTracklet(assembleTracklet64(hcHeader.format, mcmHeader, mcmData, iCpu, hcid));
        mEventRecords.getCurrentEventRecord().incTrackletsFound(1);
        ++numberOfTrackletsFound;
        ++mTrackletsFound;
        addedTracklet = true;
        ++wordsRead;
        if (wordsRead == linkSize32) {
          incrementErrors(TrackletNoTrackletEndMarker, hcid, fmt::format("After reading the word {:#010x} we are at the end of the link data", currWord));
          return wordsRead;
        }
        currWord = mHBFPayload[mHBFoffset32 + wordsRead];
      }
      if (state == StateSecondEndmarker) {
        ++wordsRead;
        continue;
      }
      if (!addedTracklet) {
        // we did not add a tracklet -> do we expect empty MCM headers?
        if ((hcHeader.format & 0x2) != 0x2) {
          // yes, this is OK and the next word should be MCM header or end marker
          ++wordsRead;
          state = StateTrackletMCMHeader;
          continue;
        } else {
          // no tracklet was found, but we should have found one
          ++wordsRead;
          ++wordsRejected;
          state = StateMoveToEndMarker;
          continue;
        }
      }
      state = StateTrackletMCMHeader;
      continue;
    } // StateTrackletMCMData

    else if (state == StateSecondEndmarker) {
      ++wordsRead;
      if (currWord != TRACKLETENDMARKER) {
        if (mMaxWarnPrinted > 0) {
          LOGF(warn, "Expected second end marker, but found 0x%08x instead", currWord);
          checkNoWarn();
        }
        return -1;
      }
      state = StateFinished;
    } // StateSecondEndmarker

    else if (state == StateMoveToEndMarker) {
      ++wordsRead;
      if (currWord == TRACKLETENDMARKER) {
        state = StateSecondEndmarker;
      } else {
        ++wordsRejected;
      }
      continue;
    } // StateMoveToEndMarker

    else {
      // should never happen
      LOG(error) << "Tracklet parser ended up in unknown state";
    }

  } // end of state machine

  if (mTrackletHCHeaderState == 1 && numberOfTrackletsFound == 0) {
    if (mMaxErrsPrinted > 0) {
      LOG(error) << "We found a TrackletHCHeader in mode 1, but did not add any tracklets";
      checkNoErr();
    }
  }

  if (state == StateFinished) {
    // all good, we exited the state machine in the expected state
    return wordsRead;
  } else {
    // not good, something went wrong with tracklet parsing
    // e.g. we tried to move to the end marker but reached the link size
    //      without finding one.
    incrementErrors(TrackletExitingNoTrackletEndMarker, hcid, "Exiting tracklet parsing not in the state finished");
    return -1;
  }
}

Tracklet64 CruRawReader::assembleTracklet64(int format, TrackletMCMHeader& mcmHeader, TrackletMCMData& mcmData, int cpu, int hcid) const
{
  // TODO take care of special tracklet formats?
  uint16_t pos = mcmData.pos;
  uint16_t slope = mcmData.slope;
  // The 8-th bit of position and slope are always flipped in the FEE.
  // We flip them back while reading the raw data so that they are stored
  // without flipped bits in the CTFs
  pos ^= 0x80;
  slope ^= 0x80;
  uint32_t hpid = (mcmHeader.word >> (1 + cpu * 8)) & 0xff;
  uint32_t lpid = mcmData.pid;
  uint32_t pid = (hpid << 12) | lpid;
  int q0, q1, q2;
  if (format & 0x1) {
    // DQR enabled
    int scaleFactor = (hpid >> 6) & 0x3;
    q0 = (scaleFactor << 6) | (lpid & 0x3f);
    q1 = (scaleFactor << 6) | ((lpid >> 6) & 0x3f);
    q2 = (scaleFactor << 6) | (hpid & 0x3f);
  } else {
    // DQR disabled
    q0 = lpid & 0x7f;
    q1 = ((hpid & 0x3) << 5) | ((lpid >> 7) & 0x1f);
    q2 = (hpid >> 2) & 0x3f;
  }
  return Tracklet64(format, hcid, mcmHeader.padrow, mcmHeader.col, pos, slope, q0, q1, q2);
}

void CruRawReader::dumpInputPayload() const
{
  // we print 8 32-bit words per line
  LOG(info) << "Dumping full input payload ----->";
  for (int iWord = 0; iWord < (mDataBufferSize / 4); iWord += 8) {
    LOGF(info, "Word %4i/%4i: 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x 0x%08x",
         iWord, mDataBufferSize / 4,
         *((uint32_t*)mDataBufferPtr + iWord), *((uint32_t*)mDataBufferPtr + iWord + 1), *((uint32_t*)mDataBufferPtr + iWord + 2), *((uint32_t*)mDataBufferPtr + iWord + 3),
         *((uint32_t*)mDataBufferPtr + iWord + 4), *((uint32_t*)mDataBufferPtr + iWord + 5), *((uint32_t*)mDataBufferPtr + iWord + 6), *((uint32_t*)mDataBufferPtr + iWord + 7));
  }
  LOG(info) << "<------ Done dumping full input payload";
}

void CruRawReader::run()
{
  /* FIXME: uncomment, when object available in CCDB
  if (!mLinkMap) {
    LOG(error) << "No mapping for Link ID to HCID provided from CCDB";
    return;
  }
  */
  if (mOptions[TRDVerboseBit]) {
    dumpInputPayload();
  }

  mCurrRdhPtr = mDataBufferPtr; // set the pointer to the current RDH to the beginning of the payload
  while ((mCurrRdhPtr - mDataBufferPtr) < mDataBufferSize) {

    int dataRead = processHBFs();

    if (dataRead < 0) {
      if (mMaxWarnPrinted > 0) {
        LOG(warn) << "Received invalid RDH, rejecting given HBF entirely";
        checkNoWarn();
      }
      break;
    } else if (dataRead == 0) {
      if (mMaxWarnPrinted > 0) {
        LOG(warn) << "Did not process any data for given HBF. Probably STOP bit was set in first RDH";
        checkNoWarn();
      }
      break;
    } else {
      LOG(debug) << "Done processing HBFs. Total input size was " << dataRead << " bytes (including all headers and padding words)";
    }
  }
};

//write the output data directly to the given DataAllocator from the datareader task.
void CruRawReader::buildDPLOutputs(o2::framework::ProcessingContext& pc)
{
  mEventRecords.sendData(pc, mOptions[TRDGenerateStats]);
}

void CruRawReader::reset()
{
  mEventRecords.reset();
  mTrackletsFound = 0;
  mDigitsFound = 0;
  mDigitWordsRead = 0;
  mDigitWordsRejected = 0;
  mTrackletWordsRead = 0;
  mTrackletWordsRejected = 0;
  mWordsRejected = 0;
}

void CruRawReader::checkNoWarn()
{
  if (!mOptions[TRDVerboseErrorsBit]) {
    if (--mMaxWarnPrinted == 0) {
      LOG(alarm) << "Warnings limit reached, the following ones will be suppressed";
    }
  }
}

void CruRawReader::checkNoErr()
{
  if (!mOptions[TRDVerboseErrorsBit]) {
    if (--mMaxErrsPrinted == 0) {
      LOG(error) << "Errors limit reached, the following ones will be suppressed";
    }
  }
}

} // namespace o2::trd
