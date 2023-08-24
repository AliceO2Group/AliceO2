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
  if (mOptions[TRDVerboseErrorsBit] && (ParsingErrorsString.size() - 1) != TRDLastParsingError) {
    LOG(error) << "Verbose error reporting requested, but the mapping of error code to error string is not complete";
  }
}

void CruRawReader::incrementErrors(int error, int hcid, std::string message)
{
  mEventRecords.incParsingError(error, hcid);
  if (mOptions[TRDVerboseErrorsBit] && error != NoError) {
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
  uint8_t rdhExpected = o2::raw::RDHUtils::getPacketCounter(rdhPrev) + 1; // packet counter is 8 bits
  if (o2::raw::RDHUtils::getPacketCounter(rdhCurr) != rdhExpected) {
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
  if (o2::raw::RDHUtils::getStop(rdh)) {
    if (mMaxErrsPrinted > 0) {
      LOGP(error, "First RDH for given HBF for FEE ID {:#04x} has stop bit set", o2::raw::RDHUtils::getFEEID(rdh));
      checkNoErr();
    }
    return -1;
  }

  // loop until RDH stop header
  while (!o2::raw::RDHUtils::getStop(rdh)) { // carry on till the end of the event.
    if (mOptions[TRDVerboseBit]) {
      LOG(info) << "Current RDH is as follows:";
      try {
        o2::raw::RDHUtils::printRDH(rdh);
      } catch (std::runtime_error& e) {
        LOG(error) << e.what();
      }
      LOGP(debug, "mDataBufferSize {}, mDataBufferPtr {}, mCurrRdhPtr {}, totalDataInputSize {}. Already read: {}. Current payload {}", mDataBufferSize, fmt::ptr(mDataBufferPtr), fmt::ptr(mCurrRdhPtr), totalDataInputSize, mCurrRdhPtr - mDataBufferPtr, o2::raw::RDHUtils::getMemorySize(rdh));
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

    if (totalDataInputSize + memorySize >= mDataBufferSize) {
      // the size of the current RDH is larger than it can possibly be (we still expect a STOP RDH)
      if (mMaxErrsPrinted > 0) {
        LOGP(error, "RDH memory size of {} + already read data size {} = {} >= {} (total available buffer size) from CRU with FEE ID {:#04x}",
             memorySize, totalDataInputSize, memorySize + totalDataInputSize, mDataBufferSize, mFEEID.word);
        checkNoErr();
      }
      // we drop this broken RDH block, but try to process what we have already put into mHBFPayload
      break;
    }

    // copy the contents of the current RDH into the buffer to be parsed, RDH payload is memory size minus header size
    std::memcpy((char*)&mHBFPayload[0] + mTotalHBFPayLoad, ((char*)rdh) + headerSize, rdhpayload);
    // copy the contents of the current rdh into the buffer to be parsed
    mTotalHBFPayLoad += rdhpayload;
    totalDataInputSize += offsetToNext;
    // move to next rdh
    mCurrRdhPtr += offsetToNext;
    rdh = reinterpret_cast<const o2::header::RDHAny*>(mCurrRdhPtr);
  }
  // move past the STOP RDH
  mCurrRdhPtr += o2::raw::RDHUtils::getOffsetToNext(rdh);

  if (mOptions[TRDVerboseBit]) {
    LOG(info) << "Current RDH is as follows (should have STOP bit set):";
    try {
      o2::raw::RDHUtils::printRDH(rdh);
    } catch (std::runtime_error& e) {
      LOG(error) << e.what();
    }
  }

  // at this point the entire HBF data payload is sitting in mHBFPayload and the total data count is mTotalHBFPayLoad
  int iteration = 0;
  mHBFoffset32 = 0;
  mPreviousHalfCRUHeaderSet = false;
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

bool CruRawReader::parseDigitHCHeaders(int hcid)
{
  // mHBFoffset32 is the current offset into the current buffer,
  //
  mDigitHCHeader.word = mHBFPayload[mHBFoffset32++];

  // in case DigitHCHeader1 is not available for providing the phase, flag with invalid one
  mPreTriggerPhase = INVALIDPRETRIGGERPHASE;

  // a hack used to make old data readable (e.g. Kr from 2021)
  if (mDigitHCHeader.major == 0 && mDigitHCHeader.minor == 0 && mDigitHCHeader.numberHCW == 0) {
    mDigitHCHeader.major = mHalfChamberMajor;
    mDigitHCHeader.minor = 42;
    mDigitHCHeader.numberHCW = mHalfChamberWords;
    if (mHalfChamberWords == 0 || mHalfChamberMajor == 0) {
      if (mMaxWarnPrinted > 0) {
        LOG(alarm) << "DigitHCHeader is corrupted and using a hack as workaround is not configured";
        checkNoWarn(false);
      }
      return false;
    }
  }

  // compare the half chamber ID from the digit HC header with the reference one obtained from the link ID
  int halfChamberIdHeader = mDigitHCHeader.supermodule * NHCPERSEC + mDigitHCHeader.stack * NLAYER * 2 + mDigitHCHeader.layer * 2 + mDigitHCHeader.side;
  if (hcid != halfChamberIdHeader) {
    incrementErrors(DigitHCHeaderMismatch, hcid, fmt::format("HCID mismatch detected. HCID from DigitHCHeader: {}, HCID from RDH: {}", halfChamberIdHeader, hcid));
    if (mMaxWarnPrinted > 0) {
      LOGF(warning, "HCID mismatch in DigitHCHeader detected for ref HCID %i. DigitHCHeader says HCID is %i", hcid, halfChamberIdHeader);
      checkNoWarn();
    }
    return false;
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
    switch (getDigitHCHeaderWordType(headers[headerwordcount])) {

      case 1: // header header1;
        if (headersfound.test(0)) {
          // we have a problem, we already have a Digit HC Header1, we are lost.
          if (mOptions[TRDVerboseErrorsBit]) {
            LOGF(warn, "We have more than one DigitHCHeader of type 1. Current word in hex %08x", headers[headerwordcount]);
            printDigitHCHeader(mDigitHCHeader, headers.data());
          }
          incrementErrors(DigitHCHeader1Problem, hcid);
          return false;
        }
        DigitHCHeader1 header1;
        header1.word = headers[headerwordcount];
        mPreTriggerPhase = header1.ptrigphase;

        headersfound.set(0);
        if ((header1.numtimebins > TIMEBINS) || (header1.numtimebins < 3)) {
          if (mOptions[TRDVerboseErrorsBit]) {
            LOGF(warn, "According to Digit HC Header 1 there are %i time bins configured", (int)header1.numtimebins);
            printDigitHCHeader(mDigitHCHeader, headers.data());
          }
          incrementErrors(DigitHCHeader1Problem, hcid);
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
          return false;
        }
        /* Currently we don't do anything with the information stored in DigitHCHeader2
        DigitHCHeader2 header2;
        header2.word = headers[headerwordcount];
        */
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
          return false;
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
      LOG(info) << "blank rdh payload data at " << mHBFoffset32 << ": 0x" << std::hex << mHBFPayload[mHBFoffset32] << " and 0x" << mHBFPayload[mHBFoffset32 + 1];
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

  if (mPreviousHalfCRUHeaderSet) {
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
  mPreviousHalfCRUHeaderSet = true;

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
  if (mCurrentHalfCRUHeader.EventType == ETYPECALIBRATIONTRIGGER) {
    mEventRecords.getCurrentEventRecord().setIsCalibTrigger();
  }

  //loop over links
  uint32_t linksizeAccum32 = 0;     // accumulated size of all links in 32-bit words
  auto hbfOffsetTmp = mHBFoffset32; // store current position at the beginning of the half-CRU payload data
  for (int currentlinkindex = 0; currentlinkindex < NLINKSPERHALFCRU; currentlinkindex++) {
    bool linkOK = true;                                                   // flag links which could be processed successfully, without any rejected word
    int cruIdx = mFEEID.supermodule * 2 + mFEEID.side;                    // 2 CRUs per SM, side defining A/C-side CRU
    int halfCruIdx = cruIdx * 2 + mFEEID.endpoint;                        // endpoint (0 or 1) defines half-CRU
    int linkIdxGlobal = halfCruIdx * NLINKSPERHALFCRU + currentlinkindex; // global link ID [0..1079]
    int halfChamberId = mLinkMap->getHCID(linkIdxGlobal);
    mEventRecords.getCurrentEventRecord().getCounters().mLinkWords[halfChamberId] = mCurrentHalfCRULinkLengths[currentlinkindex];
    mEventRecords.getCurrentEventRecord().getCounters().mLinkErrorFlag[halfChamberId] = mCurrentHalfCRULinkErrorFlags[currentlinkindex];
    mEventRecords.incLinkErrorFlags(halfChamberId, mCurrentHalfCRULinkErrorFlags[currentlinkindex]); // TODO maybe has more meaning on a per event basis?
    mEventRecords.incLinkWords(halfChamberId, mCurrentHalfCRULinkLengths[currentlinkindex]);
    uint32_t currentlinksize32 = mCurrentHalfCRULinkLengths[currentlinkindex] * 8; // x8 to go from 256 bits to 32 bit;
    uint32_t endOfCurrentLink = mHBFoffset32 + currentlinksize32;

    linksizeAccum32 += currentlinksize32;
    if (currentlinksize32 == 0) {
      mEventRecords.incLinkNoData(halfChamberId);
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
      int trackletWordsRead = parseTrackletLinkData(currentlinksize32, halfChamberId, trackletWordsRejected);
      std::chrono::duration<float, std::micro> trackletparsingtime = std::chrono::high_resolution_clock::now() - trackletparsingstart;
      if (trackletWordsRead == -1) {
        // something went wrong bailout of here.
        mHBFoffset32 = hbfOffsetTmp + linksizeAccum32;
        incrementErrors(TrackletsReturnedMinusOne, halfChamberId);
        continue; // move to next link of this half-CRU
      }
      if (trackletWordsRejected > 0) {
        linkOK = false;
      }
      mHBFoffset32 += trackletWordsRead;
      if (mCurrentHalfCRUHeader.EventType == ETYPEPHYSICSTRIGGER &&
          endOfCurrentLink - mHBFoffset32 >= 8) {
        if (mMaxWarnPrinted > 0) {
          LOGF(warn, "After successfully parsing the tracklet data for link %i there are %u words remaining which did not get parsed", currentlinkindex, endOfCurrentLink - mHBFoffset32);
          checkNoWarn();
        }
        incrementErrors(UnparsedTrackletDataRemaining, halfChamberId, fmt::format("On link {} there are {} words remaining which did not get parsed", currentlinkindex, endOfCurrentLink - mHBFoffset32));
        linkOK = false;
      }
      mEventRecords.getCurrentEventRecord().incTrackletTime(trackletparsingtime.count());
      if (mOptions[TRDVerboseBit]) {
        LOGF(info, "Read %i tracklet words and rejected %i words", trackletWordsRead, trackletWordsRejected);
      }
      mTrackletWordsRejected += trackletWordsRejected;
      mTrackletWordsRead += trackletWordsRead;
      mEventRecords.incLinkWordsRead(halfChamberId, trackletWordsRead);
      mEventRecords.incLinkWordsRejected(halfChamberId, trackletWordsRejected);

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
          std::chrono::duration<float, std::micro> digitsparsingtime = std::chrono::high_resolution_clock::now() - digitsparsingstart;
          if (digitWordsRead == -1) {
            // something went wrong bailout of here.
            mHBFoffset32 = hbfOffsetTmp + linksizeAccum32;
            continue; // move to next link of this half-CRU
          }
          mHBFoffset32 += digitWordsRead;
          if (endOfCurrentLink - mHBFoffset32 >= 8) {
            // check if some data is lost (probably due to bug in CRU user logic)
            // we should have max 7 padding words to align the link to 256 bits
            if (mMaxWarnPrinted > 0) {
              LOGF(warn, "After successfully parsing the digit data for link %i there are %u words remaining which did not get parsed", currentlinkindex, endOfCurrentLink - mHBFoffset32);
              checkNoWarn();
            }
            incrementErrors(UnparsedDigitDataRemaining, halfChamberId, fmt::format("On link {} there are {} words remaining which did not get parsed", currentlinkindex, endOfCurrentLink - mHBFoffset32));
            linkOK = false;
          }
          if (digitWordsRejected > 0) {
            linkOK = false;
          }
          mEventRecords.getCurrentEventRecord().incDigitTime(digitsparsingtime.count());
          mEventRecords.incLinkWordsRead(halfChamberId, digitWordsRead);
          mEventRecords.incLinkWordsRejected(halfChamberId, digitWordsRejected);

          if (mOptions[TRDVerboseBit]) {
            LOGF(info, "Read %i digit words and rejected %i words", digitWordsRead, digitWordsRejected);
          }
          mDigitWordsRead += digitWordsRead;
          mDigitWordsRejected += digitWordsRejected;
        }
      }
      if (linkOK) {
        incrementErrors(NoError, halfChamberId);
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

  std::chrono::duration<float, std::micro> cruparsingtime = std::chrono::high_resolution_clock::now() - crustart;
  mEventRecords.getCurrentEventRecord().incTime(cruparsingtime.count());

  //if we get here all is ok.
  return true;
}

bool CruRawReader::isTrackletHCHeaderOK(const TrackletHCHeader& header, int& hcid)
{
  if (!sanityCheckTrackletHCHeader(header)) {
    return false;
  }
  int detHeader = HelperMethods::getDetector(((~header.supermodule) & 0x1f), ((~header.stack) & 0x7), ((~header.layer) & 0x7));
  int hcidHeader = (detHeader * 2 + ((~header.side) & 0x1));
  if (hcid != hcidHeader) {
    mHalfChamberMismatches.insert(std::make_pair(hcid, hcidHeader));
    return false;
  } else {
    mHalfChamberHeaderOK.insert(hcid);
    return true;
  }
}

int CruRawReader::parseDigitLinkData(int maxWords32, int hcid, int& wordsRejected)
{
  int wordsRead = 0;
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

    else if (state == StateMoveToDigitMCMHeader) {
      // the parsing of ADC values went wrong, we search for the next MCM header
      bool foundHeader = false;
      while (wordsRead < maxWords32 && !foundHeader) {
        if (currWord == DIGITENDMARKER) {
          ++wordsRead;
          state = StateSecondEndmarker;
          break;
        }
        DigitMCMHeader tmpHeader;
        tmpHeader.word = currWord;
        if (sanityCheckDigitMCMHeader(tmpHeader)) {
          foundHeader = true;
          state = StateDigitMCMHeader;
          break;
        }
        ++wordsRead;
        ++wordsRejected;
        currWord = mHBFPayload[mHBFoffset32 + wordsRead];
      }
      if (state == StateMoveToDigitMCMHeader) {
        // we could neither find a MCM header, nor an endmarker
        break;
      }
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
              state = StateMoveToDigitMCMHeader;
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
          mEventRecords.getCurrentEventRecord().addDigit(Digit(hcid / 2, (int)mcmHeader.rob, (int)mcmHeader.mcm, iChannel, adcValues, mPreTriggerPhase));
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
        incrementErrors(DigitParsingNoSecondEndmarker, hcid, fmt::format("Expected second digit end marker, but found {:#010x} instead", currWord));
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
    incrementErrors(DigitParsingExitInWrongState, hcid, fmt::format("Done with digit parsing but state is not StateFinished but in state {}", state));
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
            incrementErrors(TrackletDataMissing, hcid, fmt::format("For the MCM Header {:#010x} we expected a tracklet from CPU {}, but got an endmarker instead", mcmHeader.word, iCpu));
          }
          state = StateSecondEndmarker; // we expect a second tracklet end marker to follow
          break;
        }
        if ((currWord & 0x1) == 0x1) {
          // the reserved bit of the trackler MCM data is set
          incrementErrors(TrackletMCMDataFailure, hcid, fmt::format("Invalid word {:#010x} for the expected TrackletMCMData", currWord));
          ++wordsRejected;
        }
        TrackletMCMData mcmData;
        mcmData.word = currWord;
        mEventRecords.getCurrentEventRecord().addTracklet(assembleTracklet64(hcHeader.format, mcmHeader, mcmData, iCpu, hcid));
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
        incrementErrors(TrackletNoSecondEndMarker, hcid, fmt::format("Expected second tracklet end marker, but found {:#010x} instead", currWord));
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
  // The combined 20 bit PID information from the MCM header and the tracklet word
  // is given by ((hpid << 12) | lpid).
  // hpid holds the upper 8 bit and lpid the lower 12 bit.
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
  if (!mLinkMap) {
    LOG(error) << "No mapping for Link ID to HCID provided from CCDB";
    return;
  }
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
    } else {
      if (mOptions[TRDVerboseBit]) {
        LOGP(info, "Done processing HBFs. Total input size was {} bytes (including all headers and padding words, excluding 64 bytes for stop RDH)", dataRead);
      }
    }
  }
};

void CruRawReader::printHalfChamberHeaderReport() const
{
  LOG(info) << "Listing the half-chambers from which we have seen correct TrackletHCHeaders:";
  int prevSec = -1;
  int currSec = -1;
  std::string message;
  for (auto hcid : mHalfChamberHeaderOK) {
    int currDet = hcid / 2;
    currSec = HelperMethods::getSector(currDet);
    std::string side = (hcid % 2 == 0) ? "A" : "B";
    if (currSec != prevSec) {
      if (!message.empty()) {
        LOG(info) << message;
        message.clear();
      }
      prevSec = currSec;
    }
    message += fmt::format("{:#02}_{}_{}{} ", currSec, HelperMethods::getStack(currDet), HelperMethods::getLayer(currDet), side.c_str());
  }
  if (!message.empty()) {
    LOG(info) << message;
  }

  if (!mHalfChamberMismatches.empty()) {
    LOG(warn) << "Found HCID mismatch(es). Printing one by one.";
  }
  for (const auto& elem : mHalfChamberMismatches) {
    LOGF(info, "HCID deduced from RDH (link ID): %i, HCID from TrackletHCHeader: %i", elem.first, elem.second);
  }
}

//write the output data directly to the given DataAllocator from the datareader task.
void CruRawReader::buildDPLOutputs(o2::framework::ProcessingContext& pc)
{
  mEventRecords.sendData(pc, mOptions[TRDGenerateStats], mOptions[TRDSortDigits], mOptions[TRDLinkStats]);
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

void CruRawReader::checkNoWarn(bool silently)
{
  if (!mOptions[TRDVerboseErrorsBit]) {
    if (--mMaxWarnPrinted == 0) {
      if (silently) {
        // put the warning message into the log file without polluting the InfoLogger
        LOG(warn) << "Warnings limit reached, the following ones will be suppressed";
      } else {
        // makes sense only for warnings with "alarm" severity
        LOG(alarm) << "Warnings limit reached, the following ones will be suppressed";
      }
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
