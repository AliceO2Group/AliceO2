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

#include "TOFReconstruction/Decoder.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include "CommonConstants/LHCConstants.h"
#include "TString.h"
#include <fairlogger/Logger.h>
#include "DetectorsRaw/RDHUtils.h"

//#define VERBOSE

namespace o2
{
namespace tof
{
namespace compressed
{
using RDHUtils = o2::raw::RDHUtils;

Decoder::Decoder()
{
  for (int i = 0; i < NCRU; i++) {
    mIntegratedBytes[i] = 0.;
    mBuffer[i] = nullptr;
    mCruIn[i] = false;
  }
  clearCounts();
}

bool Decoder::open(const std::string name)
{
  int nfileopened = NCRU;

  int fullsize = 0;
  for (int i = 0; i < NCRU; i++) {
    std::string nametmp;
    nametmp.append(Form("cru%02d", i));
    nametmp.append(name);

    if (mFile[i].is_open()) {
      std::cout << "Warning: a file (" << i << ") was already open, closing" << std::endl;
      mFile[i].close();
    }
    mFile[i].open(nametmp.c_str(), std::fstream::in | std::fstream::binary);
    if (!mFile[i].is_open()) {
      std::cout << "Cannot open " << nametmp << std::endl;
      nfileopened--;
      mSize[i] = 0;
      mCruIn[i] = false;
    } else {
      mFile[i].seekg(0, mFile[i].end);
      mSize[i] = mFile[i].tellg();
      mFile[i].seekg(0);

      fullsize += mSize[i];

      mCruIn[i] = true;
    }
  }

  if (!nfileopened) {
    std::cerr << "No streams available" << std::endl;
    return true;
  }

  //  mBufferLocal.resize(fullsize);

  printf("Full input buffer size = %d byte\n", fullsize);

  char* pos = new char[fullsize]; //mBufferLocal.data();
  for (int i = 0; i < NCRU; i++) {
    if (!mCruIn[i]) {
      continue;
    }

    mBuffer[i] = pos;

    // read content of infile
    mFile[i].read(mBuffer[i], mSize[i]);
    mUnion[i] = reinterpret_cast<Union_t*>(mBuffer[i]);
    mUnionEnd[i] = reinterpret_cast<Union_t*>(mBuffer[i] + mSize[i] - 1);

    pos += mSize[i];
  }

  printf("Number of TOF compressed streamers = %d/%d\n", nfileopened, NCRU);

  close();

  return false;
}

bool Decoder::close()
{
  for (int i = 0; i < NCRU; i++) {
    if (mFile[i].is_open()) {
      mFile[i].close();
    }
  }
  return false;
}

void Decoder::clear()
{
  reset();
  clearCounts();

  mPatterns.clear();
  mCratePatterns.clear();
  mCrateHeaderData.clear();
  mErrors.clear();
  mDiagnosticFrequency.clear();
}

void Decoder::InsertDigit(int icrate, int itrm, int itdc, int ichain, int channel, uint32_t orbit, uint16_t bunchid, int time_ext, int tdc, int tot)
{
  DigitInfo digitInfo;

  if (itrm == 3 && ((icrate % 2 == 0) || (itdc / 3 > 0))) { // Signal not coming from a TOF pad but -probably- from a TOF OR signal
    // add operation on LTM (Trigger) if needed (not used) if (crate % 2 == 1)
    return;
  }

  fromRawHit2Digit(icrate, itrm, itdc, ichain, channel, orbit, bunchid, time_ext + tdc, tot, digitInfo);

  if (digitInfo.channel < 0) { // check needed for the moment. To be removed once debugged
    return;
  }

  mChannelCounts[digitInfo.channel]++;

  mHitDecoded++;

  uint64_t isnext = digitInfo.bcAbs * Geo::BC_IN_WINDOW_INV;

  if (isnext >= uint64_t(MAXWINDOWS)) { // accumulate all digits which are not in the first windows
    insertDigitInFuture(digitInfo.channel, digitInfo.tdc, digitInfo.tot, digitInfo.bcAbs, 0, digitInfo.orbit, digitInfo.bc);
  } else {
    std::vector<Strip>* cstrip = mStripsCurrent; // first window
    if (isnext) {
      cstrip = mStripsNext[isnext - 1]; // next window
    }
    UInt_t istrip = digitInfo.channel / Geo::NPADS;

    // add digit
    fillDigitsInStrip(cstrip, digitInfo.channel, digitInfo.tdc, digitInfo.tot, digitInfo.bcAbs, istrip);
  }
}

void Decoder::readTRM(int icru, int icrate, uint32_t orbit, uint16_t bunchid)
{

  if (orbit < mFirstIR.orbit || (orbit == mFirstIR.orbit && bunchid < mFirstIR.bc)) {
    mFirstIR.orbit = orbit;
    mFirstIR.bc = bunchid;
  }

  if (mVerbose) {
    printTRMInfo(icru);
  }
  int nhits = mUnion[icru]->frameHeader.numberOfHits;
  int time_ext = mUnion[icru]->frameHeader.frameID << 13;
  int itrm = mUnion[icru]->frameHeader.trmID;
  int deltaBC = mUnion[icru]->frameHeader.deltaBC;

  if (deltaBC != 0) {
    printf("DeltaBC = %d\n", deltaBC);
  }
  mUnion[icru]++;
  mIntegratedBytes[icru] += 4;

  DigitInfo digitInfo;

  for (int i = 0; i < nhits; i++) {
    if (icrate % 2 == 1 && itrm == 3 && mUnion[icru]->packedHit.tdcID / 3 > 0) { // Signal not coming from a TOF pad but -probably- from a TOF OR signal
      // add operation on LTM (Trigger) if needed (not used)

      mUnion[icru]++;
      mIntegratedBytes[icru] += 4;
      continue;
    }

    fromRawHit2Digit(icrate, itrm, mUnion[icru]->packedHit.tdcID, mUnion[icru]->packedHit.chain, mUnion[icru]->packedHit.channel, orbit, bunchid,
                     time_ext + mUnion[icru]->packedHit.time, mUnion[icru]->packedHit.tot, digitInfo);

    if (digitInfo.channel < 0) { // check needed for the moment. To be removed once debugged
      mUnion[icru]++;
      mIntegratedBytes[icru] += 4;
      continue;
    }

    mChannelCounts[digitInfo.channel]++;

    mHitDecoded++;

    if (mVerbose) {
      printHitInfo(icru);
    }

    uint64_t isnext = digitInfo.bcAbs * Geo::BC_IN_WINDOW_INV;

    if (isnext >= MAXWINDOWS) { // accumulate all digits which are not in the first windows

      insertDigitInFuture(digitInfo.channel, digitInfo.tdc, digitInfo.tot, digitInfo.bcAbs, 0, digitInfo.orbit, digitInfo.bc);
    } else {
      std::vector<Strip>* cstrip = mStripsCurrent; // first window
      if (isnext) {
        cstrip = mStripsNext[isnext - 1]; // next window
      }

      UInt_t istrip = digitInfo.channel / Geo::NPADS;

      // add digit
      fillDigitsInStrip(cstrip, digitInfo.channel, digitInfo.tdc, digitInfo.tot, digitInfo.bcAbs, istrip);
    }

    mUnion[icru]++;
    mIntegratedBytes[icru] += 4;
  }
}

void Decoder::fromRawHit2Digit(int icrate, int itrm, int itdc, int ichain, int channel, uint32_t orbit, uint16_t bunchid, int tdc, int tot, Decoder::DigitInfo& dinfo)
{
  // convert raw info in digit info (channel, tdc, tot, bc)
  // tdc = packetHit.time + (frameHeader.frameID << 13)
  int echannel = Geo::getECHFromIndexes(icrate, itrm, ichain, itdc, channel);
  dinfo.channel = Geo::getCHFromECH(echannel);

  if (dinfo.channel < 0) { // it should not happen!
    LOG(error) << "No valid channel for icrate = " << icrate << ", itrm = " << itrm << ", ichain = " << ichain << ", itdc = " << itdc << ", channel = " << channel;
  }

  dinfo.tot = tot;
  dinfo.bcAbs = uint64_t(orbit) * o2::tof::Geo::BC_IN_ORBIT + bunchid + tdc / 1024;
  dinfo.tdc = tdc % 1024;
  dinfo.orbit = orbit;
  dinfo.bc = bunchid;
}

char* Decoder::nextPage(void* current, int shift)
{
  char* point = reinterpret_cast<char*>(current);
  point += shift;

  return point;
}

bool Decoder::decode() // return a vector of digits in a TOF readout window
{
  mReadoutWindowCurrent = 0;
  mFirstIR.orbit = 0;
  mFirstIR.bc = 0;

#ifdef VERBOSE
  if (mVerbose)
    std::cout << "-------- START DECODE EVENTS IN THE HB ----------------------------------------" << std::endl;
#endif
  auto start = std::chrono::high_resolution_clock::now();

  // start from the beginning of the timeframe

  // loop over CRUs
  for (int icru = 0; icru < NCRU; icru++) {
    if (!mCruIn[icru]) {
      continue; // no data stream available for this cru
    }

    printf("decoding cru %d\n", icru);

    while (mUnion[icru] < mUnionEnd[icru]) { // read all the buffer
      // read open RDH
      mRDH = reinterpret_cast<o2::header::RAWDataHeader*>(mUnion[icru]);
      if (mVerbose) {
        printRDH();
      }

      // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      // note that RDH continue is not yet considered as option (to be added)
      // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      // move after RDH
      char* shift = reinterpret_cast<char*>(mRDH);
      auto rdhsz = RDHUtils::getHeaderSize(*mRDH);
      mUnion[icru] = reinterpret_cast<Union_t*>(shift + rdhsz);
      mIntegratedBytes[icru] += rdhsz;

      if (mUnion[icru] >= mUnionEnd[icru]) {
        continue; // end of data stream reac
      }
      for (int window = 0; window < Geo::NWINDOW_IN_ORBIT; window++) {
        // read Crate Header
        int bunchid = mUnion[icru]->crateHeader.bunchID;
        int icrate = mUnion[icru]->crateHeader.drmID;
        if (mVerbose) {
          printCrateInfo(icru);
        }
        mUnion[icru]++;
        mIntegratedBytes[icru] += 4;

        //read Orbit
        int orbit = mUnion[icru]->crateOrbit.orbitID;
        if (mVerbose) {
          printf("%d) orbit ID      = %d -- bunch ID = %d\n", icrate, orbit, bunchid);
        }
        mUnion[icru]++;
        mIntegratedBytes[icru] += 4;

        while (!mUnion[icru]->frameHeader.mustBeZero) {
          readTRM(icru, icrate, orbit, bunchid);
        }

        // read Crate Trailer
        if (mVerbose) {
          printCrateTrailerInfo(icru);
        }
        auto ndw = mUnion[icru]->crateTrailer.numberOfDiagnostics;
        mUnion[icru]++;
        mIntegratedBytes[icru] += 4;

        // loop over number of diagnostic words
        for (int idw = 0; idw < ndw; ++idw) {
          mUnion[icru]++;
          mIntegratedBytes[icru] += 4;
        }
      }

      // read close RDH
      mRDH = reinterpret_cast<o2::header::RAWDataHeader*>(nextPage(mRDH, RDHUtils::getMemorySize(*mRDH)));
      if (mVerbose) {
        printRDH();
      }
      mIntegratedBytes[icru] += RDHUtils::getHeaderSize(*mRDH);

      // go to next page
      mUnion[icru] = reinterpret_cast<Union_t*>(nextPage(mRDH, RDHUtils::getMemorySize(*mRDH)));
    }
  }

  // since digits are not yet divided in tof readout window we need to do it
  // flushOutputContainer does the job
  fillWindows();
  fillDiagnosticFrequency();

  return false;
}

void Decoder::fillWindows()
{
  std::vector<Digit> digTemp;
  flushOutputContainer(digTemp);
}

void Decoder::printCrateInfo(int icru) const
{
  printf("___CRATE HEADER____\n");
  printf("DRM ID           = %d\n", mUnion[icru]->crateHeader.drmID);
  printf("Bunch ID         = %d\n", mUnion[icru]->crateHeader.bunchID);
  printf("Slot part. mask  = %d\n", mUnion[icru]->crateHeader.slotPartMask);
  printf("Must be ONE      = %d\n", mUnion[icru]->crateHeader.mustBeOne);
  printf("___________________\n");
}

void Decoder::printCrateTrailerInfo(int icru) const
{
  printf("___CRATE TRAILER___\n");
  printf("Event counter         = %d\n", mUnion[icru]->crateTrailer.eventCounter);
  printf("Number of diagnostics = %d\n", mUnion[icru]->crateTrailer.numberOfDiagnostics);
  printf("Must be ONE           = %d\n", mUnion[icru]->crateTrailer.mustBeOne);
  printf("___________________\n");
}

void Decoder::printTRMInfo(int icru) const
{
  printf("______TRM_INFO_____\n");
  printf("TRM ID        = %d\n", mUnion[icru]->frameHeader.trmID);
  printf("Frame ID      = %d\n", mUnion[icru]->frameHeader.frameID);
  printf("N. hits       = %d\n", mUnion[icru]->frameHeader.numberOfHits);
  printf("DeltaBC       = %d\n", mUnion[icru]->frameHeader.deltaBC);
  printf("Must be Zero  = %d\n", mUnion[icru]->frameHeader.mustBeZero);
  printf("___________________\n");
}

void Decoder::printHitInfo(int icru) const
{
  printf("______HIT_INFO_____\n");
  printf("TDC ID        = %d\n", mUnion[icru]->packedHit.tdcID);
  printf("CHAIN ID      = %d\n", mUnion[icru]->packedHit.chain);
  printf("CHANNEL ID    = %d\n", mUnion[icru]->packedHit.channel);
  printf("TIME          = %d\n", mUnion[icru]->packedHit.time);
  printf("TOT           = %d\n", mUnion[icru]->packedHit.tot);
  printf("___________________\n");
}

void Decoder::printRDH() const
{
  printf("______RDH_INFO_____\n");
  int v = RDHUtils::getVersion(*mRDH);
  printf("VERSION       = %d\n", v);
  if (v == 4) {
    printf("BLOCK LENGTH  = %d\n", int(RDHUtils::getBlockLength(mRDH)));
  }
  printf("HEADER SIZE   = %d\n", int(RDHUtils::getHeaderSize(*mRDH)));
  printf("MEMORY SIZE   = %d\n", int(RDHUtils::getMemorySize(*mRDH)));
  printf("PACKET COUNTER= %d\n", int(RDHUtils::getPacketCounter(*mRDH)));
  printf("CRU ID        = %d\n", int(RDHUtils::getCRUID(*mRDH)));
  printf("LINK ID       = %d\n", int(RDHUtils::getLinkID(*mRDH)));
  printf("___________________\n");
}
} // namespace compressed
} // namespace tof
} // namespace o2
