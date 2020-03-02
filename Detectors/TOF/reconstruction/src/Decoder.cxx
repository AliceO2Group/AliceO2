// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "FairLogger.h"
//#define VERBOSE

namespace o2
{
namespace tof
{
namespace compressed
{
Decoder::Decoder()
{
  for (int i = 0; i < NCRU; i++) {
    mIntegratedBytes[i] = 0.;
    mBuffer[i] = nullptr;
    mCruIn[i] = false;
  }
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
    if (!mCruIn[i])
      continue;

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
    if (mFile[i].is_open())
      mFile[i].close();
  }
  return false;
}

void Decoder::readTRM(int icru, int icrate, int orbit, int bunchid)
{

  if (orbit < mFirstOrbit || (orbit == mFirstOrbit && bunchid < mFirstBunch)) {
    mFirstOrbit = orbit;
    mFirstBunch = bunchid;
  }

  if (mVerbose)
    printTRMInfo(icru);
  int nhits = mUnion[icru]->frameHeader.numberOfHits;
  int time_ext = mUnion[icru]->frameHeader.frameID << 13;
  int itrm = mUnion[icru]->frameHeader.trmID;
  int deltaBC = mUnion[icru]->frameHeader.deltaBC;

  if (deltaBC != 0)
    printf("DeltaBC = %d\n", deltaBC);
  mUnion[icru]++;
  mIntegratedBytes[icru] += 4;

  // read hits
  Int_t channel, echannel;
  Int_t tdc;
  Int_t tot;
  Int_t bc;
  Int_t time;

  std::array<int, 6> digitInfo;

  for (int i = 0; i < nhits; i++) {
    fromRawHit2Digit(icrate, itrm, mUnion[icru]->packedHit.tdcID, mUnion[icru]->packedHit.chain, mUnion[icru]->packedHit.channel, orbit, bunchid, time_ext + mUnion[icru]->packedHit.time, mUnion[icru]->packedHit.tot, digitInfo);

    mHitDecoded++;

    if (mVerbose)
      printHitInfo(icru);

    int isnext = digitInfo[3] * Geo::BC_IN_WINDOW_INV;

    if (isnext >= MAXWINDOWS) { // accumulate all digits which are not in the first windows

      insertDigitInFuture(digitInfo[0], digitInfo[1], digitInfo[2], digitInfo[3], 0, digitInfo[4], digitInfo[5]);
    } else {
      std::vector<Strip>* cstrip = mStripsCurrent; // first window
      if (isnext)
        cstrip = mStripsNext[isnext - 1]; // next window

      UInt_t istrip = digitInfo[0] / Geo::NPADS;

      // add digit
      fillDigitsInStrip(cstrip, digitInfo[0], digitInfo[1], digitInfo[2], digitInfo[3], istrip);
    }

    mUnion[icru]++;
    mIntegratedBytes[icru] += 4;
  }
}

void Decoder::fromRawHit2Digit(int icrate, int itrm, int itdc, int ichain, int channel, int orbit, int bunchid, int tdc, int tot, std::array<int, 6>& digitInfo)
{
  // convert raw info in digit info (channel, tdc, tot, bc)
  // tdc = packetHit.time + (frameHeader.frameID << 13)
  int echannel = Geo::getECHFromIndexes(icrate, itrm, ichain, itdc, channel);
  digitInfo[0] = Geo::getCHFromECH(echannel);
  digitInfo[2] = tot;

  digitInfo[3] = int(orbit * o2::tof::Geo::BC_IN_ORBIT);
  digitInfo[3] += bunchid;
  digitInfo[3] += tdc / 1024;
  digitInfo[1] = tdc % 1024;

  digitInfo[4] = orbit;
  digitInfo[5] = bunchid;
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
  mFirstOrbit = 0;
  mFirstBunch = 0;

#ifdef VERBOSE
  if (mVerbose)
    std::cout << "-------- START DECODE EVENTS IN THE HB ----------------------------------------" << std::endl;
#endif
  auto start = std::chrono::high_resolution_clock::now();

  // start from the beginning of the timeframe
  mEventTime = 0;

  // loop over CRUs
  for (int icru = 0; icru < NCRU; icru++) {
    if (!mCruIn[icru])
      continue; // no data stream available for this cru

    printf("decoding cru %d\n", icru);

    while (mUnion[icru] < mUnionEnd[icru]) { // read all the buffer
      // read open RDH
      mRDH = reinterpret_cast<o2::header::RAWDataHeader*>(mUnion[icru]);
      if (mVerbose)
        printRDH();

      // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      // note that RDH continue is not yet considered as option (to be added)
      // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      // move after RDH
      char* shift = reinterpret_cast<char*>(mRDH);
      mUnion[icru] = reinterpret_cast<Union_t*>(shift + mRDH->headerSize);
      mIntegratedBytes[icru] += mRDH->headerSize;

      if (mUnion[icru] >= mUnionEnd[icru])
        continue; // end of data stream reac
      for (int window = 0; window < Geo::NWINDOW_IN_ORBIT; window++) {
        // read Crate Header
        int bunchid = mUnion[icru]->crateHeader.bunchID;
        int icrate = mUnion[icru]->crateHeader.drmID;
        if (mVerbose)
          printCrateInfo(icru);
        mUnion[icru]++;
        mIntegratedBytes[icru] += 4;

        //read Orbit
        int orbit = mUnion[icru]->crateOrbit.orbitID;
        if (mVerbose)
          printf("orbit ID      = %d\n", orbit);
        mUnion[icru]++;
        mIntegratedBytes[icru] += 4;

        while (!mUnion[icru]->frameHeader.mustBeZero) {
          readTRM(icru, icrate, orbit, bunchid);
        }

        // read Crate Trailer
        if (mVerbose)
          printCrateTrailerInfo(icru);
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
      mRDH = reinterpret_cast<o2::header::RAWDataHeader*>(nextPage(mRDH, mRDH->memorySize));
      if (mVerbose)
        printRDH();
      mIntegratedBytes[icru] += mRDH->headerSize;

      // go to next page
      mUnion[icru] = reinterpret_cast<Union_t*>(nextPage(mRDH, mRDH->memorySize));
    }
  }

  // since digits are not yet divided in tof readout window we need to do it
  // flushOutputContainer does the job
  std::vector<Digit> digTemp;
  flushOutputContainer(digTemp);
  printf("hit decoded = %d (digits not filled = %lu)\n", mHitDecoded, mFutureDigits.size());

  return false;
}

void Decoder::printCrateInfo(int icru) const
{
  printf("___CRATE HEADER____\n");
  printf("DRM ID           = %d\n", mUnion[icru]->crateHeader.drmID);
  printf("Bunch ID         = %d\n", mUnion[icru]->crateHeader.bunchID);
  printf("Slot enable mask = %d\n", mUnion[icru]->crateHeader.slotEnableMask);
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
  printf("VERSION       = %d\n", int(mRDH->version));
  printf("BLOCK LENGTH  = %d\n", int(mRDH->blockLength));
  printf("HEADER SIZE   = %d\n", int(mRDH->headerSize));
  printf("MEMORY SIZE   = %d\n", mRDH->memorySize);
  printf("PACKET COUNTER= %d\n", mRDH->packetCounter);
  printf("CRU ID        = %d\n", mRDH->cruID);
  printf("LINK ID       = %d\n", mRDH->linkID);
  printf("___________________\n");
}
} // namespace compressed
} // namespace tof
} // namespace o2
