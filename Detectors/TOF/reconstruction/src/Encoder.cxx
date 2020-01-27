// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TOFReconstruction/Encoder.h"
#include "TOFReconstruction/Decoder.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include "CommonConstants/LHCConstants.h"
#include "CommonConstants/Triggers.h"
#include "TString.h"
#include "FairLogger.h"
#include <array>
#define VERBOSE

namespace o2
{
namespace tof
{
namespace raw
{

Encoder::Encoder()
{
  // intialize 72 buffers (one per crate)
  for (int i = 0; i < 72; i++) {
    mIntegratedBytes[i] = 0;
    mBuffer[i] = nullptr;
    mUnion[i] = nullptr;
    mStart[i] = nullptr;
    mNRDH[i] = 0;
  }
}

void Encoder::nextWord(int icrate)
{
  if (mNextWordStatus[icrate]) {
    nextWordNoEmpty(icrate);
    mUnion[icrate]->data = 0;
    nextWordNoEmpty(icrate);
    mUnion[icrate]->data = 0;
  }
  nextWordNoEmpty(icrate);
  mNextWordStatus[icrate] = !mNextWordStatus[icrate];
}

void Encoder::nextWordNoEmpty(int icrate)
{
  mUnion[icrate]++;

  // check if you went over the buffer size
  int csize = getSize(mRDH[icrate], mUnion[icrate]);
  if (csize >= Geo::RAW_PAGE_MAX_SIZE) {
    addPage(icrate);
  }
}

bool Encoder::open(const std::string name)
{
  bool status = false;

  for (int i = 0; i < NCRU; i++) {
    std::string nametmp;
    nametmp.append(Form("cru%02d", i));
    nametmp.append(name);
    printf("TOF Raw encoder: create stream to CRU: %s\n", nametmp.c_str());
    if (mFileCRU[i].is_open()) {
      std::cout << "Warning: a file (" << i << ") was already open, closing" << std::endl;
      mFileCRU[i].close();
    }
    mFileCRU[i].open(nametmp.c_str(), std::fstream::out | std::fstream::binary);
    if (!mFileCRU[i].is_open()) {
      std::cerr << "Cannot open " << nametmp << std::endl;
      status = true;
    }
  }

  return status;
}

bool Encoder::flush(int icrate)
{
  if (mIntegratedBytes[icrate]) {
    mIntegratedAllBytes += mIntegratedBytes[icrate];
    mIntegratedBytes[icrate] = 0;
    mUnion[icrate] = mStart[icrate];
  }
  return false;
}

bool Encoder::flush()
{
  bool allempty = true;
  for (int i = 0; i < 72; i++)
    if (mIntegratedBytes[i])
      allempty = false;

  if (allempty)
    return true;

  // write superpages
  for (int i = 0; i < 72; i++) {
    int icru = i / NLINKSPERCRU;
    mFileCRU[icru].write(mBuffer[i], mIntegratedBytes[i]);
  }

  for (int i = 0; i < 72; i++)
    flush(i);

  memset(mBuffer[0], 0, mSize * 72);

  return false;
}

bool Encoder::close()
{
  for (int i = 0; i < NCRU; i++)
    if (mFileCRU[i].is_open())
      mFileCRU[i].close();
  return false;
}

bool Encoder::alloc(long size)
{
  if (size < 500000)
    size = 500000;

  mSize = size;

  mBufferLocal.resize(size * 72);

  mBuffer[0] = mBufferLocal.data();
  memset(mBuffer[0], 0, mSize * 72);
  mStart[0] = reinterpret_cast<Union_t*>(mBuffer[0]);
  mUnion[0] = mStart[0]; // rewind

  for (int i = 1; i < 72; i++) {
    mBuffer[i] = mBuffer[i - 1] + size;
    mStart[i] = reinterpret_cast<Union_t*>(mBuffer[i]);
    mUnion[i] = mStart[i]; // rewind
  }
  return false;
}

void Encoder::encodeTRM(const std::vector<Digit>& summary, Int_t icrate, Int_t itrm, int& istart) // encode one TRM assuming digit vector sorted by electronic index
// return next TRM index (-1 if not in the same crate)
// start to convert digiti from istart --> then update istart to the starting position of the new TRM
{
  if (mVerbose)
    printf("Crate %d: encode TRM %d \n", icrate, itrm);

  // TRM HEADER
  mUnion[icrate]->trmDataHeader.slotId = itrm;
  mUnion[icrate]->trmDataHeader.eventWords = 0; // to be filled at the end
  mUnion[icrate]->trmDataHeader.eventCnt = mEventCounter;
  mUnion[icrate]->trmDataHeader.emptyBit = 0;
  mUnion[icrate]->trmDataHeader.dataId = 4;
  nextWord(icrate);

  // LOOP OVER CHAINS
  for (int ichain = 0; ichain < 2; ichain++) {
    // CHAIN HEADER
    mUnion[icrate]->trmChainHeader.slotId = itrm;
    mUnion[icrate]->trmChainHeader.bunchCnt = mIR.bc;
    mUnion[icrate]->trmChainHeader.mbz = 0;
    mUnion[icrate]->trmChainHeader.dataId = 2 * ichain;
    nextWord(icrate);

    while (istart < summary.size()) { // fill hits
      /** loop over hits **/
      int whatChain = summary[istart].getElChainIndex();
      if (whatChain != ichain)
        break;
      int whatTRM = summary[istart].getElTRMIndex();
      if (whatTRM != itrm)
        break;
      int whatCrate = summary[istart].getElCrateIndex();
      if (whatCrate != icrate)
        break;

      int hittimeTDC = (summary[istart].getBC() - mEventCounter * Geo::BC_IN_WINDOW) * 1024 + summary[istart].getTDC(); // time in TDC bin within the TOF WINDOW

      if (hittimeTDC < 0) {
        LOG(ERROR) << "Negative hit encoded " << hittimeTDC << ", something went wrong in filling readout window";
        printf("%d %d %d\n", summary[istart].getBC(), mEventCounter * Geo::BC_IN_WINDOW, summary[istart].getTDC());
      }
      // leading time
      mUnion[icrate]->trmDataHit.time = hittimeTDC;
      mUnion[icrate]->trmDataHit.chanId = summary[istart].getElChIndex();
      mUnion[icrate]->trmDataHit.tdcId = summary[istart].getElTDCIndex();
      mUnion[icrate]->trmDataHit.dataId = 0xa;
      nextWord(icrate);

      // trailing time
      mUnion[icrate]->trmDataHit.time = hittimeTDC + summary[istart].getTOT() * Geo::RATIO_TOT_TDC_BIN;
      mUnion[icrate]->trmDataHit.chanId = summary[istart].getElChIndex();
      mUnion[icrate]->trmDataHit.tdcId = summary[istart].getElTDCIndex();
      mUnion[icrate]->trmDataHit.dataId = 0xc;
      nextWord(icrate);

      istart++;
    }

    // CHAIN TRAILER
    mUnion[icrate]->trmChainTrailer.status = 0;
    mUnion[icrate]->trmChainTrailer.mbz = 0;
    mUnion[icrate]->trmChainTrailer.eventCnt = mEventCounter;
    mUnion[icrate]->trmChainTrailer.dataId = 1 + 2 * ichain;
    nextWord(icrate);
  }

  // TRM TRAILER
  mUnion[icrate]->trmDataTrailer.trailerMark = 3;
  mUnion[icrate]->trmDataTrailer.eventCRC = 0; // to be implemented
  mUnion[icrate]->trmDataTrailer.tempValue = 0;
  mUnion[icrate]->trmDataTrailer.tempAddress = 0;
  mUnion[icrate]->trmDataTrailer.tempChain = 0;
  mUnion[icrate]->trmDataTrailer.tempAck = 0;
  mUnion[icrate]->trmDataTrailer.lutErrorBit = 0;
  mUnion[icrate]->trmDataTrailer.dataId = 5;
  nextWord(icrate);
}

bool Encoder::encode(std::vector<std::vector<o2::tof::Digit>> digitWindow, int tofwindow) // pass a vector of digits in a TOF readout window, tof window is the entry in the vector-of-vector of digits needed to extract bunch id and orbit
{
  if (digitWindow.size() != Geo::NWINDOW_IN_ORBIT) {
    printf("Something went wrong in encoding: digitWindow=%lu different from %d\n", digitWindow.size(), Geo::NWINDOW_IN_ORBIT);
    return 999;
  }

  for (int i = 0; i < 72; i++) {
    if (mIntegratedBytes[i] + 100000 > mSize)
      flush();
  }

  for (int iwin = 0; iwin < Geo::NWINDOW_IN_ORBIT; iwin++) {
    std::vector<o2::tof::Digit>& summary = digitWindow.at(iwin);
    // caching electronic indexes in digit array
    for (auto dig = summary.begin(); dig != summary.end(); dig++) {
      int digitchannel = dig->getChannel();
      dig->setElectronicIndex(Geo::getECHFromCH(digitchannel));
    }

    // sorting by electroni indexes
    std::sort(summary.begin(), summary.end(),
              [](Digit a, Digit b) { return a.getElectronicIndex() < b.getElectronicIndex(); });
  }

#ifdef VERBOSE
  if (mVerbose)
    std::cout << "-------- START ENCODE EVENT ----------------------------------------" << std::endl;
#endif
  auto start = std::chrono::high_resolution_clock::now();

  mEventCounter = tofwindow; // tof window index
  mIR.orbit = mEventCounter / Geo::NWINDOW_IN_ORBIT;

  if (!(mIR.orbit % 256)) { // new TF
    flush();
  }

  for (int i = 0; i < 72; i++) {
    // add RDH open
    mRDH[i] = reinterpret_cast<o2::header::RAWDataHeader*>(mUnion[i]);
    mNextWordStatus[i] = false;
    openRDH(i);
  }

  // encode all windows
  for (int iwin = 0; iwin < Geo::NWINDOW_IN_ORBIT; iwin++) {
    mEventCounter = tofwindow + iwin; // tof window index

    std::vector<o2::tof::Digit>& summary = digitWindow.at(iwin);

    mIR.bc = ((mEventCounter % Geo::NWINDOW_IN_ORBIT) * Geo::BC_IN_ORBIT) / Geo::NWINDOW_IN_ORBIT; // bunch crossing in the current orbit at the beginning of the window

    int icurrentdigit = 0;
    // TOF data header
    for (int i = 0; i < 72; i++) {
      mTOFDataHeader[i] = reinterpret_cast<TOFDataHeader_t*>(mUnion[i]);
      mTOFDataHeader[i]->bytePayload = 0; // event length in byte (to be filled later)
      mTOFDataHeader[i]->mbz = 0;
      mTOFDataHeader[i]->dataId = 4;
      nextWord(i);

      mUnion[i]->tofOrbit.orbit = mIR.orbit;
      nextWord(i);

      mDRMDataHeader[i] = reinterpret_cast<DRMDataHeader_t*>(mUnion[i]);
      mDRMDataHeader[i]->slotId = 1;
      mDRMDataHeader[i]->eventWords = 0; // event length in word (to be filled later) --> word = byte/4
      mDRMDataHeader[i]->drmId = i;
      mDRMDataHeader[i]->dataId = 4;
      nextWord(i);

      mUnion[i]->drmHeadW1.slotId = 1;
      mUnion[i]->drmHeadW1.partSlotMask = (i % 2 == 0 ? 0x7fc : 0x7fe);
      mUnion[i]->drmHeadW1.clockStatus = 2;
      mUnion[i]->drmHeadW1.drmhVersion = 0x11;
      mUnion[i]->drmHeadW1.drmHSize = 5;
      mUnion[i]->drmHeadW1.mbz = 0;
      mUnion[i]->drmHeadW1.dataId = 4;
      nextWord(i);

      mUnion[i]->drmHeadW2.slotId = 1;
      mUnion[i]->drmHeadW2.enaSlotMask = (i % 2 == 0 ? 0x7fc : 0x7fe);
      mUnion[i]->drmHeadW2.mbz = 0;
      mUnion[i]->drmHeadW2.faultSlotMask = 0;
      mUnion[i]->drmHeadW2.readoutTimeOut = 0;
      mUnion[i]->drmHeadW2.dataId = 4;
      nextWord(i);

      mUnion[i]->drmHeadW3.slotId = 1;
      mUnion[i]->drmHeadW3.gbtBunchCnt = mIR.bc;
      mUnion[i]->drmHeadW3.locBunchCnt = 0;
      mUnion[i]->drmHeadW3.dataId = 4;
      nextWord(i);

      mUnion[i]->drmHeadW4.slotId = 1;
      mUnion[i]->drmHeadW4.tempValue = 0;
      mUnion[i]->drmHeadW4.mbza = 0;
      mUnion[i]->drmHeadW4.tempAddress = 0;
      mUnion[i]->drmHeadW4.mbzb = 0;
      mUnion[i]->drmHeadW4.dataId = 4;
      nextWord(i);

      mUnion[i]->drmHeadW5.slotId = 1;
      mUnion[i]->drmHeadW5.eventCRC = 0;
      mUnion[i]->drmHeadW5.irq = 0;
      mUnion[i]->drmHeadW5.mbz = 0;
      mUnion[i]->drmHeadW5.dataId = 4;
      nextWord(i);

      int trm0 = 4 - (i % 2);
      for (int itrm = trm0; itrm < 13; itrm++) {
        encodeTRM(summary, i, itrm, icurrentdigit);
      }

      mUnion[i]->drmDataTrailer.slotId = 1;
      mUnion[i]->drmDataTrailer.locEvCnt = mEventCounter;
      mUnion[i]->drmDataTrailer.mbz = 0;
      mUnion[i]->drmDataTrailer.dataId = 5;
      nextWord(i);
      mUnion[i]->data = 0x70000000;
      nextWord(i);

      mTOFDataHeader[i]->bytePayload = getSize(mTOFDataHeader[i], mUnion[i]);
      mDRMDataHeader[i]->eventWords = mTOFDataHeader[i]->bytePayload / 4;
    }

    // check that all digits were used
    if (icurrentdigit < summary.size()) {
      LOG(ERROR) << "Not all digits are been used : only " << icurrentdigit << " of " << summary.size();
    }
  }

  for (int i = 0; i < 72; i++) {
    // adjust RDH open with the size
    mRDH[i]->memorySize = getSize(mRDH[i], mUnion[i]);
    mRDH[i]->offsetToNext = mRDH[i]->memorySize;
    mIntegratedBytes[i] += mRDH[i]->offsetToNext;

    // add RDH close
    closeRDH(i);
    mIntegratedBytes[i] += mRDH[i]->offsetToNext;
    mUnion[i] = reinterpret_cast<Union_t*>(nextPage(mRDH[i], mRDH[i]->offsetToNext));
  }

  mStartRun = false;

  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  mIntegratedTime = elapsed.count();

#ifdef VERBOSE
  int allBytes = mIntegratedAllBytes;
  for (int i = 0; i < 72; i++)
    allBytes += mIntegratedBytes[i];
  if (mVerbose && mIntegratedTime)
    std::cout << "-------- END ENCODE EVENT ------------------------------------------"
              << " " << allBytes << " words"
              << " | " << 1.e3 * mIntegratedTime << " ms"
              << " | " << 1.e-6 * allBytes / mIntegratedTime << " MB/s (average)"
              << std::endl;
#endif

  return false;
}

void Encoder::openRDH(int icrate)
{
  *mRDH[icrate] = mHBFSampler.createRDH<o2::header::RAWDataHeader>(mIR);

  // word1
  mRDH[icrate]->memorySize = mRDH[icrate]->headerSize;
  mRDH[icrate]->offsetToNext = mRDH[icrate]->memorySize;
  mRDH[icrate]->linkID = icrate % NLINKSPERCRU;
  mRDH[icrate]->packetCounter = mNRDH[icrate];
  mRDH[icrate]->cruID = icrate / NLINKSPERCRU;
  mRDH[icrate]->feeId = icrate;

  // word2
  mRDH[icrate]->triggerOrbit = mIR.orbit; // to be checked
  mRDH[icrate]->heartbeatOrbit = mIR.orbit;

  // word4
  mRDH[icrate]->triggerBC = 0; // to be checked (it should be constant)
  mRDH[icrate]->heartbeatBC = 0;
  mRDH[icrate]->triggerType = o2::trigger::HB | o2::trigger::ORBIT;
  if (mStartRun) {
    //    mRDH[icrate]->triggerType |= mIsContinuous ? o2::trigger::SOC : o2::trigger::SOT;
    mRDH[icrate]->triggerType |= o2::trigger::SOT;
  }
  if (!(mIR.orbit % 256))
    mRDH[icrate]->triggerType |= o2::trigger::TF;

  // word6
  mRDH[icrate]->pageCnt = 0;

  char* shift = reinterpret_cast<char*>(mRDH[icrate]);

  mUnion[icrate] = reinterpret_cast<Union_t*>(shift + mRDH[icrate]->headerSize);
  mNRDH[icrate]++;
}

void Encoder::addPage(int icrate)
{
  // adjust RDH open with the size
  mRDH[icrate]->memorySize = getSize(mRDH[icrate], mUnion[icrate]);
  mRDH[icrate]->offsetToNext = mRDH[icrate]->memorySize;
  mIntegratedBytes[icrate] += mRDH[icrate]->offsetToNext;

  // printf("add page (crate = %d - current size = %d)\n",icrate,mRDH[icrate]->offsetToNext);

  int pgCnt = mRDH[icrate]->pageCnt + 1;

  // move to next RDH
  mRDH[icrate] = reinterpret_cast<o2::header::RAWDataHeader*>(nextPage(mRDH[icrate], mRDH[icrate]->offsetToNext));

  openRDH(icrate);
  mRDH[icrate]->pageCnt = pgCnt;
}

void Encoder::closeRDH(int icrate)
{
  int pgCnt = mRDH[icrate]->pageCnt + 1;

  mRDH[icrate] = reinterpret_cast<o2::header::RAWDataHeader*>(nextPage(mRDH[icrate], mRDH[icrate]->offsetToNext));

  *mRDH[icrate] = mHBFSampler.createRDH<o2::header::RAWDataHeader>(mIR);

  mRDH[icrate]->stop = 0x1;

  // word1
  mRDH[icrate]->memorySize = mRDH[icrate]->headerSize;
  mRDH[icrate]->offsetToNext = mRDH[icrate]->memorySize;
  mRDH[icrate]->linkID = icrate % NLINKSPERCRU;
  mRDH[icrate]->packetCounter = mNRDH[icrate];
  mRDH[icrate]->cruID = icrate / NLINKSPERCRU;
  mRDH[icrate]->feeId = icrate;

  // word2
  mRDH[icrate]->triggerOrbit = mIR.orbit; // to be checked
  mRDH[icrate]->heartbeatOrbit = mIR.orbit;

  // word4
  mRDH[icrate]->triggerBC = 0; // to be checked (it should be constant)
  mRDH[icrate]->heartbeatBC = 0;
  mRDH[icrate]->triggerType = o2::trigger::HB | o2::trigger::ORBIT;
  if (mStartRun) {
    //    mRDH[icrate]->triggerType |= mIsContinuous ? o2::trigger::SOC : o2::trigger::SOT;
    mRDH[icrate]->triggerType |= o2::trigger::SOT;
  }
  if (!(mIR.orbit % 256))
    mRDH[icrate]->triggerType |= o2::trigger::TF;

  // word6
  mRDH[icrate]->pageCnt = pgCnt;
  mNRDH[icrate]++;
}

char* Encoder::nextPage(void* current, int step)
{
  char* point = reinterpret_cast<char*>(current);
  point += step;

  return point;
}

int Encoder::getSize(void* first, void* last)
{
  char* in = reinterpret_cast<char*>(first);
  char* out = reinterpret_cast<char*>(last);

  return int(out - in);
}

} // namespace compressed
} // namespace tof
} // namespace o2
