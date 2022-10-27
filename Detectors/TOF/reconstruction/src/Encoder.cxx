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

#include "TOFReconstruction/Encoder.h"
#include "TOFReconstruction/Decoder.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include "CommonConstants/LHCConstants.h"
#include "CommonConstants/Triggers.h"
#include "TString.h"
#include <fairlogger/Logger.h>
#include "DataFormatsParameters/GRPObject.h"
#include "CommonUtils/NameConf.h"
#include "DetectorsRaw/RDHUtils.h"

#include <array>
#define VERBOSE

namespace o2
{
namespace tof
{
namespace raw
{
using RDHUtils = o2::raw::RDHUtils;

Encoder::Encoder()
{
  // intialize 72 buffers (one per crate)
  for (int i = 0; i < 72; i++) {
    mBuffer[i] = nullptr;
    mUnion[i] = nullptr;
    mStart[i] = nullptr;
    mCrateOn[i] = true;
  }
}

void Encoder::nextWord(int icrate)
{
  if (mNextWordStatus[icrate]) {
    mUnion[icrate]++;
    mUnion[icrate]->data = 0;
    mUnion[icrate]++;
    mUnion[icrate]->data = 0;
  }
  mUnion[icrate]++;
  //nextWordNoEmpty(icrate);
  mNextWordStatus[icrate] = !mNextWordStatus[icrate];
}

bool Encoder::open(const std::string& name, const std::string& path, const std::string& fileFor)
{
  bool status = false;

  // register links
  o2::header::RAWDataHeader rdh;
  mFileWriter.useRDHVersion(RDHUtils::getVersion<o2::header::RAWDataHeader>());
  for (int crateid = 0; crateid < 72; crateid++) {
    // cru=0 --> FLP 1, ... defined in Geo
    // cru=1 --> FLP 1, ... defined in Geo
    // cru=2 --> FLP 0, ... defined in Geo
    // cru=3 --> FLP 0, ... defined in Geo
    RDHUtils::setFEEID(rdh, Geo::getFEEid(crateid));
    RDHUtils::setCRUID(rdh, Geo::getCRUid(crateid));
    RDHUtils::setLinkID(rdh, Geo::getCRUlink(crateid));
    RDHUtils::setEndPointID(rdh, Geo::getCRUendpoint(crateid));
    const int served = 3;
    const int received = 3;
    uint32_t detField = (served << 24) + (received << 16);
    RDHUtils::setDetectorField(rdh, detField);
    // currently storing each CRU in a separate file
    std::string outFileLink;
    if (mCrateOn[crateid]) {
      if (fileFor == "all") { // single file for all links
        outFileLink = o2::utils::Str::concat_string(path, "/TOF.raw");
      } else if (fileFor == "cruendpoint") {
        outFileLink = o2::utils::Str::concat_string(path, "/", "TOF_alio2-cr1-flp", std::to_string(Geo::getFLPid(crateid)), "_cru", std::to_string(Geo::getCRUid(crateid)), "_", std::to_string(Geo::getCRUendpoint(crateid)), ".raw");
      } else if (fileFor == "link") {
        outFileLink = o2::utils::Str::concat_string(path, "/", "TOF_alio2-cr1-flp", std::to_string(Geo::getFLPid(crateid)), "_cru", std::to_string(Geo::getCRUid(crateid)), "_", std::to_string(Geo::getCRUendpoint(crateid)), "_link", std::to_string(RDHUtils::getLinkID(rdh)), ".raw");
      } else {
        throw std::runtime_error("invalid option provided for file grouping");
      }
      mFileWriter.registerLink(rdh, outFileLink);
    }
  }

  std::string inputGRP = o2::base::NameConf::getGRPFileName();
  const auto grp = o2::parameters::GRPObject::loadFrom(inputGRP);
  mFileWriter.setContinuousReadout(grp->isDetContinuousReadOut(o2::detectors::DetID::TOF)); // must be set explicitly

  return status;
}

bool Encoder::flush(int icrate)
{
  int nbyte = getSize(mStart[icrate], mUnion[icrate]);
  if (nbyte) {
    if (mCrateOn[icrate]) {
      //printf("flush crate %d -- byte = %d -- orbit = %d, bc = %d\n",icrate, nbyte, mIR.orbit, mIR.bc);
      mFileWriter.addData(Geo::getFEEid(icrate), Geo::getCRUid(icrate), Geo::getCRUlink(icrate), Geo::getCRUendpoint(icrate), mIR, gsl::span(mBuffer[icrate], nbyte));
    }
    mIntegratedAllBytes += nbyte;
  }
  mUnion[icrate] = mStart[icrate];
  return false;
}

bool Encoder::close()
{
  mFileWriter.close();
  return false;
}

bool Encoder::alloc(long size)
{
  if (size < 500000) {
    size = 500000;
  }

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

  static unsigned long bc_shift = uint64_t(o2::raw::HBFUtils::Instance().orbitFirstSampled) * Geo::BC_IN_ORBIT;

  if (mVerbose) {
    printf("Crate %d: encode TRM %d \n", icrate, itrm);
  }

  // TRM HEADER
  Union_t* trmheader = mUnion[icrate];
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
      if (whatChain != ichain) {
        break;
      }
      int whatTRM = summary[istart].getElTRMIndex();
      if (whatTRM != itrm) {
        break;
      }
      int whatCrate = summary[istart].getElCrateIndex();
      if (whatCrate != icrate) {
        break;
      }

      int hittimeTDC = (summary[istart].getBC() - bc_shift - mEventCounter * Geo::BC_IN_WINDOW) * 1024 + summary[istart].getTDC(); // time in TDC bin within the TOF WINDOW

      if (hittimeTDC < 0) {
        LOG(error) << "Negative hit encoded " << hittimeTDC << ", something went wrong in filling readout window";
        printf("%llu %d %d\n", (unsigned long long)summary[istart].getBC() - bc_shift, mEventCounter * Geo::BC_IN_WINDOW, summary[istart].getTDC());
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

  // set TRM data size
  int neventwords = getSize(trmheader, mUnion[icrate]) / 4 + 1;
  neventwords -= neventwords / 4 * 2;
  trmheader->trmDataHeader.eventWords = neventwords;

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
  if (mVerbose) {
    std::cout << "-------- START ENCODE EVENT ----------------------------------------" << std::endl;
  }
#endif
  auto start = std::chrono::high_resolution_clock::now();

  mEventCounter = tofwindow; // tof window index
  mIR.orbit = mEventCounter / Geo::NWINDOW_IN_ORBIT + o2::raw::HBFUtils::Instance().getFirstSampledTFIR().orbit;

  for (int i = 0; i < 72; i++) {
    mNextWordStatus[i] = false;
  }

  int bcFirstWin;

  // encode all windows
  for (int iwin = 0; iwin < Geo::NWINDOW_IN_ORBIT; iwin++) {
    mEventCounter = tofwindow + iwin; // tof window index

    std::vector<o2::tof::Digit>& summary = digitWindow.at(iwin);

    mIR.bc = ((mEventCounter % Geo::NWINDOW_IN_ORBIT) * Geo::BC_IN_ORBIT) / Geo::NWINDOW_IN_ORBIT + mFirstBC; // bunch crossing in the current orbit at the beginning of the window.

    if (iwin == 0) {
      bcFirstWin = mIR.bc;
    }

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
      mUnion[i]->drmHeadW1.drmhVersion = 0x12;
      mUnion[i]->drmHeadW1.drmHSize = 5;
      mUnion[i]->drmHeadW1.mbza = 0;
      mUnion[i]->drmHeadW1.mbzb = 0;
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
      int neventwords = getSize(mDRMDataHeader[i], mUnion[i]) / 4 + 1;
      neventwords -= neventwords / 4 * 2 + 6;
      mDRMDataHeader[i]->eventWords = neventwords;
      nextWord(i);
      mUnion[i]->data = 0x70000000;
      nextWord(i);

      mTOFDataHeader[i]->bytePayload = getSize(mTOFDataHeader[i], mUnion[i]);
    }

    // check that all digits were used
    if (icurrentdigit < summary.size()) {
      LOG(error) << "Not all digits are been used : only " << icurrentdigit << " of " << summary.size();
    }
  }

  mIR.bc = bcFirstWin;

  for (int i = 0; i < 72; i++) {
    flush(i);
  }

  mStartRun = false;

  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  mIntegratedTime = elapsed.count();

  return false;
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
