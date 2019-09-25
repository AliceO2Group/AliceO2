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
#include <array>
#define VERBOSE

namespace o2
{
namespace tof
{
namespace compressed
{

bool Encoder::open(std::string name)
{
  if (mFile.is_open()) {
    std::cout << "Warning: a file was already open, closing" << std::endl;
    mFile.close();
  }
  mFile.open(name.c_str(), std::fstream::out | std::fstream::binary);
  if (!mFile.is_open()) {
    std::cerr << "Cannot open " << name << std::endl;
    return true;
  }
  return false;
}

bool Encoder::flush()
{
  mFile.write(mBuffer, mIntegratedBytes);
  return false;
}

bool Encoder::close()
{
  if (mFile.is_open())
    mFile.close();
  return false;
}

bool Encoder::alloc(long size)
{
  mSize = size;
  mBufferLocal.resize(mSize);
  mBuffer = mBufferLocal.data();
  mUnion = reinterpret_cast<Union_t*>(mBuffer);
  return false;
}

int Encoder::encodeTRM(const std::vector<Digit>& summary, Int_t icrate, Int_t itrm, int& istart) // encode one TRM assuming digit vector sorted by electronic index
// return next TRM index (-1 if not in the same crate)
// start to convert digiti from istart --> then update istart to the starting position of the new TRM
{
  // printf("Encode TRM %d \n",itrm);
  unsigned char nPackedHits[256] = {0};
  PackedHit_t PackedHit[256][256];

  /** check if TRM is empty **/
  //      if (summary.nTRMSpiderHits[itrm] == 0)
  //	continue;

  unsigned char firstFilledFrame = 255;
  unsigned char lastFilledFrame = 0;

  /** loop over hits **/
  int whatTRM = summary[istart].getElTRMIndex();
  int whatCrate = summary[istart].getElCrateIndex();
  double hittime;
  while (whatTRM == itrm && whatCrate == icrate) {
    int hittimeTDC = (summary[istart].getBC() - mEventCounter * Geo::BC_IN_WINDOW) * 1024 + summary[istart].getTDC(); // time in TDC bin within the TOF WINDOW

    auto iframe = hittimeTDC >> 13; // 0 to be replaced with hittime

    auto phit = nPackedHits[iframe];
    PackedHit[iframe][phit].chain = summary[istart].getElChainIndex();
    PackedHit[iframe][phit].tdcID = summary[istart].getElTDCIndex();
    PackedHit[iframe][phit].channel = summary[istart].getElChIndex();
    PackedHit[iframe][phit].time = hittimeTDC & 0x1FFF;
    PackedHit[iframe][phit].tot = summary[istart].getTOT() /*bin 48.8 ns*/; // to be checked
    nPackedHits[iframe]++;

    if (mVerbose) {
      auto Chain = PackedHit[iframe][phit].chain;
      auto TDCID = PackedHit[iframe][phit].tdcID;
      auto Channel = PackedHit[iframe][phit].channel;
      auto Time = PackedHit[iframe][phit].time;
      auto TOT = PackedHit[iframe][phit].tot;

      std::array<int, 4> digitInfo;
      int ext_time = iframe << 13;
      Decoder::fromRawHit2Digit(icrate, itrm, TDCID, Chain, Channel, mOrbitID, mBunchID, Time + ext_time, TOT, digitInfo);

      printf("orbit = %d\n", mOrbitID);
      printf("DigitOriginal: channel = %d -- TDC = %d -- BC = %d -- TOT = %d \n", summary[istart].getChannel(), summary[istart].getTDC(), summary[istart].getBC(), summary[istart].getTOT());
      printf("DigitAfter   :  channel = %d -- TDC = %d -- BC = %d -- TOT = %d \n", digitInfo[0], digitInfo[1], digitInfo[3], digitInfo[2]);
    }

    if (iframe < firstFilledFrame)
      firstFilledFrame = iframe;
    if (iframe > lastFilledFrame)
      lastFilledFrame = iframe;

    istart++;
    if (istart < int(summary.size())) {
      whatTRM = summary[istart].getElTRMIndex();
      whatCrate = summary[istart].getElCrateIndex();
    } else {
      whatTRM = -1;
      whatCrate = -1;
    }
  }

  /** loop over frames **/
  for (int iframe = firstFilledFrame; iframe < lastFilledFrame + 1; iframe++) {

    /** check if frame is empty **/
    if (nPackedHits[iframe] == 0)
      continue;

    // frame header
    mUnion->frameHeader = {0x0};
    mUnion->frameHeader.mustBeZero = 0;
    mUnion->frameHeader.trmID = itrm;
    mUnion->frameHeader.frameID = iframe;
    mUnion->frameHeader.numberOfHits = nPackedHits[iframe];
#ifdef VERBOSE
    if (mVerbose) {
      auto NumberOfHits = mUnion->frameHeader.numberOfHits;
      auto FrameID = mUnion->frameHeader.frameID;
      auto TRMID = mUnion->frameHeader.trmID;
      //        std::cout << boost::format("%08x") % mUnion->data
      //                  << " "
      //                  << boost::format("Frame header (TRMID=%d, FrameID=%d, NumberOfHits=%d)") % TRMID % FrameID % NumberOfHits
      //                  << std::endl;
    }
#endif
    mUnion++;
    mIntegratedBytes += 4;

    // packed hits
    for (int ihit = 0; ihit < nPackedHits[iframe]; ++ihit) {
      mUnion->packedHit = PackedHit[iframe][ihit];
      mUnion++;
      mIntegratedBytes += 4;
    }

    nPackedHits[iframe] = 0;
  }

  // if current crate is over
  if (whatCrate != icrate)
    return -1;
  // otherwise point to the next trm of the same crate
  if (istart < int(summary.size()))
    return whatTRM;

  return -1;
}

void Encoder::encodeEmptyCrate(Int_t icrate)
{
  printf("Encode Empty Crate %d \n", icrate);
  mUnion->crateHeader = {0x0};
  mUnion->crateHeader.mustBeOne = 1;
  mUnion->crateHeader.drmID = icrate;
  mUnion->crateHeader.eventCounter = mEventCounter;
  mUnion->crateHeader.bunchID = mBunchID;
  mUnion++;
  mIntegratedBytes += 4;

  // crate orbit
  mUnion->crateOrbit.orbitID = mOrbitID;
  mUnion++;
  mIntegratedBytes += 4;

  // crate trailer
  mUnion->crateTrailer = {0x0};
  mUnion->crateTrailer.mustBeOne = 1;
  mUnion++;
  mIntegratedBytes += 4;
}

int Encoder::encodeCrate(const std::vector<Digit>& summary, Int_t icrate, int& istart) // encode one crate assuming digit vector sorted by electronic index
// return next crate index (-1 if not)
// start to convert digiti from istart --> then update istart to the starting position of the new crate
{

  printf("Encode Crate %d \n", icrate);
  // crate header
  mUnion->crateHeader = {0x0};
  mUnion->crateHeader.mustBeOne = 1;
  mUnion->crateHeader.drmID = icrate;
  mUnion->crateHeader.eventCounter = mEventCounter;
  mUnion->crateHeader.bunchID = mBunchID;
#ifdef VERBOSE
  if (mVerbose) {
    auto BunchID = mUnion->crateHeader.bunchID;
    auto EventCounter = mUnion->crateHeader.eventCounter;
    auto DRMID = mUnion->crateHeader.drmID;

    printf("BunchID = %d -- EventCounter = %d -- DRMID = %d\n", BunchID, EventCounter, DRMID);
    //    std::cout << boost::format("%08x") % mUnion->data
    //              << " "
    //              << boost::format("Crate header (DRMID=%d, EventCounter=%d, BunchID=%d)") % DRMID % EventCounter % BunchID
    //              << std::endl;
  }
#endif
  mUnion++;
  mIntegratedBytes += 4;

  // crate orbit
  mUnion->crateOrbit.orbitID = mOrbitID;
  //  mUnion->crateOrbit.orbitID = 0;
#ifdef VERBOSE
  if (mVerbose) {
    auto OrbitID = mUnion->crateOrbit.orbitID;
    //    std::cout << boost::format("%08x") % mUnion->data
    //	      << " "
    //	      << boost::format("Crate orbit (OrbitID=%d)") % BunchID
    //	      << std::endl;
  }
#endif
  mUnion++;
  mIntegratedBytes += 4;

  /** loop over TRMs **/
  Int_t currentTRM = summary[istart].getElTRMIndex();
  while (currentTRM > -1) {
    currentTRM = encodeTRM(summary, icrate, currentTRM, istart);
  }

  // crate trailer
  mUnion->crateTrailer = {0x0};
  mUnion->crateTrailer.mustBeOne = 1;
#ifdef VERBOSE
  if (mVerbose) {
    //    std::cout << boost::format("%08x") % mUnion->data
    //              << " "
    //              << "Crate trailer"
    //              << std::endl;
  }
#endif
  mUnion++;
  mIntegratedBytes += 4;

  if (istart < int(summary.size()))
    return summary[istart].getElCrateIndex();

  return -1;
}

bool Encoder::encode(std::vector<Digit> summary, int tofwindow) // pass a vector of digits in a TOF readout window, tof window is the entry in the vector-of-vector of digits needed to extract bunch id and orbit
{

  mEventCounter = tofwindow;                                                                       // tof window index
  mOrbitID = mEventCounter / Geo::NWINDOW_IN_ORBIT;                                                // since 3 tof window = 1 orbit
  mBunchID = ((mEventCounter % Geo::NWINDOW_IN_ORBIT) * Geo::BC_IN_ORBIT) / Geo::NWINDOW_IN_ORBIT; // bunch crossing in the current orbit at the beginning of the window

  if (!summary.size()) {
    for (int iemptycrate = 0; iemptycrate < 72; iemptycrate++) {
      encodeEmptyCrate(iemptycrate);
    }
    return 1; // empty array
  }

  // caching electronic indexes in digit array
  for (auto dig = summary.begin(); dig != summary.end(); dig++) {
    int digitchannel = dig->getChannel();
    dig->setElectronicIndex(Geo::getECHFromCH(digitchannel));
  }

  // sorting by electroni indexes
  std::sort(summary.begin(), summary.end(),
            [](Digit a, Digit b) { return a.getElectronicIndex() < b.getElectronicIndex(); });

#ifdef VERBOSE
  if (mVerbose)
    std::cout << "-------- START ENCODE EVENT ----------------------------------------" << std::endl;
#endif
  auto start = std::chrono::high_resolution_clock::now();

  int currentEvent = 0;
  int currentCrate = summary[0].getElCrateIndex();

  for (int iemptycrate = 0; iemptycrate < currentCrate; iemptycrate++) {
    encodeEmptyCrate(iemptycrate);
  }

  while (currentCrate > -1) { // process also empty crates
    int prevCrate = currentCrate;
    currentCrate = encodeCrate(summary, currentCrate, currentEvent);

    if (currentCrate == -1) {
      for (int iemptycrate = prevCrate + 1; iemptycrate < 72; iemptycrate++) {
        encodeEmptyCrate(iemptycrate);
      }
    } else {
      for (int iemptycrate = prevCrate + 1; iemptycrate < currentCrate; iemptycrate++) {
        encodeEmptyCrate(iemptycrate);
      }
    }
  }

  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  mIntegratedTime = elapsed.count();

#ifdef VERBOSE
  if (mVerbose && mIntegratedTime)
    std::cout << "-------- END ENCODE EVENT ------------------------------------------"
              << " " << mIntegratedBytes << " words"
              << " | " << 1.e3 * mIntegratedTime << " ms"
              << " | " << 1.e-6 * mIntegratedBytes / mIntegratedTime << " MB/s (average)"
              << std::endl;
#endif

  return false;
}
} // namespace compressed
} // namespace tof
} // namespace o2
