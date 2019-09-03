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
#define VERBOSE

namespace o2
{
namespace tof
{
namespace compressed
{

bool Decoder::open(std::string name)
{
  if (mFile.is_open()) {
    std::cout << "Warning: a file was already open, closing" << std::endl;
    mFile.close();
  }
  mFile.open(name.c_str(), std::fstream::in | std::fstream::binary);
  if (!mFile.is_open()) {
    std::cerr << "Cannot open " << name << std::endl;
    return true;
  }

  mFile.seekg(0, mFile.end);
  mSize = mFile.tellg();
  mFile.seekg(0);

  mBufferLocal.resize(mSize);
  mBuffer = mBufferLocal.data();

  // read content of infile
  mFile.read(mBuffer, mSize);
  mUnion = reinterpret_cast<Union_t*>(mBuffer);
  mUnionEnd = reinterpret_cast<Union_t*>(mBuffer + mSize - 1);
  close();

  return false;
}

bool Decoder::close()
{
  if (mFile.is_open())
    mFile.close();
  return false;
}

void Decoder::readTRM(std::vector<Digit>* digits, int iddl, int orbit, int bunchid)
{
  if (mVerbose)
    printTRMInfo();
  int nhits = mUnion->frameHeader.numberOfHits;
  int time_ext = mUnion->frameHeader.frameID << 13;
  int itrm = mUnion->frameHeader.trmID;
  mUnion++;

  // read hits
  Int_t channel, echannel;
  Int_t tdc;
  Int_t tot;
  Int_t bc;
  Int_t time;

  std::array<int, 4> digitInfo;

  for (int i = 0; i < nhits; i++) {
    fromRawHit2Digit(iddl, itrm, mUnion->packedHit.tdcID, mUnion->packedHit.chain, mUnion->packedHit.channel, orbit, bunchid, time_ext + mUnion->packedHit.time, mUnion->packedHit.tot, digitInfo);

    if (mVerbose)
      printHitInfo();
    digits->emplace_back(digitInfo[0], digitInfo[1], digitInfo[2], digitInfo[3]);
    mUnion++;
  }
}

void Decoder::fromRawHit2Digit(int iddl, int itrm, int itdc, int ichain, int channel, int orbit, int bunchid, int tdc, int tot, std::array<int, 4>& digitInfo)
{
  // convert raw info in digit info (channel, tdc, tot, bc)
  // tdc = packetHit.time + (frameHeader.frameID << 13)
  int echannel = Geo::getECHFromIndexes(iddl, itrm, ichain, itdc, channel);
  digitInfo[0] = Geo::getCHFromECH(echannel);
  digitInfo[2] = tot;

  digitInfo[3] = int(orbit * o2::tof::Geo::BC_IN_ORBIT);
  digitInfo[3] += bunchid;
  digitInfo[3] += tdc / 1024;
  digitInfo[1] = tdc % 1024;
}

bool Decoder::decode(std::vector<Digit>* digits) // return a vector of digits in a TOF readout window
{
#ifdef VERBOSE
  if (mVerbose)
    std::cout << "-------- START DECODE EVENT ----------------------------------------" << std::endl;
#endif
  auto start = std::chrono::high_resolution_clock::now();

  if (mUnion > mUnionEnd)
    return 1;

  // .. decoding

  // int eventcounter;
  for (int icrate = 0; icrate < 72; icrate++) {
    // read Crate Header
    //eventcounter = mUnion->crateHeader.eventCounter;
    int bunchid = mUnion->crateHeader.bunchID;
    if (mVerbose)
      printCrateInfo();
    mUnion++;

    //read Orbit
    int orbit = mUnion->crateOrbit.orbitID;
    if (mVerbose)
      printf("orbit ID      = %d\n", orbit);
    mUnion++;

    while (!mUnion->frameHeader.mustBeZero) {
      readTRM(digits, icrate, orbit, bunchid);
    }

    // read Crate Tralier
    if (mVerbose)
      printCrateTrailerInfo();
    mUnion++;
  }

  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  mIntegratedTime = elapsed.count();

#ifdef VERBOSE
  if (mVerbose && mIntegratedTime)
    std::cout << "-------- END DECODE EVENT ------------------------------------------"
              << " " << mIntegratedBytes << " words"
              << " | " << 1.e3 * mIntegratedTime << " ms"
              << " | " << 1.e-6 * mIntegratedBytes / mIntegratedTime << " MB/s (average)"
              << std::endl;
#endif

  return 0;
}

void Decoder::printCrateInfo() const
{
  printf("___CRATE HEADER____\n");
  printf("DRM ID        = %d\n", mUnion->crateHeader.drmID);
  printf("Bunch ID      = %d\n", mUnion->crateHeader.bunchID);
  printf("Event Counter = %d\n", mUnion->crateHeader.eventCounter);
  printf("Must be ONE   = %d\n", mUnion->crateHeader.mustBeOne);
  printf("___________________\n");
}

void Decoder::printCrateTrailerInfo() const
{
  printf("___CRATE TRAILER___\n");
  printf("TRM fault 03  = %d\n", mUnion->crateTrailer.trmFault03);
  printf("TRM fault 04  = %d\n", mUnion->crateTrailer.trmFault04);
  printf("TRM fault 05  = %d\n", mUnion->crateTrailer.trmFault05);
  printf("TRM fault 06  = %d\n", mUnion->crateTrailer.trmFault06);
  printf("TRM fault 07  = %d\n", mUnion->crateTrailer.trmFault07);
  printf("TRM fault 08  = %d\n", mUnion->crateTrailer.trmFault08);
  printf("TRM fault 09  = %d\n", mUnion->crateTrailer.trmFault09);
  printf("TRM fault 10  = %d\n", mUnion->crateTrailer.trmFault10);
  printf("TRM fault 11  = %d\n", mUnion->crateTrailer.trmFault11);
  printf("TRM fault 12  = %d\n", mUnion->crateTrailer.trmFault12);
  printf("crate fault   = %d\n", mUnion->crateTrailer.crateFault);
  printf("Must be ONE   = %d\n", mUnion->crateTrailer.mustBeOne);
  printf("___________________\n");
}

void Decoder::printTRMInfo() const
{
  printf("______TRM_INFO_____\n");
  printf("TRM ID        = %d\n", mUnion->frameHeader.trmID);
  printf("Frame ID      = %d\n", mUnion->frameHeader.frameID);
  printf("N. hits       = %d\n", mUnion->frameHeader.numberOfHits);
  printf("DeltaBC       = %d\n", mUnion->frameHeader.deltaBC);
  printf("Must be Zero  = %d\n", mUnion->frameHeader.mustBeZero);
  printf("___________________\n");
}

void Decoder::printHitInfo() const
{
  printf("______HIT_INFO_____\n");
  printf("TDC ID        = %d\n", mUnion->packedHit.tdcID);
  printf("CHAIN ID      = %d\n", mUnion->packedHit.chain);
  printf("CHANNEL ID    = %d\n", mUnion->packedHit.channel);
  printf("TIME          = %d\n", mUnion->packedHit.time);
  printf("TOT           = %d\n", mUnion->packedHit.tot);
  printf("___________________\n");
}
} // namespace compressed
} // namespace tof
} // namespace o2
