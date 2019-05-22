// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Encoder.h"
#include <iostream>
#include <chrono>

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
  mBuffer = new char[mSize];
  mUnion = reinterpret_cast<Union_t*>(mBuffer);
  return false;
}

bool Encoder::encode(/*define input structure*/)
{

#ifdef VERBOSE
  if (mVerbose)
    std::cout << "-------- START ENCODE EVENT ----------------------------------------" << std::endl;
#endif
  auto start = std::chrono::high_resolution_clock::now();

  unsigned int nWords = 0;

  // crate header
  mUnion->CrateHeader = { 0x0 };
  mUnion->CrateHeader.mustBeOne = 1;
  //    mUnion->CrateHeader.drmID = summary.DRMGlobalHeader.drmID;
  //    mUnion->CrateHeader.eventCounter = summary.DRMGlobalTrailer.LocalEventCounter;
  //    mUnion->CrateHeader.bunchID = summary.DRMStatusHeader3.l0BCID;
#ifdef VERBOSE
  if (mVerbose) {
    auto BunchID = mUnion->CrateHeader.bunchID;
    auto EventCounter = mUnion->CrateHeader.eventCounter;
    auto DRMID = mUnion->CrateHeader.drmID;
    //    std::cout << boost::format("%08x") % mUnion->Data
    //              << " "
    //              << boost::format("Crate header (DRMID=%d, EventCounter=%d, BunchID=%d)") % DRMID % EventCounter % BunchID
    //              << std::endl;
  }
#endif
  mUnion++;
  nWords++;

  /** loop over TRMs **/

  unsigned char nPackedHits[256] = { 0 };
  PackedHit_t PackedHit[256][256];
  for (int itrm = 0; itrm < 10; itrm++) {

    /** check if TRM is empty **/
    //      if (summary.nTRMSpiderHits[itrm] == 0)
    //	continue;

    unsigned char firstFilledFrame = 255;
    unsigned char lastFilledFrame = 0;

    /** loop over hits **/
    /*
      for (int ihit = 0; ihit < summary.nTRMSpiderHits[itrm]; ++ihit) {
	
	auto hit = summary.TRMSpiderHit[itrm][ihit];
	auto iframe = hit.HitTime >> 13;
	auto phit = nPackedHits[iframe];
	PackedHit[iframe][phit].Chain = hit.Chain;
	PackedHit[iframe][phit].TDCID = hit.TDCID;
	PackedHit[iframe][phit].Channel = hit.Chan;
	PackedHit[iframe][phit].Time = hit.HitTime;
	PackedHit[iframe][phit].TOT = hit.TOTWidth;
	nPackedHits[iframe]++;
	
	if (iframe < firstFilledFrame)
	  firstFilledFrame = iframe;
	if (iframe > lastFilledFrame)
	  lastFilledFrame = iframe;
      }
      */

    /** loop over frames **/
    for (int iframe = firstFilledFrame; iframe < lastFilledFrame + 1;
         iframe++) {

      /** check if frame is empty **/
      if (nPackedHits[iframe] == 0)
        continue;

      // frame header
      mUnion->FrameHeader = { 0x0 };
      mUnion->FrameHeader.mustBeZero = 0;
      mUnion->FrameHeader.trmID = itrm + 3;
      mUnion->FrameHeader.frameID = iframe;
      mUnion->FrameHeader.numberOfHits = nPackedHits[iframe];
#ifdef VERBOSE
      if (mVerbose) {
        auto NumberOfHits = mUnion->FrameHeader.numberOfHits;
        auto FrameID = mUnion->FrameHeader.frameID;
        auto TRMID = mUnion->FrameHeader.trmID;
        //        std::cout << boost::format("%08x") % mUnion->Data
        //                  << " "
        //                  << boost::format("Frame header (TRMID=%d, FrameID=%d, NumberOfHits=%d)") % TRMID % FrameID % NumberOfHits
        //                  << std::endl;
      }
#endif
      mUnion++;
      nWords++;

      // packed hits
      for (int ihit = 0; ihit < nPackedHits[iframe]; ++ihit) {
        mUnion->PackedHit = PackedHit[iframe][ihit];
#ifdef VERBOSE
        if (mVerbose) {
          auto Chain = mUnion->PackedHit.chain;
          auto TDCID = mUnion->PackedHit.tdcID;
          auto Channel = mUnion->PackedHit.channel;
          auto Time = mUnion->PackedHit.time;
          auto TOT = mUnion->PackedHit.tot;
          //          std::cout << boost::format("%08x") % mUnion->Data << " "
          //                    << boost::format(
          //                         "Packed hit (Chain=%d, TDCID=%d, "
          //                         "Channel=%d, Time=%d, TOT=%d)") %
          //                         Chain % TDCID % Channel % Time % TOT
          //                    << std::endl;
        }
#endif
        mUnion++;
        nWords++;
      }

      nPackedHits[iframe] = 0;
    }
  }

  // crate trailer
  mUnion->CrateTrailer = { 0x0 };
  mUnion->CrateTrailer.mustBeOne = 1;
#ifdef VERBOSE
  if (mVerbose) {
    //    std::cout << boost::format("%08x") % mUnion->Data
    //              << " "
    //              << "Crate trailer"
    //              << std::endl;
  }
#endif
  mUnion++;
  nWords++;

  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;

  mIntegratedBytes += nWords * 4;
  mIntegratedTime += elapsed.count();

#ifdef VERBOSE
  if (mVerbose)
    std::cout << "-------- END ENCODE EVENT ------------------------------------------"
              << " " << nWords << " words"
              << " | " << 1.e3 * elapsed.count() << " ms"
              << " | " << 1.e-6 * mIntegratedBytes / mIntegratedTime << " MB/s (average)"
              << std::endl;
#endif

  return false;
}
} // namespace compressed
} // namespace tof
} // namespace o2
