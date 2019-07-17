#include "TOFReconstruction/Encoder.h"
#include <iostream>
#include <chrono>
#include <algorithm>
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
  mBuffer = new char[mSize];
  mUnion = reinterpret_cast<Union_t*>(mBuffer);
  return false;
}

int Encoder::encodeTRM(const std::vector<Digit> &summary, Int_t icrate, Int_t itrm, int &istart) // encode one TRM assuming digit vector sorted by electronic index
// return next TRM index (-1 if not in the same crate)
// start to convert digiti from istart --> then update istart to the starting position of the new TRM
{
  // printf("Encode TRM %d \n",itrm);
  unsigned char nPackedHits[256] = { 0 };
  PackedHit_t PackedHit[256][256];
  
  /** check if TRM is empty **/
  //      if (summary.nTRMSpiderHits[itrm] == 0)
  //	continue;
  
  unsigned char firstFilledFrame = 255;
  unsigned char lastFilledFrame = 0;
  
  /** loop over hits **/
  static unsigned char round=0;
  int whatTRM = summary[istart].getElTRMIndex();
  while(whatTRM == itrm){ 
    auto iframe = 0  >> 13; // 0 to be replace with hittime
    iframe = round;
    round++;
    auto phit = nPackedHits[iframe];
    PackedHit[iframe][phit].chain = summary[istart].getElChainIndex();
    PackedHit[iframe][phit].tdcID = summary[istart].getElTDCIndex();
    PackedHit[iframe][phit].channel = summary[istart].getElTDCIndex();
    PackedHit[iframe][phit].time = summary[istart].getTDC()/*0:1023 bin 24.4 ps*/; // to be checked
    PackedHit[iframe][phit].tot = summary[istart].getTOT()/*bin 48.8 ns*/; // to be checked
    nPackedHits[iframe]++;
    
    if (iframe < firstFilledFrame)
      firstFilledFrame = iframe;
    if (iframe > lastFilledFrame)
      lastFilledFrame = iframe;

    istart++;
    if(istart < summary.size())
      whatTRM = summary[istart].getElTRMIndex();
    else
      whatTRM = -1;
  }
  
  /** loop over frames **/
  for (int iframe = firstFilledFrame; iframe < lastFilledFrame + 1; iframe++) {
    
    /** check if frame is empty **/
    if (nPackedHits[iframe] == 0)
      continue;

    // frame header
    mUnion->frameHeader = { 0x0 };
    mUnion->frameHeader.mustBeZero = 0;
    mUnion->frameHeader.trmID = itrm + 3;
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
#ifdef VERBOSE
      if (mVerbose) {
	auto Chain = mUnion->packedHit.chain;
	auto TDCID = mUnion->packedHit.tdcID;
	auto Channel = mUnion->packedHit.channel;
	auto Time = mUnion->packedHit.time;
	auto TOT = mUnion->packedHit.tot;
	//	printf("hit: Chain=%d -- TDC=%d -- ch=%d -- time=%d -- tot=%d\n",Chain,TDCID,Channel,Time,TOT);

	//          std::cout << boost::format("%08x") % mUnion->data << " "
	//                    << boost::format(
	//                         "Packed hit (Chain=%d, TDCID=%d, "
	//                         "Channel=%d, Time=%d, TOT=%d)") %
	//                         Chain % TDCID % Channel % Time % TOT
	//                    << std::endl;
      }
#endif
      mUnion++;
      mIntegratedBytes += 4;
    }
    
    nPackedHits[iframe] = 0;
  }

  if(istart < summary.size()) return whatTRM;

  return -1;
}

int Encoder::encodeCrate(const std::vector<Digit> &summary, Int_t icrate, int &istart) // encode one crate assuming digit vector sorted by electronic index
// return next crate index (-1 if not)
// start to convert digiti from istart --> then update istart to the starting position of the new crate
{

  printf("Encode Crate %d \n",icrate);
  // crate header
  mUnion->crateHeader = { 0x0 };
  mUnion->crateHeader.mustBeOne = 1;
  mUnion->crateHeader.drmID = icrate;
  //    mUnion->crateHeader.eventCounter = summary.DRMGlobalTrailer.LocalEventCounter;
  //    mUnion->crateHeader.bunchID = summary.DRMStatusHeader3.l0BCID;
#ifdef VERBOSE
  if (mVerbose) {
    auto BunchID = mUnion->crateHeader.bunchID;
    auto EventCounter = mUnion->crateHeader.eventCounter;
    auto DRMID = mUnion->crateHeader.drmID;

    printf("BunchID = %d -- EventCounter = %d -- DRMID = %d\n",BunchID,EventCounter,DRMID);
//    std::cout << boost::format("%08x") % mUnion->data
//              << " "
//              << boost::format("Crate header (DRMID=%d, EventCounter=%d, BunchID=%d)") % DRMID % EventCounter % BunchID
//              << std::endl;
  }
#endif
  mUnion++;
  mIntegratedBytes += 4;

  // crate orbit
  mUnion->crateOrbit = {0x0};
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
  while(currentTRM > -1){
    currentTRM = encodeTRM(summary, icrate, currentTRM, istart);
  }

  // crate trailer
  mUnion->crateTrailer = { 0x0 };
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

  if(istart < summary.size()) return summary[istart].getElCrateIndex();

  return -1;
}
bool Encoder::encode(std::vector<Digit> summary) // pass a vector of digits in a TOF readout window
{

  if(!summary.size()) return 1; // empty array

  // caching electronic indexes in digit array
  int digitchannel;
  for ( auto dig = summary.begin(); dig != summary.end(); dig++ ) {
    digitchannel = dig->getChannel();
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

  while(currentCrate > -1){ // process also empty crates --> to be added
    currentCrate = encodeCrate(summary, currentCrate, currentEvent);
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
