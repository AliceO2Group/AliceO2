// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// @brief Class to sample HBFrames for simulated interaction records + RDH utils
// @author ruben.shahoyan@cern.ch

#ifndef ALICEO2_HBFSAMPLER_H
#define ALICEO2_HBFSAMPLER_H

#include <Rtypes.h>
#include "CommonDataFormat/InteractionRecord.h"
#include "Headers/RAWDataHeader.h"
#include "CommonConstants/Triggers.h"

namespace o2
{
namespace raw
{
/*
    In the MC->Raw conversion we have to make sure that
    1) The HB and TF starts are in sync for all detectors regardless on time (bc/orbir)
    distribution of its signal.
    2) All HBF and TF (RAWDataHeaders with corresponding HB and TF trigger flags) are present
    in the emulated raw data, even if some of them had no data in particular detector.
    
    The HBFUtils class provides tools for interaction record -> HBF conversion and sampling
    of IRs for which the HBF RDH should be added to the raw data from the CRU.
    
    See testHBFUtils.cxx for the outline of generating HBF frames for simulated data.
  */

using LinkSubSpec_t = uint32_t;

//_____________________________________________________________________
class HBFUtils
{
  using IR = o2::InteractionRecord;

 public:
  static constexpr int GBTWord = 16; // length of GBT word
  static constexpr int MAXCRUPage = 512 * GBTWord;

  HBFUtils() = default;
  HBFUtils(const IR& ir0) : mFirstIR(ir0) {}
  const IR& getFirstIR() const { return mFirstIR; }
  IR& getFirstIR() { return mFirstIR; }

  void setNOrbitsPerTF(int n) { mNHBFPerTF = n > 0 ? n : 1; }
  int getNOrbitsPerTF() const { return mNHBFPerTF; }

  ///< get IR corresponding to start of the HBF
  IR getIRHBF(uint32_t hbf) const { return mFirstIR + int64_t(hbf) * o2::constants::lhc::LHCMaxBunches; }

  ///< get IR corresponding to start of the TF
  IR getIRTF(uint32_t tf) const { return getIRHBF(tf * mNHBFPerTF); }

  ///< get HBF ID corresponding to this IR
  int getHBF(const IR& rec) const;

  ///< get TF ID corresponding to this IR
  int getTF(const IR& rec) const { return getHBF(rec) / mNHBFPerTF; }

  ///< get TF and HB (within TF) for this IR
  std::pair<int, int> getTFandHBinTF(const IR& rec) const
  {
    auto hbf = getHBF(rec);
    return std::pair<int, int>(hbf / mNHBFPerTF, hbf % mNHBFPerTF);
  }

  ///< get TF and HB (abs) for this IR
  std::pair<int, int> getTFandHB(const IR& rec) const
  {
    auto hbf = getHBF(rec);
    return std::pair<int, int>(hbf / mNHBFPerTF, hbf);
  }

  ///< create RDH for given IR
  template <class H>
  H createRDH(const IR& rec) const;

  ///< update RDH for with given IR info
  template <class H>
  void updateRDH(H& rdh, const IR& rec) const;

  /*//-------------------------------------------------------------------------------------
    Fill provided vector (cleaned) by interaction records (bc/orbit) for HBFs, considering 
    BCs between interaction records "fromIR"  and "toIR" (inclusive). 
    This method provides the IRs for RDHs to add obligatory for the MC->raw conversion,
    in order to avoid missing HBFs (or even TFs)
    Typical use case: assume we are converting to RAW data the digits corresponding to triggers
    for Int.records ir[0], ir[1], ... ir[N]
    The pseudo-code:

    HBFUtils sampler;
    uint8_t packetCounter = 0;
    std::vector<o2::InteractionRecord> HBIRVec;
    auto irFrom = sampler.getFirstIR();
    for (int i=0;i<N;i++) {
      int nHBF = sampler.fillHBIRvector(HBIRVec, irFrom, ir[i]);
      irFrom = ir[i]+1;
      // nHBF-1 HBframes don't have data, we need to create empty HBFs for them
      for (int j=0;j<nHBF-1;j++) {
        auto rdh = sampler.createRDH<RAWDataHeader>( HBIRVec[j] );
        // dress rdh with cruID/FEE/Link ID
        rdh.packetCounter = packetCounter++;
        rdh.memorySize = sizeof(rdh);
        rdh.offsetToNext = sizeof(rdh);
        FLUSH_TO_SINK(&rdh, sizeof(rdh));  // open empty HBF
        rdh.stop = 0x1;
        rdh.pageCnt++;
        FLUSH_TO_SINK(&rdh, sizeof(rdh));  // close empty HBF
      }
      // write RDH for the HBF with data
      auto rdh = HBIRVec.back();
      // dress rdh with cruID/FEE/Link ID, estimate size, offset etc. and flush
      // flush raw data payload
    }
    // see tstHBFUtils for more details
    //-------------------------------------------------------------------------------------*/
  int fillHBIRvector(std::vector<IR>& dst, const IR& fromIR, const IR& toIR) const;

  void print() const;

  // some fields of the same meaning have different names in the RDH of different versions
  static uint32_t getHBOrbit(const void* rdhP);
  static uint32_t getHBBC(const void* rdhP);
  static IR getHBIR(const void* rdhP);

  static void printRDH(const void* rdhP);
  static void dumpRDH(const void* rdhP);

  static bool checkRDH(const void* rdhP, bool verbose = true);

  static uint32_t getHBOrbit(const o2::header::RAWDataHeaderV4& rdh) { return rdh.heartbeatOrbit; }
  static uint32_t getHBOrbit(const o2::header::RAWDataHeaderV5& rdh) { return rdh.orbit; }

  static uint32_t getHBBC(const o2::header::RAWDataHeaderV4& rdh) { return rdh.heartbeatBC; }
  static uint32_t getHBBC(const o2::header::RAWDataHeaderV5& rdh) { return rdh.bunchCrossing; }

  static IR getHBIR(const o2::header::RAWDataHeaderV4& rdh) { return {uint16_t(rdh.heartbeatBC), uint32_t(rdh.heartbeatOrbit)}; }
  static IR getHBIR(const o2::header::RAWDataHeaderV5& rdh) { return {uint16_t(rdh.bunchCrossing), uint32_t(rdh.orbit)}; }

  static void printRDH(const o2::header::RAWDataHeaderV5& rdh);
  static void printRDH(const o2::header::RAWDataHeaderV4& rdh);
  static void dumpRDH(const o2::header::RAWDataHeaderV5& rdh) { dumpRDH(&rdh); }
  static void dumpRDH(const o2::header::RAWDataHeaderV4& rdh) { dumpRDH(&rdh); }

  static bool checkRDH(const o2::header::RAWDataHeaderV4& rdh, bool verbose = true);
  static bool checkRDH(const o2::header::RAWDataHeaderV5& rdh, bool verbose = true);

  static LinkSubSpec_t getSubSpec(uint16_t cru, uint8_t link, uint8_t endpoint);
  static LinkSubSpec_t getSubSpec(const o2::header::RAWDataHeaderV4& rdh) { return getSubSpec(rdh.cruID, rdh.linkID, rdh.endPointID); }
  static LinkSubSpec_t getSubSpec(const o2::header::RAWDataHeaderV5& rdh) { return getSubSpec(rdh.cruID, rdh.linkID, rdh.endPointID); }

 protected:
  int mNHBFPerTF = 1 + 0xff; // number of orbits per BC
  IR mFirstIR = {0, 0};      // 1st record of the 1st TF

  ClassDefNV(HBFUtils, 1);
};

//_________________________________________________
template <>
inline void HBFUtils::updateRDH<o2::header::RAWDataHeaderV5>(o2::header::RAWDataHeaderV5& rdh, const o2::InteractionRecord& rec) const
{
  auto tfhb = getTFandHBinTF(rec);
  rdh.bunchCrossing = mFirstIR.bc;
  rdh.orbit = rec.orbit;
  //
  if (rec.bc == mFirstIR.bc) { // if we are starting new HB, set the HB trigger flag
    rdh.triggerType |= o2::trigger::ORBIT | o2::trigger::HB;
    if (tfhb.second == 0) { // if we are starting new TF, set the TF trigger flag
      rdh.triggerType |= o2::trigger::TF;
    }
  }
}

//_________________________________________________
template <>
inline void HBFUtils::updateRDH<o2::header::RAWDataHeaderV4>(o2::header::RAWDataHeaderV4& rdh, const o2::InteractionRecord& rec) const
{
  auto tfhb = getTFandHBinTF(rec);
  rdh.triggerBC = rec.bc;
  rdh.triggerOrbit = rec.orbit;

  rdh.heartbeatBC = mFirstIR.bc;
  rdh.heartbeatOrbit = rec.orbit;
  //
  if (rec.bc == mFirstIR.bc) { // if we are starting new HB, set the HB trigger flag
    rdh.triggerType |= o2::trigger::ORBIT | o2::trigger::HB;
    if (tfhb.second == 0) { // if we are starting new TF, set the TF trigger flag
      rdh.triggerType |= o2::trigger::TF;
    }
  }
}

//_________________________________________________
template <>
inline o2::header::RAWDataHeaderV4 HBFUtils::createRDH<o2::header::RAWDataHeaderV4>(const o2::InteractionRecord& rec) const
{
  o2::header::RAWDataHeaderV4 rdh;
  updateRDH(rdh, rec);
  return rdh;
}

//_________________________________________________
template <>
inline o2::header::RAWDataHeaderV5 HBFUtils::createRDH<o2::header::RAWDataHeaderV5>(const o2::InteractionRecord& rec) const
{
  o2::header::RAWDataHeaderV5 rdh;
  updateRDH(rdh, rec);
  return rdh;
}

//_____________________________________________________________________
inline LinkSubSpec_t HBFUtils::getSubSpec(uint16_t cru, uint8_t link, uint8_t endpoint)
{
  // define subspecification as in DataDistribution
  int linkValue = (LinkSubSpec_t(link) + 1) << (endpoint == 1 ? 8 : 0);
  return (LinkSubSpec_t(cru) << 16) | linkValue;
}

} // namespace raw
} // namespace o2

#endif
