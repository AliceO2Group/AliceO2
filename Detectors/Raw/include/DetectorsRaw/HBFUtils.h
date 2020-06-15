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

#ifndef ALICEO2_HBFUTILS_H
#define ALICEO2_HBFUTILS_H

#include <Rtypes.h>
#include "DetectorsRaw/RDHUtils.h"
#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include "CommonDataFormat/InteractionRecord.h"
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

//_____________________________________________________________________
struct HBFUtils : public o2::conf::ConfigurableParamHelper<HBFUtils> {
  using IR = o2::InteractionRecord;


  IR getFirstIR() const { return {bcFirst, orbitFirst}; }

  int getNOrbitsPerTF() const { return nHBFPerTF; }

  ///< get IR corresponding to start of the HBF
  IR getIRHBF(uint32_t hbf) const { return getFirstIR() + int64_t(hbf) * o2::constants::lhc::LHCMaxBunches; }

  ///< get IR corresponding to start of the TF
  IR getIRTF(uint32_t tf) const { return getIRHBF(tf * nHBFPerTF); }

  ///< get HBF ID corresponding to this IR
  int64_t getHBF(const IR& rec) const;

  ///< get TF ID corresponding to this IR
  int64_t getTF(const IR& rec) const { return getHBF(rec) / nHBFPerTF; }

  ///< get TF and HB (within TF) for this IR
  std::pair<int, int> getTFandHBinTF(const IR& rec) const
  {
    auto hbf = getHBF(rec);
    return std::pair<int, int>(hbf / nHBFPerTF, hbf % nHBFPerTF);
  }

  ///< get TF and HB (abs) for this IR
  std::pair<int, int> getTFandHB(const IR& rec) const
  {
    auto hbf = getHBF(rec);
    return std::pair<int, int>(hbf / nHBFPerTF, hbf);
  }

  ///< create RDH for given IR
  template <typename H>
  H createRDH(const IR& rec) const;

  ///< update RDH for with given IR info
  template <typename H>
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
        RDHUtils::setPacketCounter(rdh, packetCounter++);
        RDHUtils::setMemorySize(rdh, sizeof(rdh));
        RDHUtils::setOffsetToNext(rdh, sizeof(rdh));
        FLUSH_TO_SINK(&rdh, sizeof(rdh));  // open empty HBF
        RDHUtils::setStop(rdh, 0x1);
        RDHUtils::setPageCounter(rdh, RDHUtils::getPageCounter()+1 );
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

  void print() const { printKeyValues(true); }

  int nHBFPerTF = 1 + 0xff; // number of orbits per BC
  uint16_t bcFirst = 0;     ///< BC of 1st TF
  uint32_t orbitFirst = 0;  ///< orbit of 1st TF

  O2ParamDef(HBFUtils, "HBFUtils");
};

//_________________________________________________
template <typename H>
void HBFUtils::updateRDH(H& rdh, const IR& rec) const
{
  auto tfhb = getTFandHBinTF(rec);
  RDHUtils::setHeartBeatBC(rdh, bcFirst);
  RDHUtils::setHeartBeatOrbit(rdh, rec.orbit);
  if (RDHUtils::getVersion(rdh) < 5) { // v3,4 have separate fields for trigger IR
    RDHUtils::setTriggerBC(rdh, bcFirst);
    RDHUtils::setTriggerOrbit(rdh, rec.orbit);
  }

  if (rec.bc == bcFirst) { // if we are starting new HB, set the HB trigger flag
    auto trg = RDHUtils::getTriggerType(rdh) | (o2::trigger::ORBIT | o2::trigger::HB);
    if (tfhb.second == 0) { // if we are starting new TF, set the TF trigger flag
      trg |= o2::trigger::TF;
    }
    RDHUtils::setTriggerType(rdh, trg);
  }
}

//_________________________________________________
template <typename H>
inline H HBFUtils::createRDH(const o2::InteractionRecord& rec) const
{
  H rdh;
  updateRDH(rdh, rec);
  return rdh;
}

} // namespace raw
} // namespace o2

#endif
