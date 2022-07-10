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

/// \file   CTFCoder.h
/// \author ruben.shahoyan@cern.ch
/// \brief class for entropy encoding/decoding of TRD data

#ifndef O2_TRD_CTFCODER_H
#define O2_TRD_CTFCODER_H

#include <algorithm>
#include <iterator>
#include <string>
#include <array>
#include "DataFormatsTRD/CTF.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsBase/CTFCoderBase.h"
#include "rANS/rans.h"
#include "TRDReconstruction/CTFHelper.h"
#include "CommonConstants/LHCConstants.h"

class TTree;

namespace o2
{
namespace trd
{

class CTFCoder : public o2::ctf::CTFCoderBase
{
 public:
  CTFCoder(o2::ctf::CTFCoderBase::OpType op) : o2::ctf::CTFCoderBase(op, CTF::getNBlocks(), o2::detectors::DetID::TRD) {}
  ~CTFCoder() final = default;

  /// entropy-encode data to buffer with CTF
  template <typename VEC>
  o2::ctf::CTFIOSize encode(VEC& buff, const gsl::span<const TriggerRecord>& trigData, const gsl::span<const Tracklet64>& trkData, const gsl::span<const Digit>& digData);

  /// entropy decode data from buffer with CTF
  template <typename VTRG, typename VTRK, typename VDIG>
  o2::ctf::CTFIOSize decode(const CTF::base& ec, VTRG& trigVec, VTRK& trkVec, VDIG& digVec);

  void createCoders(const std::vector<char>& bufVec, o2::ctf::CTFCoderBase::OpType op) final;

  void setBCShift(int n) { mBCShift = 0; }
  void setFirstTFOrbit(uint32_t n) { mFirstTFOrbit = n; }
  void setCheckBogusTrig(int v) { mCheckBogusTrig = v; }

 private:
  void appendToTree(TTree& tree, CTF& ec);
  void readFromTree(TTree& tree, int entry, std::vector<TriggerRecord>& trigVec, std::vector<Tracklet64>& trkVec, std::vector<Digit>& digVec);
  int mBCShift = 0; // shift to apply to decoded IR (i.e. CTP offset if was not corrected on raw data decoding level)
  uint32_t mFirstTFOrbit = 0;
  int mCheckBogusTrig = 1;
};

/// entropy-encode digits and tracklets to buffer with CTF
template <typename VEC>
o2::ctf::CTFIOSize CTFCoder::encode(VEC& buff, const gsl::span<const TriggerRecord>& trigData, const gsl::span<const Tracklet64>& trkData, const gsl::span<const Digit>& digData)
{
  using MD = o2::ctf::Metadata::OptStore;
  // what to do which each field: see o2::ctd::Metadata explanation
  constexpr MD optField[CTF::getNBlocks()] = {
    MD::EENCODE, // BLC_bcIncTrig
    MD::EENCODE, // BLC_orbitIncTrig
    MD::EENCODE, // BLC_entriesTrk
    MD::EENCODE, // BLC_entriesDig
    MD::EENCODE, // BLC_HCIDTrk
    MD::EENCODE, // BLC_padrowTrk
    MD::EENCODE, // BLC_colTrk
    MD::EENCODE, // BLC_posTrk
    MD::EENCODE, // BLC_slopeTrk
    MD::EENCODE, // BLC_pidTrk
    MD::EENCODE, // BLC_CIDDig
    MD::EENCODE, // BLC_ROBDig
    MD::EENCODE, // BLC_MCMDig
    MD::EENCODE, // BLC_chanDig
    MD::EENCODE, // BLC_ADCDig
  };
  static size_t bogusWarnMsg = 0;
  if (mCheckBogusTrig && bogusWarnMsg < mCheckBogusTrig) {
    uint32_t orbitPrev = mFirstTFOrbit;
    uint16_t bcPrev = 0;
    int cnt = 0;
    for (const auto& trig : trigData) {
      LOGP(debug, "Trig#{} Old: {}/{} New: {}/{}", cnt++, bcPrev, orbitPrev, trig.getBCData().bc, trig.getBCData().orbit);
      auto orbitPrevT = orbitPrev;
      auto bcPrevT = bcPrev;
      bcPrev = trig.getBCData().bc;
      orbitPrev = trig.getBCData().orbit;
      if (trig.getBCData().orbit < orbitPrevT || trig.getBCData().bc >= o2::constants::lhc::LHCMaxBunches || (trig.getBCData().orbit == orbitPrevT && trig.getBCData().bc < bcPrevT)) {
        LOGP(alarm, "Bogus TRD trigger at bc:{}/orbit:{} (previous was {}/{}), with {} tracklets and {} digits",
             trig.getBCData().bc, trig.getBCData().orbit, bcPrevT, orbitPrevT, trig.getNumberOfTracklets(), trig.getNumberOfDigits());
        if (++bogusWarnMsg >= mCheckBogusTrig) {
          LOGP(alarm, "Max amount of warnings ({}) was issued, will not warn anymore", size_t(mCheckBogusTrig));
          break;
        }
      }
    }
  }

  CTFHelper helper(trigData, trkData, digData);
  // book output size with some margin
  auto szIni = sizeof(CTFHeader) + helper.getSize() * 2. / 3; // will be autoexpanded if needed
  buff.resize(szIni);

  auto ec = CTF::create(buff);
  using ECB = CTF::base;

  ec->setHeader(helper.createHeader());
  assignDictVersion(static_cast<o2::ctf::CTFDictHeader&>(ec->getHeader()));
  ec->getANSHeader().majorVersion = 0;
  ec->getANSHeader().minorVersion = 1;
  // at every encoding the buffer might be autoexpanded, so we don't work with fixed pointer ec
  o2::ctf::CTFIOSize iosize;
#define ENCODETRD(beg, end, slot, bits) CTF::get(buff.data())->encode(beg, end, int(slot), bits, optField[int(slot)], &buff, mCoders[int(slot)].get(), getMemMarginFactor());
  // clang-format off
  iosize += ENCODETRD(helper.begin_bcIncTrig(),    helper.end_bcIncTrig(),     CTF::BLC_bcIncTrig,    0);
  iosize += ENCODETRD(helper.begin_orbitIncTrig(), helper.end_orbitIncTrig(),  CTF::BLC_orbitIncTrig, 0);
  iosize += ENCODETRD(helper.begin_entriesTrk(),   helper.end_entriesTrk(),    CTF::BLC_entriesTrk,   0);
  iosize += ENCODETRD(helper.begin_entriesDig(),   helper.end_entriesDig(),    CTF::BLC_entriesDig,   0);

  iosize += ENCODETRD(helper.begin_HCIDTrk(),      helper.end_HCIDTrk(),       CTF::BLC_HCIDTrk,      0);
  iosize += ENCODETRD(helper.begin_padrowTrk(),    helper.end_padrowTrk(),     CTF::BLC_padrowTrk,    0);
  iosize += ENCODETRD(helper.begin_colTrk(),       helper.end_colTrk(),        CTF::BLC_colTrk,       0);
  iosize += ENCODETRD(helper.begin_posTrk(),       helper.end_posTrk(),        CTF::BLC_posTrk,       0);
  iosize += ENCODETRD(helper.begin_slopeTrk(),     helper.end_slopeTrk(),      CTF::BLC_slopeTrk,     0);
  iosize += ENCODETRD(helper.begin_pidTrk(),       helper.end_pidTrk(),        CTF::BLC_pidTrk,       0);

  iosize += ENCODETRD(helper.begin_CIDDig(),       helper.end_CIDDig(),        CTF::BLC_CIDDig,       0);
  iosize += ENCODETRD(helper.begin_ROBDig(),       helper.end_ROBDig(),        CTF::BLC_ROBDig,       0);
  iosize += ENCODETRD(helper.begin_MCMDig(),       helper.end_MCMDig(),        CTF::BLC_MCMDig,       0);
  iosize += ENCODETRD(helper.begin_chanDig(),      helper.end_chanDig(),       CTF::BLC_chanDig,      0);
  iosize += ENCODETRD(helper.begin_ADCDig(),       helper.end_ADCDig(),        CTF::BLC_ADCDig,       0);

  // clang-format on
  CTF::get(buff.data())->print(getPrefix(), mVerbosity);
  finaliseCTFOutput<CTF>(buff);
  iosize.rawIn = trigData.size() * sizeof(TriggerRecord) + sizeof(Tracklet64) * trkData.size() + sizeof(Digit) * digData.size();
  return iosize;
}

/// decode entropy-encoded data to tracklets and digits
template <typename VTRG, typename VTRK, typename VDIG>
o2::ctf::CTFIOSize CTFCoder::decode(const CTF::base& ec, VTRG& trigVec, VTRK& trkVec, VDIG& digVec)
{
  auto header = ec.getHeader();
  checkDictVersion(static_cast<const o2::ctf::CTFDictHeader&>(header));
  ec.print(getPrefix(), mVerbosity);
  std::vector<uint16_t> HCIDTrk, posTrk, CIDDig, ADCDig;
  std::vector<int16_t> bcInc; // RS to not crash at negative increments
  std::vector<uint32_t> entriesTrk, entriesDig, pidTrk;
  std::vector<int32_t> orbitInc; // RS to not crash at negative increments
  std::vector<uint8_t> padrowTrk, colTrk, slopeTrk, ROBDig, MCMDig, chanDig;

  o2::ctf::CTFIOSize iosize;
#define DECODETRD(part, slot) ec.decode(part, int(slot), mCoders[int(slot)].get())
  // clang-format off
  iosize += DECODETRD(bcInc,       CTF::BLC_bcIncTrig);
  iosize += DECODETRD(orbitInc,    CTF::BLC_orbitIncTrig);
  iosize += DECODETRD(entriesTrk,  CTF::BLC_entriesTrk);
  iosize += DECODETRD(entriesDig,  CTF::BLC_entriesDig);

  iosize += DECODETRD(HCIDTrk,     CTF::BLC_HCIDTrk);
  iosize += DECODETRD(padrowTrk,   CTF::BLC_padrowTrk);
  iosize += DECODETRD(colTrk,      CTF::BLC_colTrk);
  iosize += DECODETRD(posTrk,      CTF::BLC_posTrk);
  iosize += DECODETRD(slopeTrk,    CTF::BLC_slopeTrk);
  iosize += DECODETRD(pidTrk,      CTF::BLC_pidTrk);

  iosize += DECODETRD(CIDDig,      CTF::BLC_CIDDig);
  iosize += DECODETRD(ROBDig,      CTF::BLC_ROBDig);
  iosize += DECODETRD(MCMDig,      CTF::BLC_MCMDig);
  iosize += DECODETRD(chanDig,     CTF::BLC_chanDig);
  iosize += DECODETRD(ADCDig,      CTF::BLC_ADCDig);
  // clang-format on
  //
  trigVec.clear();
  trkVec.clear();
  digVec.clear();
  trigVec.reserve(header.nTriggers);
  trkVec.reserve(header.nTracklets);
  digVec.reserve(header.nDigits);
  uint32_t trkCount = 0, digCount = 0, adcCount = 0;
  uint32_t orbit = header.firstOrbit, orbitPrev = 0, orbitPrevGood = mFirstTFOrbit;
  uint16_t bc = header.firstBC;
  bool checkIROK = (mBCShift == 0); // need to check if CTP offset correction does not make the local time negative ?
  static size_t countDiscardMsg = 0;

  for (uint32_t itrig = 0; itrig < header.nTriggers; itrig++) {
    // restore TrigRecord
    if (orbitInc[itrig]) {  // non-0 increment => new orbit
      bc = bcInc[itrig];    // bcInc has absolute meaning
      orbit += orbitInc[itrig];
    } else {
      bc += bcInc[itrig];
    }
    bool triggerOK = true;
    if (mCheckBogusTrig && (bc >= o2::constants::lhc::LHCMaxBunches || orbitInc[itrig] < 0 || bcInc[itrig] < 0 || orbit < orbitPrevGood || (entriesTrk[itrig] == 0 && entriesDig[itrig] == 0))) {
      if (countDiscardMsg < size_t(mCheckBogusTrig) || mCheckBogusTrig < 0) {
        LOGP(alarm, "Bogus TRD trigger at bc:{}/orbit:{} (increments: {}/{}, 1st TF orbit: {}) with {} tracklets and {} digits{}: {}",
             bc, orbit, bcInc[itrig], orbitInc[itrig], mFirstTFOrbit, entriesTrk[itrig], entriesDig[itrig],
             orbitInc[itrig] < 0 ? " (decreasing orbit!) " : "",
             mCheckBogusTrig > 0 ? "discarding" : "discarding disabled");
        if (++countDiscardMsg == size_t(mCheckBogusTrig) && mCheckBogusTrig > 0) {
          LOGP(alarm, "Max amount of warnings ({}) was issued, will not warn anymore", size_t(mCheckBogusTrig));
        }
      }
      if (mCheckBogusTrig > 0) {
        triggerOK = false;
      }
    }
    orbitPrev = orbit;
    o2::InteractionRecord ir{bc, orbit};
    if (triggerOK && (checkIROK || ir.differenceInBC({0, mFirstTFOrbit}) >= mBCShift)) { // correction will be ok
      checkIROK = true;                                                   // don't check anymore since the following checks will yield same
      orbitPrevGood = orbit;
      uint32_t firstEntryTrk = trkVec.size();
      uint16_t hcid = 0;
      for (uint32_t it = 0; it < entriesTrk[itrig]; it++) {
        hcid += HCIDTrk[trkCount]; // 1st tracklet of trigger was encoded with abs HCID, then increments
        trkVec.emplace_back(header.format, hcid, padrowTrk[trkCount], colTrk[trkCount], posTrk[trkCount], slopeTrk[trkCount], pidTrk[trkCount]);
        trkCount++;
      }
      uint32_t firstEntryDig = digVec.size();
      int16_t cid = 0;
      for (uint32_t id = 0; id < entriesDig[itrig]; id++) {
        cid += CIDDig[digCount]; // 1st digit of trigger was encoded with abs CID, then increments
        auto& dig = digVec.emplace_back(cid, ROBDig[digCount], MCMDig[digCount], chanDig[digCount]);
        dig.setADC({&ADCDig[adcCount], constants::TIMEBINS});
        digCount++;
        adcCount += constants::TIMEBINS;
      }
      if (mBCShift && bc < o2::constants::lhc::LHCMaxBunches) { // we don't want corrupted orbit to look as good one after correction
        ir -= mBCShift;
      }

      LOGP(debug, "Storing TRD trigger at {} (increments: {}/{}) with {} tracklets and {} digits", ir.asString(), bcInc[itrig], orbitInc[itrig], entriesTrk[itrig], entriesDig[itrig]);
      trigVec.emplace_back(ir, firstEntryDig, entriesDig[itrig], firstEntryTrk, entriesTrk[itrig]);
    } else { // skip the trigger with negative local time
      trkCount += entriesTrk[itrig];
      digCount += entriesDig[itrig];
      adcCount += constants::TIMEBINS * entriesDig[itrig];
      continue;
    }
  }
  assert(digCount == header.nDigits && trkCount == header.nTracklets && adcCount == (int)ADCDig.size());
  iosize.rawIn = trigVec.size() * sizeof(TriggerRecord) + sizeof(Tracklet64) * trkVec.size() + sizeof(Digit) * digVec.size();
  return iosize;
}

} // namespace trd
} // namespace o2

#endif // O2_TRD_CTFCODER_H
