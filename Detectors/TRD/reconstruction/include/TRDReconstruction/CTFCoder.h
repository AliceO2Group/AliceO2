// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

class TTree;

namespace o2
{
namespace trd
{

class CTFCoder : public o2::ctf::CTFCoderBase
{
 public:
  CTFCoder() : o2::ctf::CTFCoderBase(CTF::getNBlocks(), o2::detectors::DetID::TRD) {}
  ~CTFCoder() = default;

  /// entropy-encode data to buffer with CTF
  template <typename VEC>
  void encode(VEC& buff, const gsl::span<const TriggerRecord>& trigData, const gsl::span<const Tracklet64>& trkData, const gsl::span<const Digit>& digData);

  /// entropy decode data from buffer with CTF
  template <typename VTRG, typename VTRK, typename VDIG>
  void decode(const CTF::base& ec, VTRG& trigVec, VTRK& trkVec, VDIG& digVec);

  void createCoders(const std::string& dictPath, o2::ctf::CTFCoderBase::OpType op);

 private:
  void appendToTree(TTree& tree, CTF& ec);
  void readFromTree(TTree& tree, int entry, std::vector<TriggerRecord>& trigVec, std::vector<Tracklet64>& trkVec, std::vector<Digit>& digVec);
};

/// entropy-encode digits and tracklets to buffer with CTF
template <typename VEC>
void CTFCoder::encode(VEC& buff, const gsl::span<const TriggerRecord>& trigData, const gsl::span<const Tracklet64>& trkData, const gsl::span<const Digit>& digData)
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

  CTFHelper helper(trigData, trkData, digData);

  // book output size with some margin
  auto szIni = sizeof(CTFHeader) + helper.getSize() * 2. / 3; // will be autoexpanded if needed
  buff.resize(szIni);

  auto ec = CTF::create(buff);
  using ECB = CTF::base;

  ec->setHeader(helper.createHeader());
  ec->getANSHeader().majorVersion = 0;
  ec->getANSHeader().minorVersion = 1;
  // at every encoding the buffer might be autoexpanded, so we don't work with fixed pointer ec
#define ENCODETRD(beg, end, slot, bits) CTF::get(buff.data())->encode(beg, end, int(slot), bits, optField[int(slot)], &buff, mCoders[int(slot)].get());
  // clang-format off
  ENCODETRD(helper.begin_bcIncTrig(),    helper.end_bcIncTrig(),     CTF::BLC_bcIncTrig,    0);
  ENCODETRD(helper.begin_orbitIncTrig(), helper.end_orbitIncTrig(),  CTF::BLC_orbitIncTrig, 0);
  ENCODETRD(helper.begin_entriesTrk(),   helper.end_entriesTrk(),    CTF::BLC_entriesTrk,   0);
  ENCODETRD(helper.begin_entriesDig(),   helper.end_entriesDig(),    CTF::BLC_entriesDig,   0);

  ENCODETRD(helper.begin_HCIDTrk(),      helper.end_HCIDTrk(),       CTF::BLC_HCIDTrk,      0);
  ENCODETRD(helper.begin_padrowTrk(),    helper.end_padrowTrk(),     CTF::BLC_padrowTrk,    0);
  ENCODETRD(helper.begin_colTrk(),       helper.end_colTrk(),        CTF::BLC_colTrk,       0);
  ENCODETRD(helper.begin_posTrk(),       helper.end_posTrk(),        CTF::BLC_posTrk,       0);
  ENCODETRD(helper.begin_slopeTrk(),     helper.end_slopeTrk(),      CTF::BLC_slopeTrk,     0);
  ENCODETRD(helper.begin_pidTrk(),       helper.end_pidTrk(),        CTF::BLC_pidTrk,       0);

  ENCODETRD(helper.begin_CIDDig(),       helper.end_CIDDig(),        CTF::BLC_CIDDig,       0);
  ENCODETRD(helper.begin_ROBDig(),       helper.end_ROBDig(),        CTF::BLC_ROBDig,       0);
  ENCODETRD(helper.begin_MCMDig(),       helper.end_MCMDig(),        CTF::BLC_MCMDig,       0);
  ENCODETRD(helper.begin_chanDig(),      helper.end_chanDig(),       CTF::BLC_chanDig,      0);
  ENCODETRD(helper.begin_ADCDig(),       helper.end_ADCDig(),        CTF::BLC_ADCDig,       0);

  // clang-format on
  CTF::get(buff.data())->print(getPrefix());
}

/// decode entropy-encoded data to tracklets and digits
template <typename VTRG, typename VTRK, typename VDIG>
void CTFCoder::decode(const CTF::base& ec, VTRG& trigVec, VTRK& trkVec, VDIG& digVec)
{
  auto header = ec.getHeader();
  ec.print(getPrefix());
  std::vector<uint16_t> bcInc, HCIDTrk, posTrk, CIDDig, ADCDig;
  std::vector<uint32_t> orbitInc, entriesTrk, entriesDig, pidTrk;
  std::vector<uint8_t> padrowTrk, colTrk, slopeTrk, ROBDig, MCMDig, chanDig;

#define DECODETRD(part, slot) ec.decode(part, int(slot), mCoders[int(slot)].get())
  // clang-format off
  DECODETRD(bcInc,       CTF::BLC_bcIncTrig);
  DECODETRD(orbitInc,    CTF::BLC_orbitIncTrig);
  DECODETRD(entriesTrk,  CTF::BLC_entriesTrk);
  DECODETRD(entriesDig,  CTF::BLC_entriesDig);

  DECODETRD(HCIDTrk,     CTF::BLC_HCIDTrk);
  DECODETRD(padrowTrk,   CTF::BLC_padrowTrk);
  DECODETRD(colTrk,      CTF::BLC_colTrk);
  DECODETRD(posTrk,      CTF::BLC_posTrk);
  DECODETRD(slopeTrk,    CTF::BLC_slopeTrk);
  DECODETRD(pidTrk,      CTF::BLC_pidTrk);

  DECODETRD(CIDDig,      CTF::BLC_CIDDig);
  DECODETRD(ROBDig,      CTF::BLC_ROBDig);
  DECODETRD(MCMDig,      CTF::BLC_MCMDig);
  DECODETRD(chanDig,     CTF::BLC_chanDig);
  DECODETRD(ADCDig,      CTF::BLC_ADCDig);
  // clang-format on
  //
  trigVec.clear();
  trkVec.clear();
  digVec.clear();
  trigVec.reserve(header.nTriggers);
  trkVec.reserve(header.nTracklets);
  digVec.reserve(header.nDigits);

  uint32_t trkCount = 0, digCount = 0, adcCount = 0;
  o2::InteractionRecord ir(header.firstBC, header.firstOrbit);

  for (uint32_t itrig = 0; itrig < header.nTriggers; itrig++) {
    // restore TrigRecord
    if (orbitInc[itrig]) {  // non-0 increment => new orbit
      ir.bc = bcInc[itrig]; // bcInc has absolute meaning
      ir.orbit += orbitInc[itrig];
    } else {
      ir.bc += bcInc[itrig];
    }

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

    trigVec.emplace_back(ir, firstEntryDig, entriesDig[itrig], firstEntryTrk, entriesTrk[itrig]);
  }
  assert(digCount == header.nDigits && trkCount == header.nTracklets && adcCount == (int)ADCDig.size());
}

} // namespace trd
} // namespace o2

#endif // O2_TRD_CTFCODER_H
