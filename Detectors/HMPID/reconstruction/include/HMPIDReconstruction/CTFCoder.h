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
/// \brief class for entropy encoding/decoding of HMPID data

#ifndef O2_HMPID_CTFCODER_H
#define O2_HMPID_CTFCODER_H

#include <algorithm>
#include <iterator>
#include <string>
#include <array>
#include "DataFormatsHMP/CTF.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsBase/CTFCoderBase.h"
#include "rANS/rans.h"
#include "HMPIDReconstruction/CTFHelper.h"

class TTree;

namespace o2
{
namespace hmpid
{

class CTFCoder : public o2::ctf::CTFCoderBase
{
 public:
  CTFCoder() : o2::ctf::CTFCoderBase(CTF::getNBlocks(), o2::detectors::DetID::HMP) {}
  ~CTFCoder() = default;

  /// entropy-encode data to buffer with CTF
  template <typename VEC>
  void encode(VEC& buff, const gsl::span<const Trigger>& trigData, const gsl::span<const Digit>& digData);

  /// entropy decode data from buffer with CTF
  template <typename VTRG, typename VDIG>
  void decode(const CTF::base& ec, VTRG& trigVec, VDIG& digVec);

  void createCoders(const std::string& dictPath, o2::ctf::CTFCoderBase::OpType op);

 private:
  void appendToTree(TTree& tree, CTF& ec);
  void readFromTree(TTree& tree, int entry, std::vector<Trigger>& trigVec, std::vector<Digit>& digVec);
};

/// entropy-encode digits and to buffer with CTF
template <typename VEC>
void CTFCoder::encode(VEC& buff, const gsl::span<const Trigger>& trigData, const gsl::span<const Digit>& digData)
{
  using MD = o2::ctf::Metadata::OptStore;
  // what to do which each field: see o2::ctd::Metadata explanation
  constexpr MD optField[CTF::getNBlocks()] = {
    MD::EENCODE, // BLC_bcIncTrig
    MD::EENCODE, // BLC_orbitIncTrig
    MD::EENCODE, // BLC_entriesDig
    MD::EENCODE, // BLC_ChID
    MD::EENCODE, // BLC_Q
    MD::EENCODE, // BLC_Ph
    MD::EENCODE, // BLC_X
    MD::EENCODE  // BLC_Y
  };

  CTFHelper helper(trigData, digData);

  // book output size with some margin
  auto szIni = sizeof(CTFHeader) + helper.getSize() * 2. / 3; // will be autoexpanded if needed
  buff.resize(szIni);

  auto ec = CTF::create(buff);
  using ECB = CTF::base;

  ec->setHeader(helper.createHeader());
  ec->getANSHeader().majorVersion = 0;
  ec->getANSHeader().minorVersion = 1;
  // at every encoding the buffer might be autoexpanded, so we don't work with fixed pointer ec
#define ENCODEHMP(beg, end, slot, bits) CTF::get(buff.data())->encode(beg, end, int(slot), bits, optField[int(slot)], &buff, mCoders[int(slot)].get());
  // clang-format off
  ENCODEHMP(helper.begin_bcIncTrig(),    helper.end_bcIncTrig(),     CTF::BLC_bcIncTrig,    0);
  ENCODEHMP(helper.begin_orbitIncTrig(), helper.end_orbitIncTrig(),  CTF::BLC_orbitIncTrig, 0);
  ENCODEHMP(helper.begin_entriesDig(),   helper.end_entriesDig(),    CTF::BLC_entriesDig,   0);
  
  ENCODEHMP(helper.begin_ChID(),         helper.end_ChID(),          CTF::BLC_ChID,         0);
  ENCODEHMP(helper.begin_Q(),            helper.end_Q(),             CTF::BLC_Q,            0);
  ENCODEHMP(helper.begin_Ph(),           helper.end_Ph(),            CTF::BLC_Ph,           0);
  ENCODEHMP(helper.begin_X(),            helper.end_X(),             CTF::BLC_X,            0);
  ENCODEHMP(helper.begin_Y(),            helper.end_Y(),             CTF::BLC_Y,            0);

  // clang-format on
  CTF::get(buff.data())->print(getPrefix());
}

/// decode entropy-encoded data to digits
template <typename VTRG, typename VDIG>
void CTFCoder::decode(const CTF::base& ec, VTRG& trigVec, VDIG& digVec)
{
  auto header = ec.getHeader();
  ec.print(getPrefix());
  std::vector<uint16_t> bcInc, q;
  std::vector<uint32_t> orbitInc, entriesDig;
  std::vector<uint8_t> chID, ph, x, y;

#define DECODEHMP(part, slot) ec.decode(part, int(slot), mCoders[int(slot)].get())
  // clang-format off
  DECODEHMP(bcInc,       CTF::BLC_bcIncTrig);
  DECODEHMP(orbitInc,    CTF::BLC_orbitIncTrig);
  DECODEHMP(entriesDig,  CTF::BLC_entriesDig);

  DECODEHMP(chID,        CTF::BLC_ChID);
  DECODEHMP(q,           CTF::BLC_Q);
  DECODEHMP(ph,          CTF::BLC_Ph);
  DECODEHMP(x,           CTF::BLC_X);
  DECODEHMP(y,           CTF::BLC_Y);
  // clang-format on
  //
  trigVec.clear();
  digVec.clear();
  trigVec.reserve(header.nTriggers);
  digVec.reserve(header.nDigits);

  uint32_t digCount = 0;
  o2::InteractionRecord ir(header.firstBC, header.firstOrbit);

  for (uint32_t itrig = 0; itrig < header.nTriggers; itrig++) {
    // restore TrigRecord
    if (orbitInc[itrig]) {  // non-0 increment => new orbit
      ir.bc = bcInc[itrig]; // bcInc has absolute meaning
      ir.orbit += orbitInc[itrig];
    } else {
      ir.bc += bcInc[itrig];
    }

    uint32_t firstEntryDig = digVec.size();
    int8_t chid = 0;
    for (uint32_t id = 0; id < entriesDig[itrig]; id++) {
      chid += chID[digCount]; // 1st digit of trigger was encoded with abs ChID, then increments
      auto& dig = digVec.emplace_back(chid, ph[digCount], x[digCount], y[digCount], q[digCount]);
      digCount++;
    }

    trigVec.emplace_back(ir, firstEntryDig, entriesDig[itrig]);
  }
  assert(digCount == header.nDigits);
}

} // namespace hmpid
} // namespace o2

#endif // O2_HMP_CTFCODER_H
