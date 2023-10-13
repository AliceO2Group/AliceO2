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
#include "HMPIDReconstruction/CTFHelper.h"

class TTree;

namespace o2
{
namespace hmpid
{

class CTFCoder : public o2::ctf::CTFCoderBase
{
 public:
  CTFCoder(o2::ctf::CTFCoderBase::OpType op) : o2::ctf::CTFCoderBase(op, CTF::getNBlocks(), o2::detectors::DetID::HMP) {}
  ~CTFCoder() final = default;

  /// entropy-encode data to buffer with CTF
  template <typename VEC>
  o2::ctf::CTFIOSize encode(VEC& buff, const gsl::span<const Trigger>& trigData, const gsl::span<const Digit>& digData);

  /// entropy decode data from buffer with CTF
  template <typename VTRG, typename VDIG>
  o2::ctf::CTFIOSize decode(const CTF::base& ec, VTRG& trigVec, VDIG& digVec);

  void createCoders(const std::vector<char>& bufVec, o2::ctf::CTFCoderBase::OpType op) final;

 private:
  template <typename VEC>
  o2::ctf::CTFIOSize encode_impl(VEC& buff, const gsl::span<const Trigger>& trigData, const gsl::span<const Digit>& digData);
  void appendToTree(TTree& tree, CTF& ec);
  void readFromTree(TTree& tree, int entry, std::vector<Trigger>& trigVec, std::vector<Digit>& digVec);
  std::vector<Trigger> mTrgRecFilt;
  std::vector<Digit> mDigDataFilt;
};

/// entropy-encode digits and to buffer with CTF
template <typename VEC>
o2::ctf::CTFIOSize CTFCoder::encode(VEC& buff, const gsl::span<const Trigger>& trigData, const gsl::span<const Digit>& digData)
{
  if (mIRFrameSelector.isSet()) { // preselect data
    mTrgRecFilt.clear();
    mDigDataFilt.clear();
    for (const auto& trig : trigData) {
      if (mIRFrameSelector.check(trig.getIr()) >= 0) {
        mTrgRecFilt.push_back(trig);
        auto digIt = digData.begin() + trig.getFirstEntry();
        auto& trigC = mTrgRecFilt.back();
        trigC.setDataRange((int)mDigDataFilt.size(), trig.getNumberOfObjects());
        std::copy(digIt, digIt + trig.getNumberOfObjects(), std::back_inserter(mDigDataFilt));
      }
    }
    return encode_impl(buff, mTrgRecFilt, mDigDataFilt);
  }
  return encode_impl(buff, trigData, digData);
}

template <typename VEC>
o2::ctf::CTFIOSize CTFCoder::encode_impl(VEC& buff, const gsl::span<const Trigger>& trigData, const gsl::span<const Digit>& digData)
{
  using MD = o2::ctf::Metadata::OptStore;
  // what to do which each field: see o2::ctd::Metadata explanation
  constexpr MD optField[CTF::getNBlocks()] = {
    MD::EENCODE_OR_PACK, // BLC_bcIncTrig
    MD::EENCODE_OR_PACK, // BLC_orbitIncTrig
    MD::EENCODE_OR_PACK, // BLC_entriesDig
    MD::EENCODE_OR_PACK, // BLC_ChID
    MD::EENCODE_OR_PACK, // BLC_Q
    MD::EENCODE_OR_PACK, // BLC_Ph
    MD::EENCODE_OR_PACK, // BLC_X
    MD::EENCODE_OR_PACK  // BLC_Y
  };

  CTFHelper helper(trigData, digData);

  // book output size with some margin
  auto szIni = sizeof(CTFHeader) + helper.getSize() * 2. / 3; // will be autoexpanded if needed
  buff.resize(szIni);

  auto ec = CTF::create(buff);
  using ECB = CTF::base;

  ec->setHeader(helper.createHeader());
  assignDictVersion(static_cast<o2::ctf::CTFDictHeader&>(ec->getHeader()));
  ec->setANSHeader(mANSVersion);
  // at every encoding the buffer might be autoexpanded, so we don't work with fixed pointer ec
  o2::ctf::CTFIOSize iosize;
#define ENCODEHMP(beg, end, slot, bits) CTF::get(buff.data())->encode(beg, end, int(slot), bits, optField[int(slot)], &buff, mCoders[int(slot)], getMemMarginFactor());
  // clang-format off
  iosize += ENCODEHMP(helper.begin_bcIncTrig(),    helper.end_bcIncTrig(),     CTF::BLC_bcIncTrig,    0);
  iosize += ENCODEHMP(helper.begin_orbitIncTrig(), helper.end_orbitIncTrig(),  CTF::BLC_orbitIncTrig, 0);
  iosize += ENCODEHMP(helper.begin_entriesDig(),   helper.end_entriesDig(),    CTF::BLC_entriesDig,   0);

  iosize += ENCODEHMP(helper.begin_ChID(),         helper.end_ChID(),          CTF::BLC_ChID,         0);
  iosize += ENCODEHMP(helper.begin_Q(),            helper.end_Q(),             CTF::BLC_Q,            0);
  iosize += ENCODEHMP(helper.begin_Ph(),           helper.end_Ph(),            CTF::BLC_Ph,           0);
  iosize += ENCODEHMP(helper.begin_X(),            helper.end_X(),             CTF::BLC_X,            0);
  iosize += ENCODEHMP(helper.begin_Y(),            helper.end_Y(),             CTF::BLC_Y,            0);

  // clang-format on
  CTF::get(buff.data())->print(getPrefix(), mVerbosity);
  finaliseCTFOutput<CTF>(buff);
  iosize.rawIn = trigData.size() * sizeof(Trigger) + digData.size() * sizeof(Digit);
  return iosize;
}

/// decode entropy-encoded data to digits
template <typename VTRG, typename VDIG>
o2::ctf::CTFIOSize CTFCoder::decode(const CTF::base& ec, VTRG& trigVec, VDIG& digVec)
{
  auto header = ec.getHeader();
  checkDictVersion(static_cast<const o2::ctf::CTFDictHeader&>(header));
  ec.print(getPrefix(), mVerbosity);
  std::vector<int16_t> bcInc;
  std::vector<int32_t> orbitInc;
  std::vector<uint16_t> q;
  std::vector<uint32_t> entriesDig;
  std::vector<uint8_t> chID, ph, x, y;

  o2::ctf::CTFIOSize iosize;
#define DECODEHMP(part, slot) ec.decode(part, int(slot), mCoders[int(slot)])
  // clang-format off
  iosize += DECODEHMP(bcInc,       CTF::BLC_bcIncTrig);
  iosize += DECODEHMP(orbitInc,    CTF::BLC_orbitIncTrig);
  iosize += DECODEHMP(entriesDig,  CTF::BLC_entriesDig);

  iosize += DECODEHMP(chID,        CTF::BLC_ChID);
  iosize += DECODEHMP(q,           CTF::BLC_Q);
  iosize += DECODEHMP(ph,          CTF::BLC_Ph);
  iosize += DECODEHMP(x,           CTF::BLC_X);
  iosize += DECODEHMP(y,           CTF::BLC_Y);
  // clang-format on
  //
  trigVec.clear();
  digVec.clear();
  trigVec.reserve(header.nTriggers);
  digVec.reserve(header.nDigits);

  uint32_t digCount = 0;
  o2::InteractionRecord ir(header.firstBC, header.firstOrbit);
  bool checkIROK = (mBCShift == 0); // need to check if CTP offset correction does not make the local time negative ?
  for (uint32_t itrig = 0; itrig < header.nTriggers; itrig++) {
    // restore TrigRecord
    if (orbitInc[itrig]) {  // non-0 increment => new orbit
      ir.bc = bcInc[itrig]; // bcInc has absolute meaning
      ir.orbit += orbitInc[itrig];
    } else {
      ir.bc += bcInc[itrig];
    }

    int8_t chid = 0;
    if (checkIROK || canApplyBCShift(ir)) { // correction will be ok
      checkIROK = true;
    } else { // correction would make IR prior to mFirstTFOrbit, skip
      digCount += entriesDig[itrig];
      continue;
    }
    uint32_t firstEntryDig = digVec.size();
    for (uint32_t id = 0; id < entriesDig[itrig]; id++) {
      chid += chID[digCount]; // 1st digit of trigger was encoded with abs ChID, then increments
      auto& dig = digVec.emplace_back(chid, ph[digCount], x[digCount], y[digCount], q[digCount]);
      digCount++;
    }

    trigVec.emplace_back(ir - mBCShift, firstEntryDig, entriesDig[itrig]);
  }
  assert(digCount == header.nDigits);
  iosize.rawIn = trigVec.size() * sizeof(Trigger) + digVec.size() * sizeof(Digit);
  return iosize;
}

} // namespace hmpid
} // namespace o2

#endif // O2_HMP_CTFCODER_H
