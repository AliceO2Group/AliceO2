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
/// \brief class for entropy encoding/decoding of CTP data

#ifndef O2_CTP_CTFCODER_H
#define O2_CTP_CTFCODER_H

#include <algorithm>
#include <iterator>
#include <string>
#include <array>
#include "DataFormatsCTP/CTF.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsBase/CTFCoderBase.h"
#include "rANS/rans.h"
#include "CTPReconstruction/CTFHelper.h"

class TTree;

namespace o2
{
namespace ctp
{

class CTFCoder : public o2::ctf::CTFCoderBase
{
 public:
  CTFCoder(o2::ctf::CTFCoderBase::OpType op) : o2::ctf::CTFCoderBase(op, CTF::getNBlocks(), o2::detectors::DetID::CTP) {}
  ~CTFCoder() final = default;

  /// entropy-encode data to buffer with CTF
  template <typename VEC>
  o2::ctf::CTFIOSize encode(VEC& buff, const gsl::span<const CTPDigit>& data, const LumiInfo& lumi);

  /// entropy decode data from buffer with CTF
  template <typename VTRG>
  o2::ctf::CTFIOSize decode(const CTF::base& ec, VTRG& data, LumiInfo& lumi);

  void createCoders(const std::vector<char>& bufVec, o2::ctf::CTFCoderBase::OpType op) final;

 private:
  template <typename VEC>
  o2::ctf::CTFIOSize encode_impl(VEC& buff, const gsl::span<const CTPDigit>& data, const LumiInfo& lumi);

  void appendToTree(TTree& tree, CTF& ec);
  void readFromTree(TTree& tree, int entry, std::vector<CTPDigit>& data, LumiInfo& lumi);
  std::vector<CTPDigit> mDataFilt;
};

/// entropy-encode clusters to buffer with CTF
template <typename VEC>
o2::ctf::CTFIOSize CTFCoder::encode(VEC& buff, const gsl::span<const CTPDigit>& data, const LumiInfo& lumi)
{
  if (mIRFrameSelector.isSet()) { // preselect data
    mDataFilt.clear();
    for (const auto& trig : data) {
      if (mIRFrameSelector.check(trig.intRecord) >= 0) {
        mDataFilt.push_back(trig);
      }
    }
    return encode_impl(buff, mDataFilt, lumi);
  }
  return encode_impl(buff, data, lumi);
}

template <typename VEC>
o2::ctf::CTFIOSize CTFCoder::encode_impl(VEC& buff, const gsl::span<const CTPDigit>& data, const LumiInfo& lumi)
{
  using MD = o2::ctf::Metadata::OptStore;
  // what to do which each field: see o2::ctd::Metadata explanation
  constexpr MD optField[CTF::getNBlocks()] = {
    MD::EENCODE, // BLC_bcIncTrig
    MD::EENCODE, // BLC_orbitIncTrig
    MD::EENCODE, // BLC_bytesInput
    MD::EENCODE, // BLC_bytesClass
  };

  CTFHelper helper(data);

  // book output size with some margin
  auto szIni = sizeof(CTFHeader) + helper.getSize() * 2. / 3; // will be autoexpanded if needed
  buff.resize(szIni);

  auto ec = CTF::create(buff);
  using ECB = CTF::base;

  ec->setHeader(helper.createHeader(lumi));
  assignDictVersion(static_cast<o2::ctf::CTFDictHeader&>(ec->getHeader()));
  ec->getANSHeader().majorVersion = 0;
  ec->getANSHeader().minorVersion = 1;
  // at every encoding the buffer might be autoexpanded, so we don't work with fixed pointer ec
  o2::ctf::CTFIOSize iosize;
#define ENCODECTP(beg, end, slot, bits) CTF::get(buff.data())->encode(beg, end, int(slot), bits, optField[int(slot)], &buff, mCoders[int(slot)].get(), getMemMarginFactor());
  // clang-format off
  iosize += ENCODECTP(helper.begin_bcIncTrig(),    helper.end_bcIncTrig(),     CTF::BLC_bcIncTrig,    0);
  iosize += ENCODECTP(helper.begin_orbitIncTrig(), helper.end_orbitIncTrig(),  CTF::BLC_orbitIncTrig, 0);
  iosize += ENCODECTP(helper.begin_bytesInput(),  helper.end_bytesInput(),     CTF::BLC_bytesInput,   0);
  iosize += ENCODECTP(helper.begin_bytesClass(),  helper.end_bytesClass(),     CTF::BLC_bytesClass,   0);
  // clang-format on
  CTF::get(buff.data())->print(getPrefix(), mVerbosity);
  finaliseCTFOutput<CTF>(buff);
  iosize.rawIn = data.size() * sizeof(CTPDigit);
  return iosize;
}

/// decode entropy-encoded digits
template <typename VTRG>
o2::ctf::CTFIOSize CTFCoder::decode(const CTF::base& ec, VTRG& data, LumiInfo& lumi)
{
  auto header = ec.getHeader();
  checkDictVersion(static_cast<const o2::ctf::CTFDictHeader&>(header));
  ec.print(getPrefix(), mVerbosity);
  std::vector<uint16_t> bcInc;
  std::vector<uint32_t> orbitInc;
  std::vector<uint8_t> bytesInput, bytesClass;

  o2::ctf::CTFIOSize iosize;
#define DECODECTP(part, slot) ec.decode(part, int(slot), mCoders[int(slot)].get())
  // clang-format off
  iosize += DECODECTP(bcInc,       CTF::BLC_bcIncTrig);
  iosize += DECODECTP(orbitInc,    CTF::BLC_orbitIncTrig);
  iosize += DECODECTP(bytesInput,  CTF::BLC_bytesInput);
  iosize += DECODECTP(bytesClass,  CTF::BLC_bytesClass);
  // clang-format on
  //
  data.clear();

  uint32_t firstEntry = 0, digCount = 0;
  o2::InteractionRecord ir(header.firstBC, header.firstOrbit);
  lumi.nHBFCounted = header.lumiNHBFs;
  lumi.counts = header.lumiCounts;
  lumi.orbit = header.lumiOrbit;
  auto itInp = bytesInput.begin();
  auto itCls = bytesClass.begin();

  for (uint32_t itrig = 0; itrig < header.nTriggers; itrig++) {
    // restore TrigRecord
    if (orbitInc[itrig]) {  // non-0 increment => new orbit
      ir.bc = bcInc[itrig]; // bcInc has absolute meaning
      ir.orbit += orbitInc[itrig];
    } else {
      ir.bc += bcInc[itrig];
    }
    auto& dig = data.emplace_back();
    dig.intRecord = ir;
    for (int i = 0; i < CTFHelper::CTPInpNBytes; i++) {
      dig.CTPInputMask |= static_cast<uint64_t>(*itInp++) << (8 * i);
    }
    for (int i = 0; i < CTFHelper::CTPClsNBytes; i++) {
      dig.CTPClassMask |= static_cast<uint64_t>(*itCls++) << (8 * i);
    }
  }
  assert(itInp == bytesInput.end());
  assert(itCls == bytesClass.end());
  iosize.rawIn = data.size() * sizeof(CTPDigit);
  return iosize;
}

} // namespace ctp
} // namespace o2

#endif // O2_CTP_CTFCODER_H
