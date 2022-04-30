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
  void encode(VEC& buff, const gsl::span<const CTPDigit>& data);

  /// entropy decode data from buffer with CTF
  template <typename VTRG>
  void decode(const CTF::base& ec, VTRG& data);

  void createCoders(const std::vector<char>& bufVec, o2::ctf::CTFCoderBase::OpType op) final;

 private:
  void appendToTree(TTree& tree, CTF& ec);
  void readFromTree(TTree& tree, int entry, std::vector<CTPDigit>& data);
};

/// entropy-encode clusters to buffer with CTF
template <typename VEC>
void CTFCoder::encode(VEC& buff, const gsl::span<const CTPDigit>& data)
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

  ec->setHeader(helper.createHeader());
  assignDictVersion(static_cast<o2::ctf::CTFDictHeader&>(ec->getHeader()));
  ec->getANSHeader().majorVersion = 0;
  ec->getANSHeader().minorVersion = 1;
  // at every encoding the buffer might be autoexpanded, so we don't work with fixed pointer ec
#define ENCODECTP(beg, end, slot, bits) CTF::get(buff.data())->encode(beg, end, int(slot), bits, optField[int(slot)], &buff, mCoders[int(slot)].get(), getMemMarginFactor());
  // clang-format off
  ENCODECTP(helper.begin_bcIncTrig(),    helper.end_bcIncTrig(),     CTF::BLC_bcIncTrig,    0);
  ENCODECTP(helper.begin_orbitIncTrig(), helper.end_orbitIncTrig(),  CTF::BLC_orbitIncTrig, 0);

  ENCODECTP(helper.begin_bytesInput(),  helper.end_bytesInput(),     CTF::BLC_bytesInput,   0);
  ENCODECTP(helper.begin_bytesClass(),  helper.end_bytesClass(),     CTF::BLC_bytesClass,   0);

  // clang-format on
  CTF::get(buff.data())->print(getPrefix(), mVerbosity);
}

/// decode entropy-encoded digits
template <typename VTRG>
void CTFCoder::decode(const CTF::base& ec, VTRG& data)
{
  auto header = ec.getHeader();
  checkDictVersion(static_cast<const o2::ctf::CTFDictHeader&>(header));
  ec.print(getPrefix(), mVerbosity);
  std::vector<uint16_t> bcInc;
  std::vector<uint32_t> orbitInc;
  std::vector<uint8_t> bytesInput, bytesClass;

#define DECODECTP(part, slot) ec.decode(part, int(slot), mCoders[int(slot)].get())
  // clang-format off
  DECODECTP(bcInc,       CTF::BLC_bcIncTrig);
  DECODECTP(orbitInc,    CTF::BLC_orbitIncTrig);
  DECODECTP(bytesInput,  CTF::BLC_bytesInput);
  DECODECTP(bytesClass,  CTF::BLC_bytesClass);
  // clang-format on
  //
  data.clear();

  uint32_t firstEntry = 0, digCount = 0;
  o2::InteractionRecord ir(header.firstBC, header.firstOrbit);
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
}

} // namespace ctp
} // namespace o2

#endif // O2_CTP_CTFCODER_H
