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
/// \brief class for entropy encoding/decoding of MCH digit data

#ifndef O2_MCH_CTFCODER_H
#define O2_MCH_CTFCODER_H

#include <algorithm>
#include <iterator>
#include <string>
#include <array>
#include "DataFormatsMCH/CTF.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/Digit.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsBase/CTFCoderBase.h"
#include "MCHCTF/CTFHelper.h"

class TTree;

namespace o2
{
namespace mch
{

class CTFCoder : public o2::ctf::CTFCoderBase
{
 public:
  CTFCoder(o2::ctf::CTFCoderBase::OpType op) : o2::ctf::CTFCoderBase(op, CTF::getNBlocks(), o2::detectors::DetID::MCH) {}
  ~CTFCoder() final = default;

  /// entropy-encode data to buffer with CTF
  template <typename VEC>
  o2::ctf::CTFIOSize encode(VEC& buff, const gsl::span<const ROFRecord>& rofData, const gsl::span<const Digit>& digData);

  /// entropy decode data from buffer with CTF
  template <typename VROF, typename VCOL>
  o2::ctf::CTFIOSize decode(const CTF::base& ec, VROF& rofVec, VCOL& digVec);

  void createCoders(const std::vector<char>& bufVec, o2::ctf::CTFCoderBase::OpType op) final;

 private:
  template <typename VEC>
  o2::ctf::CTFIOSize encode_impl(VEC& buff, const gsl::span<const ROFRecord>& rofData, const gsl::span<const Digit>& digData);
  void appendToTree(TTree& tree, CTF& ec);
  void readFromTree(TTree& tree, int entry, std::vector<ROFRecord>& rofVec, std::vector<Digit>& digVec);

  std::vector<ROFRecord> mROFRecFilt;
  std::vector<Digit> mDigDataFilt;
};

/// entropy-encode clusters to buffer with CTF
template <typename VEC>
o2::ctf::CTFIOSize CTFCoder::encode(VEC& buff, const gsl::span<const ROFRecord>& rofData, const gsl::span<const Digit>& digData)
{
  if (mIRFrameSelector.isSet()) { // preselect data
    mROFRecFilt.clear();
    mDigDataFilt.clear();
    for (const auto& rof : rofData) {
      if (mIRFrameSelector.check(rof.getBCData()) >= 0) {
        mROFRecFilt.push_back(rof);
        auto digIt = digData.begin() + rof.getFirstIdx();
        auto& rofC = mROFRecFilt.back();
        rofC.setDataRef((int)mDigDataFilt.size(), rof.getNEntries());
        std::copy(digIt, digIt + rofC.getNEntries(), std::back_inserter(mDigDataFilt));
      }
    }
    return encode_impl(buff, mROFRecFilt, mDigDataFilt);
  }
  return encode_impl(buff, rofData, digData);
}

template <typename VEC>
o2::ctf::CTFIOSize CTFCoder::encode_impl(VEC& buff, const gsl::span<const ROFRecord>& rofData, const gsl::span<const Digit>& digData)
{
  using MD = o2::ctf::Metadata::OptStore;
  // what to do which each field: see o2::ctd::Metadata explanation
  constexpr MD optField[CTF::getNBlocks()] = {
    MD::EENCODE_OR_PACK, // BLC_bcIncROF
    MD::EENCODE_OR_PACK, // BLC_orbitIncROF
    MD::EENCODE_OR_PACK, // BLC_nDigitsROF
    MD::EENCODE_OR_PACK, // BLC_tfTime
    MD::EENCODE_OR_PACK, // BLC_nSamples
    MD::EENCODE_OR_PACK, // BLC_isSaturated
    MD::EENCODE_OR_PACK, // BLC_detID
    MD::EENCODE_OR_PACK, // BLC_padID
    MD::EENCODE_OR_PACK  // BLC_ADC
  };
  CTFHelper helper(rofData, digData);
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
#define ENCODEMCH(beg, end, slot, bits) CTF::get(buff.data())->encode(beg, end, int(slot), bits, optField[int(slot)], &buff, mCoders[int(slot)], getMemMarginFactor());
  // clang-format off
  iosize += ENCODEMCH(helper.begin_bcIncROF(),    helper.end_bcIncROF(),     CTF::BLC_bcIncROF,     0);
  iosize += ENCODEMCH(helper.begin_orbitIncROF(), helper.end_orbitIncROF(),  CTF::BLC_orbitIncROF,  0);
  iosize += ENCODEMCH(helper.begin_nDigitsROF(),  helper.end_nDigitsROF(),   CTF::BLC_nDigitsROF,   0);

  iosize += ENCODEMCH(helper.begin_tfTime(),      helper.end_tfTime(),       CTF::BLC_tfTime,       0);
  iosize += ENCODEMCH(helper.begin_nSamples(),    helper.end_nSamples(),     CTF::BLC_nSamples,     0);
  iosize += ENCODEMCH(helper.begin_isSaturated(), helper.end_isSaturated(),  CTF::BLC_isSaturated,  0);
  iosize += ENCODEMCH(helper.begin_detID(),       helper.end_detID(),        CTF::BLC_detID,        0);
  iosize += ENCODEMCH(helper.begin_padID(),       helper.end_padID(),        CTF::BLC_padID,        0);
  iosize += ENCODEMCH(helper.begin_ADC()  ,       helper.end_ADC(),          CTF::BLC_ADC,          0);
  // clang-format on
  CTF::get(buff.data())->print(getPrefix(), mVerbosity);
  finaliseCTFOutput<CTF>(buff);
  iosize.rawIn = sizeof(ROFRecord) * rofData.size() + sizeof(Digit) * digData.size();
  return iosize;
}

/// decode entropy-encoded clusters to standard compact clusters
template <typename VROF, typename VCOL>
o2::ctf::CTFIOSize CTFCoder::decode(const CTF::base& ec, VROF& rofVec, VCOL& digVec)
{
  auto header = ec.getHeader();
  checkDictVersion(static_cast<const o2::ctf::CTFDictHeader&>(header));
  ec.print(getPrefix(), mVerbosity);

  std::vector<uint16_t> nSamples;
  std::vector<uint32_t> ADC, nDigits;
  std::vector<int32_t> orbitInc;
  std::vector<int32_t> tfTime;
  std::vector<int16_t> bcInc, detID, padID;
  std::vector<uint8_t> isSaturated;

  o2::ctf::CTFIOSize iosize;
#define DECODEMCH(part, slot) ec.decode(part, int(slot), mCoders[int(slot)])
  // clang-format off
  iosize += DECODEMCH(bcInc,       CTF::BLC_bcIncROF);
  iosize += DECODEMCH(orbitInc,    CTF::BLC_orbitIncROF);
  iosize += DECODEMCH(nDigits,     CTF::BLC_nDigitsROF);

  iosize += DECODEMCH(tfTime,      CTF::BLC_tfTime);
  iosize += DECODEMCH(nSamples,    CTF::BLC_nSamples);
  iosize += DECODEMCH(isSaturated, CTF::BLC_isSaturated);
  iosize += DECODEMCH(detID,       CTF::BLC_detID);
  iosize += DECODEMCH(padID,       CTF::BLC_padID);
  iosize += DECODEMCH(ADC,         CTF::BLC_ADC);
  // clang-format on
  //
  rofVec.clear();
  digVec.clear();
  rofVec.reserve(header.nROFs);
  digVec.reserve(header.nDigits);

  uint32_t firstEntry = 0, rofCount = 0, digCount = 0;
  o2::InteractionRecord ir(header.firstBC, header.firstOrbit);

  for (uint32_t irof = 0; irof < header.nROFs; irof++) {
    // restore ROFRecord
    if (orbitInc[irof]) {  // non-0 increment => new orbit
      ir.bc = bcInc[irof]; // bcInc has absolute meaning
      ir.orbit += orbitInc[irof];
    } else {
      ir.bc += bcInc[irof];
    }

    firstEntry = digVec.size();
    for (auto ic = 0; ic < nDigits[irof]; ic++) {
      digVec.emplace_back(Digit{detID[digCount], padID[digCount], ADC[digCount], tfTime[digCount], nSamples[digCount]});
      digVec.back().setSaturated(isSaturated[digCount]);
      digCount++;
    }
    rofVec.emplace_back(ROFRecord{ir, int(firstEntry), static_cast<int>(nDigits[irof])});
  }
  assert(rofVec.size() == header.nROFs);
  assert(digCount == header.nDigits);
  iosize.rawIn = sizeof(ROFRecord) * rofVec.size() + sizeof(Digit) * digVec.size();
  return iosize;
}

} // namespace mch
} // namespace o2

#endif // O2_MCH_CTFCODER_H
