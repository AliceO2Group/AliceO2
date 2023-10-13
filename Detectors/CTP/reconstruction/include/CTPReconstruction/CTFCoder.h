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
#include "CTPReconstruction/CTFHelper.h"
#include "CTPReconstruction/RawDataDecoder.h"

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

  /// add CTP related shifts
  template <typename CTF>
  bool finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj);

  void createCoders(const std::vector<char>& bufVec, o2::ctf::CTFCoderBase::OpType op) final;
  void setDecodeInps(bool decodeinps) { mDecodeInps = decodeinps; }
  bool canApplyBCShiftInputs(const o2::InteractionRecord& ir) const { return canApplyBCShift(ir, mBCShiftInputs); }

 private:
  template <typename VEC>
  o2::ctf::CTFIOSize encode_impl(VEC& buff, const gsl::span<const CTPDigit>& data, const LumiInfo& lumi);

  void appendToTree(TTree& tree, CTF& ec);
  void readFromTree(TTree& tree, int entry, std::vector<CTPDigit>& data, LumiInfo& lumi);
  std::vector<CTPDigit> mDataFilt;
  int mBCShiftInputs = 0;
  bool mDecodeInps = false;
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
    MD::EENCODE_OR_PACK, // BLC_bcIncTrig
    MD::EENCODE_OR_PACK, // BLC_orbitIncTrig
    MD::EENCODE_OR_PACK, // BLC_bytesInput
    MD::EENCODE_OR_PACK, // BLC_bytesClass
  };

  CTFHelper helper(data);

  // book output size with some margin
  auto szIni = sizeof(CTFHeader) + helper.getSize() * 2. / 3; // will be autoexpanded if needed
  buff.resize(szIni);

  auto ec = CTF::create(buff);
  using ECB = CTF::base;

  ec->setHeader(helper.createHeader(lumi));
  assignDictVersion(static_cast<o2::ctf::CTFDictHeader&>(ec->getHeader()));
  ec->setANSHeader(mANSVersion);
  // at every encoding the buffer might be autoexpanded, so we don't work with fixed pointer ec
  o2::ctf::CTFIOSize iosize;
#define ENCODECTP(beg, end, slot, bits) CTF::get(buff.data())->encode(beg, end, int(slot), bits, optField[int(slot)], &buff, mCoders[int(slot)], getMemMarginFactor());
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
  std::vector<int16_t> bcInc;
  std::vector<int32_t> orbitInc;
  std::vector<uint8_t> bytesInput, bytesClass;

  o2::ctf::CTFIOSize iosize;
#define DECODECTP(part, slot) ec.decode(part, int(slot), mCoders[int(slot)])
  // clang-format off
  iosize += DECODECTP(bcInc,       CTF::BLC_bcIncTrig);
  iosize += DECODECTP(orbitInc,    CTF::BLC_orbitIncTrig);
  iosize += DECODECTP(bytesInput,  CTF::BLC_bytesInput);
  iosize += DECODECTP(bytesClass,  CTF::BLC_bytesClass);
  // clang-format on
  //
  data.clear();
  std::map<o2::InteractionRecord, CTPDigit> digitsMap;
  o2::InteractionRecord ir(header.firstBC, header.firstOrbit);
  lumi.nHBFCounted = header.lumiNHBFs;
  lumi.nHBFCountedFV0 = header.lumiNHBFsFV0 ? header.lumiNHBFsFV0 : header.lumiNHBFs;
  lumi.counts = header.lumiCounts;
  lumi.countsFV0 = header.lumiCountsFV0;
  lumi.orbit = header.lumiOrbit;
  lumi.inp1 = int(header.inp1);
  lumi.inp2 = int(header.inp2);
  auto itInp = bytesInput.begin();
  auto itCls = bytesClass.begin();
  bool checkIROK = (mBCShift == 0); // need to check if CTP offset correction does not make the local time negative ?
  bool checkIROKInputs = (mBCShiftInputs == 0); // need to check if CTP offset correction does not make the local time negative ?
  for (uint32_t itrig = 0; itrig < header.nTriggers; itrig++) {
    // restore TrigRecord
    if (orbitInc[itrig]) {  // non-0 increment => new orbit
      ir.bc = bcInc[itrig]; // bcInc has absolute meaning
      ir.orbit += orbitInc[itrig];
    } else {
      ir.bc += bcInc[itrig];
    }
    if (checkIROKInputs || canApplyBCShiftInputs(ir)) { // correction will be ok
      auto irs = ir - mBCShiftInputs;
      uint64_t CTPInputMask = 0;
      for (int i = 0; i < CTFHelper::CTPInpNBytes; i++) {
        CTPInputMask |= static_cast<uint64_t>(*itInp++) << (8 * i);
      }
      if (CTPInputMask) {
        if (digitsMap.count(irs)) {
          if (digitsMap[irs].isInputEmpty()) {
            digitsMap[irs].CTPInputMask = CTPInputMask;
            // LOG(info) << "IR1:";
            // digitsMap[irs].printStream(std::cout);
          } else {
            LOG(error) << "CTPInpurMask already exist:" << irs << " dig.CTPInputMask:" << digitsMap[irs].CTPInputMask << " CTPInputMask:" << CTPInputMask;
          }
        } else {
          CTPDigit dig = {irs, CTPInputMask, 0};
          digitsMap[irs] = dig;
          // LOG(info) << "IR2:";
          // digitsMap[irs].printStream(std::cout);
        }
      }
    } else { // correction would make IR prior to mFirstTFOrbit, skip
      itInp += CTFHelper::CTPInpNBytes;
    }
    if (checkIROK || canApplyBCShift(ir)) { // correction will be ok
      auto irs = ir - mBCShift;
      uint64_t CTPClassMask = 0;
      for (int i = 0; i < CTFHelper::CTPClsNBytes; i++) {
        CTPClassMask |= static_cast<uint64_t>(*itCls++) << (8 * i);
      }
      if (CTPClassMask) {
        if (digitsMap.count(irs)) {
          if (digitsMap[irs].isClassEmpty()) {
            digitsMap[irs].CTPClassMask = CTPClassMask;
            // LOG(info) << "TCM1:";
            // digitsMap[irs].printStream(std::cout);
          } else {
            LOG(error) << "CTPClassMask already exist:" << irs << " dig.CTPClassMask:" << digitsMap[irs].CTPClassMask << " CTPClassMask:" << CTPClassMask;
          }
        } else {
          CTPDigit dig = {irs, 0, CTPClassMask};
          digitsMap[irs] = dig;
          // LOG(info) << "TCM2:";
          // digitsMap[irs].printStream(std::cout);
        }
      }
    } else { // correction would make IR prior to mFirstTFOrbit, skip
      itCls += CTFHelper::CTPClsNBytes;
    }
  }
  if (mDecodeInps) {
    o2::pmr::vector<CTPDigit> digits;
    o2::ctp::RawDataDecoder::shiftInputs(digitsMap, digits, mFirstTFOrbit);
    for (auto const& dig : digits) {
      data.emplace_back(dig);
    }
  } else {
    for (auto const& dig : digitsMap) {
      data.emplace_back(dig.second);
    }
  }
  assert(itInp == bytesInput.end());
  assert(itCls == bytesClass.end());
  iosize.rawIn = header.nTriggers * sizeof(CTPDigit);
  return iosize;
}
///________________________________
template <typename CTF = o2::ctp::CTF>
bool CTFCoder::finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
  auto match = o2::ctf::CTFCoderBase::finaliseCCDB<CTF>(matcher, obj);
  mBCShiftInputs = -o2::ctp::TriggerOffsetsParam::Instance().globalInputsShift;
  LOG(info) << "BCShiftInputs:" << mBCShiftInputs;
  return match;
}

} // namespace ctp
} // namespace o2

#endif // O2_CTP_CTFCODER_H
