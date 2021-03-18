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
/// \brief class for entropy encoding/decoding of MCH digit data

#ifndef O2_MCH_CTFCODER_H
#define O2_MCH_CTFCODER_H

#include <algorithm>
#include <iterator>
#include <string>
#include <array>
#include "DataFormatsMCH/CTF.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "MCHBase/Digit.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsBase/CTFCoderBase.h"
#include "MCHCTF/CTFHelper.h"
#include "rANS/rans.h"

class TTree;

namespace o2
{
namespace mch
{

class CTFCoder : public o2::ctf::CTFCoderBase
{
 public:
  CTFCoder() : o2::ctf::CTFCoderBase(CTF::getNBlocks(), o2::detectors::DetID::MCH) {}
  ~CTFCoder() = default;

  /// entropy-encode data to buffer with CTF
  template <typename VEC>
  void encode(VEC& buff, const gsl::span<const ROFRecord>& rofData, const gsl::span<const Digit>& digData);

  /// entropy decode data from buffer with CTF
  template <typename VROF, typename VCOL>
  void decode(const CTF::base& ec, VROF& rofVec, VCOL& digVec);

  void createCoders(const std::string& dictPath, o2::ctf::CTFCoderBase::OpType op);

 private:
  void appendToTree(TTree& tree, CTF& ec);
  void readFromTree(TTree& tree, int entry, std::vector<ROFRecord>& rofVec, std::vector<Digit>& digVec);
};

/// entropy-encode clusters to buffer with CTF
template <typename VEC>
void CTFCoder::encode(VEC& buff, const gsl::span<const ROFRecord>& rofData, const gsl::span<const Digit>& digData)
{
  using MD = o2::ctf::Metadata::OptStore;
  // what to do which each field: see o2::ctd::Metadata explanation
  constexpr MD optField[CTF::getNBlocks()] = {
    MD::EENCODE, // BLC_bcIncROF
    MD::EENCODE, // BLC_orbitIncROF
    MD::EENCODE, // BLC_nDigitsROF
    MD::EENCODE, // BLC_tfTime
    MD::EENCODE, // BLC_nSamples
    MD::EENCODE, // BLC_detID
    MD::EENCODE, // BLC_padID
    MD::EENCODE  // BLC_ADC
  };
  CTFHelper helper(rofData, digData);

  // book output size with some margin
  auto szIni = sizeof(CTFHeader) + helper.getSize() * 2. / 3; // will be autoexpanded if needed
  buff.resize(szIni);

  auto ec = CTF::create(buff);
  using ECB = CTF::base;

  ec->setHeader(helper.createHeader());
  ec->getANSHeader().majorVersion = 0;
  ec->getANSHeader().minorVersion = 1;
  // at every encoding the buffer might be autoexpanded, so we don't work with fixed pointer ec
#define ENCODEMCH(beg, end, slot, bits) CTF::get(buff.data())->encode(beg, end, int(slot), bits, optField[int(slot)], &buff, mCoders[int(slot)].get());
  // clang-format off
  ENCODEMCH(helper.begin_bcIncROF(),    helper.end_bcIncROF(),     CTF::BLC_bcIncROF,    0);
  ENCODEMCH(helper.begin_orbitIncROF(), helper.end_orbitIncROF(),  CTF::BLC_orbitIncROF, 0);
  ENCODEMCH(helper.begin_nDigitsROF(),  helper.end_nDigitsROF(),   CTF::BLC_nDigitsROF,  0);

  ENCODEMCH(helper.begin_tfTime(),      helper.end_tfTime(),       CTF::BLC_tfTime,      0);
  ENCODEMCH(helper.begin_nSamples(),    helper.end_nSamples(),     CTF::BLC_nSamples,    0);
  ENCODEMCH(helper.begin_detID(),       helper.end_detID(),        CTF::BLC_detID,       0);
  ENCODEMCH(helper.begin_padID(),       helper.end_padID(),        CTF::BLC_padID,       0);
  ENCODEMCH(helper.begin_ADC()  ,       helper.end_ADC(),          CTF::BLC_ADC,         0);
  // clang-format on
  CTF::get(buff.data())->print(getPrefix());
}

/// decode entropy-encoded clusters to standard compact clusters
template <typename VROF, typename VCOL>
void CTFCoder::decode(const CTF::base& ec, VROF& rofVec, VCOL& digVec)
{
  auto header = ec.getHeader();
  ec.print(getPrefix());
  std::vector<uint16_t> bcInc, nDigits, nSamples;
  std::vector<uint32_t> orbitInc, ADC;
  std::vector<int32_t> tfTime;
  std::vector<int16_t> detID, padID;

#define DECODEMCH(part, slot) ec.decode(part, int(slot), mCoders[int(slot)].get())
  // clang-format off
  DECODEMCH(bcInc,       CTF::BLC_bcIncROF);
  DECODEMCH(orbitInc,    CTF::BLC_orbitIncROF);
  DECODEMCH(nDigits,     CTF::BLC_nDigitsROF);

  DECODEMCH(tfTime,      CTF::BLC_tfTime);
  DECODEMCH(nSamples,    CTF::BLC_nSamples);
  DECODEMCH(detID,       CTF::BLC_detID);
  DECODEMCH(padID,       CTF::BLC_padID);
  DECODEMCH(ADC,         CTF::BLC_ADC);
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
    for (uint8_t ic = 0; ic < nDigits[irof]; ic++) {
      digVec.emplace_back(Digit{detID[digCount], padID[digCount], ADC[digCount], tfTime[digCount], nSamples[digCount]});
      digCount++;
    }
    rofVec.emplace_back(ROFRecord{ir, int(firstEntry), nDigits[irof]});
  }
  assert(digCount == header.nDigits);
}

} // namespace mch
} // namespace o2

#endif // O2_MCH_CTFCODER_H
