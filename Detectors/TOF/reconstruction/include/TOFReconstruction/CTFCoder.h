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
/// \author fnoferin@cern.ch
/// \brief class for entropy encoding/decoding of TOF compressed infos data

#ifndef O2_TOF_CTFCODER_H
#define O2_TOF_CTFCODER_H

#include <algorithm>
#include <iterator>
#include <string>
#include "DataFormatsTOF/CTF.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsBase/CTFCoderBase.h"
#include "TOFBase/Digit.h"

class TTree;

namespace o2
{
namespace tof
{

class CTFCoder : public o2::ctf::CTFCoderBase
{
 public:
  CTFCoder(o2::ctf::CTFCoderBase::OpType op) : o2::ctf::CTFCoderBase(op, CTF::getNBlocks(), o2::detectors::DetID::TOF) {}
  ~CTFCoder() final = default;

  /// entropy-encode clusters to buffer with CTF
  template <typename VEC>
  o2::ctf::CTFIOSize encode(VEC& buff, const gsl::span<const ReadoutWindowData>& rofRecVec, const gsl::span<const Digit>& cdigVec, const gsl::span<const uint8_t>& pattVec);

  /// entropy decode clusters from buffer with CTF
  template <typename VROF, typename VDIG, typename VPAT>
  o2::ctf::CTFIOSize decode(const CTF::base& ec, VROF& rofRecVec, VDIG& cdigVec, VPAT& pattVec);

  void createCoders(const std::vector<char>& bufVec, o2::ctf::CTFCoderBase::OpType op) final;

 private:
  /// compres compact clusters to CompressedInfos
  void compress(CompressedInfos& cc, const gsl::span<const ReadoutWindowData>& rofRecVec, const gsl::span<const Digit>& cdigVec, const gsl::span<const uint8_t>& pattVec);
  size_t estimateCompressedSize(const CompressedInfos& cc);
  /// decompress CompressedInfos to compact clusters
  template <typename VROF, typename VDIG, typename VPAT>
  void decompress(const CompressedInfos& cc, VROF& rofRecVec, VDIG& cdigVec, VPAT& pattVec);

  void appendToTree(TTree& tree, CTF& ec);
  void readFromTree(TTree& tree, int entry, std::vector<ReadoutWindowData>& rofRecVec, std::vector<Digit>& cdigVec, std::vector<uint8_t>& pattVec);
};

///___________________________________________________________________________________
/// entropy-encode digits to buffer with CTF
template <typename VEC>
o2::ctf::CTFIOSize CTFCoder::encode(VEC& buff, const gsl::span<const ReadoutWindowData>& rofRecVec, const gsl::span<const Digit>& cdigVec, const gsl::span<const uint8_t>& pattVec)
{
  using MD = o2::ctf::Metadata::OptStore;
  // what to do which each field: see o2::ctd::Metadata explanation
  constexpr MD optField[CTF::getNBlocks()] = {
    MD::EENCODE_OR_PACK, // BLCbcIncROF
    MD::EENCODE_OR_PACK, // BLCorbitIncROF
    MD::EENCODE_OR_PACK, // BLCndigROF
    MD::EENCODE_OR_PACK, // BLCndiaROF
    MD::EENCODE_OR_PACK, // BLCndiaCrate
    MD::EENCODE_OR_PACK, // BLCtimeFrameInc
    MD::EENCODE_OR_PACK, // BLCtimeTDCInc
    MD::EENCODE_OR_PACK, // BLCstripID
    MD::EENCODE_OR_PACK, // BLCchanInStrip
    MD::EENCODE_OR_PACK, // BLCtot
    MD::EENCODE_OR_PACK, // BLCpattMap
  };
  CompressedInfos cc;
  compress(cc, rofRecVec, cdigVec, pattVec);
  // book output size with some margin
  auto szIni = estimateCompressedSize(cc);
  buff.resize(szIni);

  auto ec = CTF::create(buff);
  using ECB = CTF::base;

  ec->setHeader(cc.header);
  assignDictVersion(static_cast<o2::ctf::CTFDictHeader&>(ec->getHeader()));
  ec->setANSHeader(mANSVersion);
  // at every encoding the buffer might be autoexpanded, so we don't work with fixed pointer ec
  o2::ctf::CTFIOSize iosize;
#define ENCODETOF(part, slot, bits) CTF::get(buff.data())->encode(part, int(slot), bits, optField[int(slot)], &buff, mCoders[int(slot)], getMemMarginFactor());
  // clang-format off
  iosize += ENCODETOF(cc.bcIncROF,     CTF::BLCbcIncROF,     0);
  iosize += ENCODETOF(cc.orbitIncROF,  CTF::BLCorbitIncROF,  0);
  iosize += ENCODETOF(cc.ndigROF,      CTF::BLCndigROF,      0);
  iosize += ENCODETOF(cc.ndiaROF,      CTF::BLCndiaROF,      0);
  iosize += ENCODETOF(cc.ndiaCrate,    CTF::BLCndiaCrate,    0);
  iosize += ENCODETOF(cc.timeFrameInc, CTF::BLCtimeFrameInc, 0);
  iosize += ENCODETOF(cc.timeTDCInc,   CTF::BLCtimeTDCInc,   0);
  iosize += ENCODETOF(cc.stripID,      CTF::BLCstripID,      0);
  iosize += ENCODETOF(cc.chanInStrip,  CTF::BLCchanInStrip,  0);
  iosize += ENCODETOF(cc.tot,          CTF::BLCtot,          0);
  iosize += ENCODETOF(cc.pattMap,      CTF::BLCpattMap,      0);
  // clang-format on
  CTF::get(buff.data())->print(getPrefix(), mVerbosity);
  finaliseCTFOutput<CTF>(buff);
  iosize.rawIn = sizeof(ReadoutWindowData) * rofRecVec.size() + sizeof(Digit) * cdigVec.size() + sizeof(uint8_t) * pattVec.size();
  return iosize;
}

///___________________________________________________________________________________
/// decode entropy-encoded digits to standard compact digits
template <typename VROF, typename VDIG, typename VPAT>
o2::ctf::CTFIOSize CTFCoder::decode(const CTF::base& ec, VROF& rofRecVec, VDIG& cdigVec, VPAT& pattVec)
{
  CompressedInfos cc;
  ec.print(getPrefix(), mVerbosity);
  cc.header = ec.getHeader();
  checkDictVersion(static_cast<const o2::ctf::CTFDictHeader&>(cc.header));
  o2::ctf::CTFIOSize iosize;
#define DECODETOF(part, slot) ec.decode(part, int(slot), mCoders[int(slot)])
  // clang-format off
  iosize += DECODETOF(cc.bcIncROF,     CTF::BLCbcIncROF);
  iosize += DECODETOF(cc.orbitIncROF,  CTF::BLCorbitIncROF);
  iosize += DECODETOF(cc.ndigROF,      CTF::BLCndigROF);
  iosize += DECODETOF(cc.ndiaROF,      CTF::BLCndiaROF);
  iosize += DECODETOF(cc.ndiaCrate,    CTF::BLCndiaCrate);

  iosize += DECODETOF(cc.timeFrameInc, CTF::BLCtimeFrameInc);
  iosize += DECODETOF(cc.timeTDCInc,   CTF::BLCtimeTDCInc);
  iosize += DECODETOF(cc.stripID,      CTF::BLCstripID);
  iosize += DECODETOF(cc.chanInStrip,  CTF::BLCchanInStrip);
  iosize += DECODETOF(cc.tot,          CTF::BLCtot);
  iosize += DECODETOF(cc.pattMap,      CTF::BLCpattMap);
  // clang-format on
  //
  decompress(cc, rofRecVec, cdigVec, pattVec);
  iosize.rawIn = sizeof(ReadoutWindowData) * rofRecVec.size() + sizeof(Digit) * cdigVec.size() + sizeof(uint8_t) * pattVec.size();
  return iosize;
}
///___________________________________________________________________________________
/// decompress compressed infos to standard compact digits
template <typename VROF, typename VDIG, typename VPAT>
void CTFCoder::decompress(const CompressedInfos& cc, VROF& rofRecVec, VDIG& cdigVec, VPAT& pattVec)
{
  rofRecVec.resize(cc.header.nROFs);
  cdigVec.resize(cc.header.nDigits);
  pattVec.resize(cc.header.nPatternBytes);
  std::vector<Digit> digCopy;

  o2::InteractionRecord prevIR(cc.header.firstBC, cc.header.firstOrbit);
  uint32_t firstEntry = 0, digCount = 0, stripCount = 0, ndiagnostic = 0;
  for (uint32_t irof = 0; irof < cc.header.nROFs; irof++) {
    // restore ROFRecord
    auto& rofRec = rofRecVec[irof];
    if (cc.orbitIncROF[irof]) {      // new orbit
      prevIR.bc = cc.bcIncROF[irof]; // bcInc has absolute meaning
      prevIR.orbit += cc.orbitIncROF[irof];
    } else {
      prevIR.bc += cc.bcIncROF[irof];
    }
    rofRec.setBCData(prevIR);
    rofRec.setFirstEntry(firstEntry);
    rofRec.setNEntries(cc.ndigROF[irof]);
    rofRec.setFirstEntryDia(ndiagnostic);
    rofRec.setNEntriesDia(cc.ndiaROF[irof]);
    for (int icrate = 0; icrate < 72; icrate++) {
      rofRec.setDiagnosticInCrate(icrate, cc.ndiaCrate[irof * 72 + icrate] - 1); // -1 because number were traslated since (-1 means crate not available)
    }
    firstEntry += cc.ndigROF[irof];
    ndiagnostic += cc.ndiaROF[irof];

    if (!cc.ndigROF[irof]) {
      continue;
    }

    // restore hit data
    uint ctimeframe = 0;
    uint ctdc = 0;

    int firstDig = digCount;

    int64_t BCrow = prevIR.toLong();

    digCopy.resize(cc.ndigROF[irof]);
    for (uint32_t idig = 0; idig < cc.ndigROF[irof]; idig++) {
      auto& digit = digCopy[idig]; //cdigVec[digCount];
      LOGF(debug, "%d) TF=%d, TDC=%d, STRIP=%d, CH=%d", idig, cc.timeFrameInc[digCount], cc.timeTDCInc[digCount], cc.stripID[digCount], cc.chanInStrip[digCount]);
      if (cc.timeFrameInc[digCount]) { // new time frame
        ctdc = cc.timeTDCInc[digCount];
        ctimeframe += cc.timeFrameInc[digCount];
      } else {
        ctdc += cc.timeTDCInc[digCount];
      }
      LOGF(debug, "BC=%ld, TDC=%d, TOT=%d, CH=%d", uint32_t(ctimeframe) * 64 + ctdc / 1024 + BCrow, ctdc % 1024, cc.tot[digCount], uint32_t(cc.stripID[digCount]) * 96 + cc.chanInStrip[digCount]);

      digit.setBC(uint32_t(ctimeframe) * 64 + ctdc / 1024 + BCrow);
      digit.setTDC(ctdc % 1024);
      digit.setTOT(cc.tot[digCount]);
      digit.setChannel(uint32_t(cc.stripID[digCount]) * 96 + cc.chanInStrip[digCount]);

      digCount++;
    }

    // sort digits according to strip number within the ROF
    std::sort(digCopy.begin(), digCopy.end(),
              [](o2::tof::Digit a, o2::tof::Digit b) {
                int str1 = a.getChannel() / Geo::NPADS;
                int str2 = b.getChannel() / Geo::NPADS;
                if (str1 == str2) {
                  return (a.getOrderingKey() < b.getOrderingKey());
                }
                return (str1 < str2);
              });

    // fill digits, once sorted, of rof in digit vector
    for (uint32_t idig = 0; idig < digCopy.size(); idig++) {
      cdigVec[firstDig + idig] = digCopy[idig];
    }

    digCopy.clear();
  }
  // explicit patterns
  memcpy(pattVec.data(), cc.pattMap.data(), cc.header.nPatternBytes); // RSTODO use swap?
  assert(digCount == cc.header.nDigits);

  if (digCount != cc.header.nDigits) {
    LOG(error) << "expected " << cc.header.nDigits << " but counted " << digCount << " in ROFRecords";
    throw std::runtime_error("mismatch between expected and counter number of digits");
  }
}

} // namespace tof
} // namespace o2

#endif // O2_TOF_CTFCODER_H
