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
/// \author fnoferin@cern.ch
/// \brief class for entropy encoding/decoding of TOF compressed infos data

#ifndef O2_TOF_CTFCODER_H
#define O2_TOF_CTFCODER_H

#include <algorithm>
#include <iterator>
#include <string>
#include "DataFormatsTOF/CTF.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "rANS/rans.h"
#include "TOFBase/Digit.h"

class TTree;

namespace o2
{
namespace tof
{

class CTFCoder
{
 public:
  /// entropy-encode clusters to buffer with CTF
  template <typename VEC>
  static void encode(VEC& buff, const gsl::span<const ReadoutWindowData>& rofRecVec, const gsl::span<const Digit>& cdigVec, const gsl::span<const uint32_t>& pattVec);

  /// entropy decode clusters from buffer with CTF
  template <typename VROF, typename VDIG, typename VPAT>
  static void decode(const CTF::base& ec, VROF& rofRecVec, VDIG& cdigVec, VPAT& pattVec);

 private:
  /// compres compact clusters to CompressedInfos
  static void compress(CompressedInfos& cc, const gsl::span<const ReadoutWindowData>& rofRecVec, const gsl::span<const Digit>& cdigVec, const gsl::span<const uint32_t>& pattVec);

  /// decompress CompressedInfos to compact clusters
  template <typename VROF, typename VDIG, typename VPAT>
  static void decompress(const CompressedInfos& cc, VROF& rofRecVec, VDIG& cdigVec, VPAT& pattVec);

  static void appendToTree(TTree& tree, o2::detectors::DetID id, CTF& ec);
  static void readFromTree(TTree& tree, int entry, o2::detectors::DetID id, std::vector<ReadoutWindowData>& rofRecVec, std::vector<Digit>& cdigVec, std::vector<uint32_t>& pattVec);

 protected:
  ClassDefNV(CTFCoder, 1);
};

///___________________________________________________________________________________
/// entropy-encode digits to buffer with CTF
template <typename VEC>
void CTFCoder::encode(VEC& buff, const gsl::span<const ReadoutWindowData>& rofRecVec, const gsl::span<const Digit>& cdigVec, const gsl::span<const uint32_t>& pattVec)
{
  using MD = o2::ctf::Metadata::OptStore;
  // what to do which each field: see o2::ctd::Metadata explanation
  constexpr MD optField[CTF::getNBlocks()] = {
    MD::EENCODE, //BLCbcIncROF
    MD::EENCODE, //BLCorbitIncROF
    MD::EENCODE, //BLCndigROF
    MD::EENCODE, //BLCndiaROF
    MD::EENCODE, //BLCtimeFrameInc
    MD::EENCODE, //BLCtimeTDCInc
    MD::EENCODE, //BLCstripID
    MD::EENCODE, //BLCchanInStrip
    MD::EENCODE, //BLCtot
    MD::EENCODE, //BLCpattMap
  };
  CompressedInfos cc;
  compress(cc, rofRecVec, cdigVec, pattVec);
  auto ec = CTF::create(buff);
  using ECB = CTF::base;

  ec->setHeader(cc.header);
  ec->getANSHeader().majorVersion = 0;
  ec->getANSHeader().minorVersion = 1;
  // at every encoding the buffer might be autoexpanded, so we don't work with fixed pointer ec
#define ENCODE CTF::get(buff.data())->encode
  // clang-format off
  ENCODE(cc.bcIncROF, CTF::BLCbcIncROF, o2::rans::ProbabilityBits16Bit, optField[CTF::BLCbcIncROF], &buff);
  ENCODE(cc.orbitIncROF, CTF::BLCorbitIncROF, o2::rans::ProbabilityBits16Bit, optField[CTF::BLCorbitIncROF], &buff);
  ENCODE(cc.ndigROF, CTF::BLCndigROF, o2::rans::ProbabilityBits16Bit, optField[CTF::BLCndigROF], &buff);
  ENCODE(cc.ndiaROF, CTF::BLCndiaROF, o2::rans::ProbabilityBits16Bit, optField[CTF::BLCndiaROF], &buff);
  ENCODE(cc.timeFrameInc, CTF::BLCtimeFrameInc, o2::rans::ProbabilityBits16Bit, optField[CTF::BLCtimeFrameInc], &buff);
  ENCODE(cc.timeTDCInc, CTF::BLCtimeTDCInc, o2::rans::ProbabilityBits16Bit, optField[CTF::BLCtimeTDCInc], &buff);
  ENCODE(cc.stripID, CTF::BLCstripID, o2::rans::ProbabilityBits16Bit, optField[CTF::BLCstripID], &buff);
  ENCODE(cc.chanInStrip, CTF::BLCchanInStrip, o2::rans::ProbabilityBits16Bit, optField[CTF::BLCchanInStrip], &buff);
  ENCODE(cc.tot, CTF::BLCtot, o2::rans::ProbabilityBits16Bit, optField[CTF::BLCtot], &buff);
  ENCODE(cc.pattMap,      CTF::BLCpattMap,      o2::rans::ProbabilityBits16Bit, optField[CTF::BLCpattMap], &buff);
  // clang-format on
}
///___________________________________________________________________________________
/// decode entropy-encoded digits to standard compact digits
template <typename VROF, typename VDIG, typename VPAT>
void CTFCoder::decode(const CTF::base& ec, VROF& rofRecVec, VDIG& cdigVec, VPAT& pattVec)
{
  CompressedInfos cc;
  cc.header = ec.getHeader();
  // clang-format off
    ec.decode(cc.bcIncROF, CTF::BLCbcIncROF);
    ec.decode(cc.orbitIncROF, CTF::BLCorbitIncROF);
    ec.decode(cc.ndigROF, CTF::BLCndigROF);
    ec.decode(cc.ndiaROF, CTF::BLCndiaROF);

    ec.decode(cc.timeFrameInc, CTF::BLCtimeFrameInc);
    ec.decode(cc.timeTDCInc, CTF::BLCtimeTDCInc);
    ec.decode(cc.stripID, CTF::BLCstripID);
    ec.decode(cc.chanInStrip, CTF::BLCchanInStrip);
    ec.decode(cc.tot, CTF::BLCtot);
    ec.decode(cc.pattMap,      CTF::BLCpattMap);
  // clang-format on
  //
  decompress(cc, rofRecVec, cdigVec, pattVec);
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
    firstEntry += cc.ndigROF[irof];
    ndiagnostic += cc.ndiaROF[irof];

    if (!cc.ndigROF[irof])
      continue;

    // restore hit data
    uint ctimeframe = 0;
    uint ctdc = 0;

    int firstDig = digCount;

    int BCrow = prevIR.orbit * Geo::BC_IN_ORBIT + prevIR.bc;

    digCopy.resize(cc.ndigROF[irof]);
    for (uint32_t idig = 0; idig < cc.ndigROF[irof]; idig++) {
      auto& digit = digCopy[idig]; //cdigVec[digCount];
      printf("%d) TF=%d, TDC=%d, STRIP=%d, CH=%d\n", idig, cc.timeFrameInc[digCount], cc.timeTDCInc[digCount], cc.stripID[digCount], cc.chanInStrip[digCount]);
      if (cc.timeFrameInc[digCount]) { // new time frame
        ctdc = cc.timeTDCInc[digCount];
        ctimeframe += cc.timeFrameInc[digCount];
      } else {
        ctdc += cc.timeTDCInc[digCount];
      }
      printf("BC=%d, TDC=%d, TOT=%d, CH=%d \n", uint32_t(ctimeframe) * 64 + ctdc / 1024 + BCrow, ctdc % 1024, cc.tot[digCount], uint32_t(cc.stripID[digCount]) * 96 + cc.chanInStrip[digCount]);

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
    LOG(ERROR) << "expected " << cc.header.nDigits << " but counted " << digCount << " in ROFRecords";
    throw std::runtime_error("mismatch between expected and counter number of digits");
  }
}

} // namespace tof
} // namespace o2

#endif // O2_TOF_CTFCODER_H
