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
  CTFCoder() : o2::ctf::CTFCoderBase(CTF::getNBlocks(), o2::detectors::DetID::TOF) {}
  ~CTFCoder() = default;

  /// entropy-encode clusters to buffer with CTF
  template <typename VEC>
  void encode(VEC& buff, const gsl::span<const ReadoutWindowData>& rofRecVec, const gsl::span<const Digit>& cdigVec, const gsl::span<const uint32_t>& pattVec);

  /// entropy decode clusters from buffer with CTF
  template <typename VROF, typename VDIG, typename VPAT>
  void decode(const CTF::base& ec, VROF& rofRecVec, VDIG& cdigVec, VPAT& pattVec);

  void createCoders(const std::string& dictPath, o2::ctf::CTFCoderBase::OpType op);

 private:
  /// compres compact clusters to CompressedInfos
  void compress(CompressedInfos& cc, const gsl::span<const ReadoutWindowData>& rofRecVec, const gsl::span<const Digit>& cdigVec, const gsl::span<const uint32_t>& pattVec);
  size_t estimateCompressedSize(const CompressedInfos& cc);
  /// decompress CompressedInfos to compact clusters
  template <typename VROF, typename VDIG, typename VPAT>
  void decompress(const CompressedInfos& cc, VROF& rofRecVec, VDIG& cdigVec, VPAT& pattVec);

  void appendToTree(TTree& tree, CTF& ec);
  void readFromTree(TTree& tree, int entry, std::vector<ReadoutWindowData>& rofRecVec, std::vector<Digit>& cdigVec, std::vector<uint32_t>& pattVec);

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
    MD::EENCODE, //BLCndiaCrate
    MD::EENCODE, //BLCtimeFrameInc
    MD::EENCODE, //BLCtimeTDCInc
    MD::EENCODE, //BLCstripID
    MD::EENCODE, //BLCchanInStrip
    MD::EENCODE, //BLCtot
    MD::EENCODE, //BLCpattMap
  };
  CompressedInfos cc;
  compress(cc, rofRecVec, cdigVec, pattVec);
  // book output size with some margin
  auto szIni = estimateCompressedSize(cc);
  buff.resize(szIni);

  auto ec = CTF::create(buff);
  using ECB = CTF::base;

  ec->setHeader(cc.header);
  ec->getANSHeader().majorVersion = 0;
  ec->getANSHeader().minorVersion = 1;
  // at every encoding the buffer might be autoexpanded, so we don't work with fixed pointer ec
#define ENCODETOF(part, slot, bits) CTF::get(buff.data())->encode(part, int(slot), bits, optField[int(slot)], &buff, mCoders[int(slot)].get());
  // clang-format off
  ENCODETOF(cc.bcIncROF,     CTF::BLCbcIncROF,     0);
  ENCODETOF(cc.orbitIncROF,  CTF::BLCorbitIncROF,  0);
  ENCODETOF(cc.ndigROF,      CTF::BLCndigROF,      0);
  ENCODETOF(cc.ndiaROF,      CTF::BLCndiaROF,      0);
  ENCODETOF(cc.ndiaCrate,    CTF::BLCndiaCrate,    0);
  ENCODETOF(cc.timeFrameInc, CTF::BLCtimeFrameInc, 0);
  ENCODETOF(cc.timeTDCInc,   CTF::BLCtimeTDCInc,   0);
  ENCODETOF(cc.stripID,      CTF::BLCstripID,      0);
  ENCODETOF(cc.chanInStrip,  CTF::BLCchanInStrip,  0);
  ENCODETOF(cc.tot,          CTF::BLCtot,          0);
  ENCODETOF(cc.pattMap,      CTF::BLCpattMap,      0);
  // clang-format on
  CTF::get(buff.data())->print(getPrefix());
}

///___________________________________________________________________________________
/// decode entropy-encoded digits to standard compact digits
template <typename VROF, typename VDIG, typename VPAT>
void CTFCoder::decode(const CTF::base& ec, VROF& rofRecVec, VDIG& cdigVec, VPAT& pattVec)
{
  CompressedInfos cc;
  ec.print(getPrefix());
  cc.header = ec.getHeader();
#define DECODETOF(part, slot) ec.decode(part, int(slot), mCoders[int(slot)].get())
  // clang-format off
  DECODETOF(cc.bcIncROF,     CTF::BLCbcIncROF);
  DECODETOF(cc.orbitIncROF,  CTF::BLCorbitIncROF);
  DECODETOF(cc.ndigROF,      CTF::BLCndigROF);
  DECODETOF(cc.ndiaROF,      CTF::BLCndiaROF);
  DECODETOF(cc.ndiaCrate,    CTF::BLCndiaCrate);
  
  DECODETOF(cc.timeFrameInc, CTF::BLCtimeFrameInc);
  DECODETOF(cc.timeTDCInc,   CTF::BLCtimeTDCInc);
  DECODETOF(cc.stripID,      CTF::BLCstripID);
  DECODETOF(cc.chanInStrip,  CTF::BLCchanInStrip);
  DECODETOF(cc.tot,          CTF::BLCtot);
  DECODETOF(cc.pattMap,      CTF::BLCpattMap);
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
    for (int icrate = 0; icrate < 72; icrate++) {
      rofRec.setDiagnosticInCrate(icrate, cc.ndiaCrate[irof * 72 + icrate]);
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

    int BCrow = prevIR.orbit * Geo::BC_IN_ORBIT + prevIR.bc;

    digCopy.resize(cc.ndigROF[irof]);
    for (uint32_t idig = 0; idig < cc.ndigROF[irof]; idig++) {
      auto& digit = digCopy[idig]; //cdigVec[digCount];
      LOGF(DEBUG, "%d) TF=%d, TDC=%d, STRIP=%d, CH=%d", idig, cc.timeFrameInc[digCount], cc.timeTDCInc[digCount], cc.stripID[digCount], cc.chanInStrip[digCount]);
      if (cc.timeFrameInc[digCount]) { // new time frame
        ctdc = cc.timeTDCInc[digCount];
        ctimeframe += cc.timeFrameInc[digCount];
      } else {
        ctdc += cc.timeTDCInc[digCount];
      }
      LOGF(DEBUG, "BC=%d, TDC=%d, TOT=%d, CH=%d", uint32_t(ctimeframe) * 64 + ctdc / 1024 + BCrow, ctdc % 1024, cc.tot[digCount], uint32_t(cc.stripID[digCount]) * 96 + cc.chanInStrip[digCount]);

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
