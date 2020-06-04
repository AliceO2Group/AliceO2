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
/// \brief class for entropy encoding/decoding of ITS/MFT compressed clusters data

#ifndef O2_ITSMFT_CTFCODER_H
#define O2_ITSMFT_CTFCODER_H

#include <algorithm>
#include <iterator>
#include <string>
#include "DataFormatsITSMFT/CTF.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "rANS/rans.h"

class TTree;

namespace o2
{
namespace itsmft
{

class CTFCoder
{
 public:
  /// entropy-encode clusters to buffer with CTF
  template <typename VEC>
  static void encode(VEC& buff, const gsl::span<const ROFRecord>& rofRecVec, const gsl::span<const CompClusterExt>& cclusVec, const gsl::span<const unsigned char>& pattVec);

  /// entropy decode clusters from buffer with CTF
  template <typename VROF, typename VCLUS, typename VPAT>
  static void decode(const CTF::base& ec, VROF& rofRecVec, VCLUS& cclusVec, VPAT& pattVec);

 private:
  /// compres compact clusters to CompressedClusters
  static void compress(CompressedClusters& cc, const gsl::span<const ROFRecord>& rofRecVec, const gsl::span<const CompClusterExt>& cclusVec, const gsl::span<const unsigned char>& pattVec);

  /// decompress CompressedClusters to compact clusters
  template <typename VROF, typename VCLUS, typename VPAT>
  static void decompress(const CompressedClusters& cc, VROF& rofRecVec, VCLUS& cclusVec, VPAT& pattVec);

  static void appendToTree(TTree& tree, o2::detectors::DetID id, CTF& ec);
  static void readFromTree(TTree& tree, int entry, o2::detectors::DetID id, std::vector<ROFRecord>& rofRecVec, std::vector<CompClusterExt>& cclusVec, std::vector<unsigned char>& pattVec);

 protected:
  ClassDefNV(CTFCoder, 1);
};

/// entropy-encode clusters to buffer with CTF
template <typename VEC>
void CTFCoder::encode(VEC& buff, const gsl::span<const ROFRecord>& rofRecVec, const gsl::span<const CompClusterExt>& cclusVec, const gsl::span<const unsigned char>& pattVec)
{
  using MD = o2::ctf::Metadata::OptStore;
  // what to do which each field: see o2::ctd::Metadata explanation
  constexpr MD optField[CTF::getNBlocks()] = {
    MD::EENCODE, //BLCfirstChipROF
    MD::EENCODE, //BLCbcIncROF
    MD::EENCODE, //BLCorbitIncROF
    MD::EENCODE, //BLCnclusROF
    MD::EENCODE, //BLCchipInc
    MD::EENCODE, //BLCchipMul
    MD::EENCODE, //BLCrow
    MD::EENCODE, //BLCcolInc
    MD::EENCODE, //BLCpattID
    MD::EENCODE  //BLCpattMap
  };
  CompressedClusters cc;
  compress(cc, rofRecVec, cclusVec, pattVec);
  auto ec = CTF::create(buff);
  using ECB = CTF::base;

  ec->setHeader(cc.header);
  ec->getANSHeader().majorVersion = 0;
  ec->getANSHeader().minorVersion = 1;
  // at every encoding the buffer might be autoexpanded, so we don't work with fixed pointer ec
#define ENCODE CTF::get(buff.data())->encode
  // clang-format off
  ENCODE(cc.firstChipROF, CTF::BLCfirstChipROF, o2::rans::ProbabilityBits16Bit, optField[CTF::BLCfirstChipROF], &buff);
  ENCODE(cc.bcIncROF,     CTF::BLCbcIncROF ,    o2::rans::ProbabilityBits16Bit, optField[CTF::BLCbcIncROF],     &buff);
  ENCODE(cc.orbitIncROF,  CTF::BLCorbitIncROF,  o2::rans::ProbabilityBits16Bit, optField[CTF::BLCorbitIncROF],  &buff);
  ENCODE(cc.nclusROF,     CTF::BLCnclusROF,     o2::rans::ProbabilityBits16Bit, optField[CTF::BLCnclusROF],     &buff);
  //
  ENCODE(cc.chipInc,      CTF::BLCchipInc,      o2::rans::ProbabilityBits16Bit, optField[CTF::BLCchipInc], &buff);
  ENCODE(cc.chipMul,      CTF::BLCchipMul,      o2::rans::ProbabilityBits16Bit, optField[CTF::BLCchipMul], &buff);
  ENCODE(cc.row,          CTF::BLCrow,          o2::rans::ProbabilityBits16Bit, optField[CTF::BLCrow],     &buff);
  ENCODE(cc.colInc,       CTF::BLCcolInc,       o2::rans::ProbabilityBits16Bit, optField[CTF::BLCcolInc],  &buff);
  ENCODE(cc.pattID,       CTF::BLCpattID,       o2::rans::ProbabilityBits16Bit, optField[CTF::BLCpattID],  &buff);
  ENCODE(cc.pattMap,      CTF::BLCpattMap,      o2::rans::ProbabilityBits16Bit, optField[CTF::BLCpattMap], &buff);
  // clang-format on
}

/// decode entropy-encoded clusters to standard compact clusters
template <typename VROF, typename VCLUS, typename VPAT>
void CTFCoder::decode(const CTF::base& ec, VROF& rofRecVec, VCLUS& cclusVec, VPAT& pattVec)
{
  CompressedClusters cc;
  cc.header = ec.getHeader();
  // clang-format off
    ec.decode(cc.firstChipROF, CTF::BLCfirstChipROF);
    ec.decode(cc.bcIncROF,     CTF::BLCbcIncROF);
    ec.decode(cc.orbitIncROF,  CTF::BLCorbitIncROF);
    ec.decode(cc.nclusROF,     CTF::BLCnclusROF);
    //    
    ec.decode(cc.chipInc,      CTF::BLCchipInc);
    ec.decode(cc.chipMul,      CTF::BLCchipMul);
    ec.decode(cc.row,          CTF::BLCrow);
    ec.decode(cc.colInc,       CTF::BLCcolInc);
    ec.decode(cc.pattID,       CTF::BLCpattID);
    ec.decode(cc.pattMap,      CTF::BLCpattMap);
  // clang-format on
  //
  decompress(cc, rofRecVec, cclusVec, pattVec);
}

/// decompress compressed clusters to standard compact clusters
template <typename VROF, typename VCLUS, typename VPAT>
void CTFCoder::decompress(const CompressedClusters& cc, VROF& rofRecVec, VCLUS& cclusVec, VPAT& pattVec)
{
  rofRecVec.resize(cc.header.nROFs);
  cclusVec.resize(cc.header.nClusters);
  pattVec.resize(cc.header.nPatternBytes);

  o2::InteractionRecord prevIR(cc.header.firstBC, cc.header.firstOrbit);
  uint32_t firstEntry = 0, clCount = 0, chipCount = 0;
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
    rofRec.setNEntries(cc.nclusROF[irof]);
    firstEntry += cc.nclusROF[irof];

    // resrore chips data
    auto prevChip = cc.firstChipROF[irof];
    uint16_t prevCol = 0, prevRow = 0;

    // >> this is the version with chipInc stored once per new chip
    int inChip = 0;
    for (uint32_t icl = 0; icl < cc.nclusROF[irof]; icl++) {
      auto& clus = cclusVec[clCount];
      if (inChip++ < cc.chipMul[chipCount]) { // still the same chip
        clus.setCol((prevCol += cc.colInc[clCount]));
      } else { // new chip starts
        prevChip += cc.chipInc[++chipCount];
        inChip = 1;
        clus.setCol((prevCol = cc.colInc[clCount])); // colInc has abs. col meaning
      }
      clus.setChipID(prevChip);
      clus.setRow(cc.row[clCount]);
      clus.setPatternID(cc.pattID[clCount]);
      clCount++;
    }
    if (cc.nclusROF[irof]) {
      chipCount++; // since next chip for sure will be new and inChip will be 0...
    }
    // << this is the version with chipInc stored once per new chip

    /* 
    // >> this is the version with chipInc stored for every pixel, requires entropy compression dealing with repeating symbols
    for (uint32_t icl = 0; icl < cc.nclusROF[irof]; icl++) {
    auto& clus = cclusVec[clCount];
    if (cc.chipInc[clCount]) { // new chip
    prevChip += cc.chipInc[clCount];
    clus.setCol((prevCol = cc.colInc[clCount])); // colInc has abs. col meaning
    } else {
    clus.setCol((prevCol += cc.colInc[clCount]));
    }
    clus.setChipID(prevChip);
    clus.setRow(cc.row[clCount]);
    clus.setPatternID(cc.pattID[clCount]);
    clCount++;
    }
    // << this is the version with chipInc stored for every pixel, requires entropy compression dealing with repeating symbols
    */
  }
  // explicit patterns
  memcpy(pattVec.data(), cc.pattMap.data(), cc.header.nPatternBytes); // RSTODO use swap?
  assert(chipCount == cc.header.nChips);

  if (clCount != cc.header.nClusters) {
    LOG(ERROR) << "expected " << cc.header.nClusters << " but counted " << clCount << " in ROFRecords";
    throw std::runtime_error("mismatch between expected and counter number of clusters");
  }
}

} // namespace itsmft
} // namespace o2

#endif // O2_ITSMFT_CTFCODER_H
