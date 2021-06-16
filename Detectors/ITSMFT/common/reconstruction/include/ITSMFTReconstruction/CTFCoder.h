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
#include "DetectorsBase/CTFCoderBase.h"
#include "rANS/rans.h"

class TTree;

namespace o2
{
namespace itsmft
{

class CTFCoder : public o2::ctf::CTFCoderBase
{
 public:
  CTFCoder(o2::detectors::DetID det) : o2::ctf::CTFCoderBase(CTF::getNBlocks(), det) {}
  ~CTFCoder() = default;

  /// entropy-encode clusters to buffer with CTF
  template <typename VEC>
  void encode(VEC& buff, const gsl::span<const ROFRecord>& rofRecVec, const gsl::span<const CompClusterExt>& cclusVec, const gsl::span<const unsigned char>& pattVec);

  /// entropy decode clusters from buffer with CTF
  template <typename VROF, typename VCLUS, typename VPAT>
  void decode(const CTF::base& ec, VROF& rofRecVec, VCLUS& cclusVec, VPAT& pattVec);

  void createCoders(const std::string& dictPath, o2::ctf::CTFCoderBase::OpType op);

 private:
  /// compres compact clusters to CompressedClusters
  void compress(CompressedClusters& cc, const gsl::span<const ROFRecord>& rofRecVec, const gsl::span<const CompClusterExt>& cclusVec, const gsl::span<const unsigned char>& pattVec);
  size_t estimateCompressedSize(const CompressedClusters& cc);

  /// decompress CompressedClusters to compact clusters
  template <typename VROF, typename VCLUS, typename VPAT>
  void decompress(const CompressedClusters& cc, VROF& rofRecVec, VCLUS& cclusVec, VPAT& pattVec);

  void appendToTree(TTree& tree, CTF& ec);
  void readFromTree(TTree& tree, int entry, std::vector<ROFRecord>& rofRecVec, std::vector<CompClusterExt>& cclusVec, std::vector<unsigned char>& pattVec);

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
  // book output size with some margin
  auto szIni = estimateCompressedSize(cc);
  buff.resize(szIni);

  auto ec = CTF::create(buff);
  using ECB = CTF::base;

  ec->setHeader(cc.header);
  ec->getANSHeader().majorVersion = 0;
  ec->getANSHeader().minorVersion = 1;
  // at every encoding the buffer might be autoexpanded, so we don't work with fixed pointer ec
#define ENCODEITSMFT(part, slot, bits) CTF::get(buff.data())->encode(part, int(slot), bits, optField[int(slot)], &buff, mCoders[int(slot)].get());
  // clang-format off
  ENCODEITSMFT(cc.firstChipROF, CTF::BLCfirstChipROF, 0);
  ENCODEITSMFT(cc.bcIncROF, CTF::BLCbcIncROF, 0);
  ENCODEITSMFT(cc.orbitIncROF, CTF::BLCorbitIncROF, 0);
  ENCODEITSMFT(cc.nclusROF, CTF::BLCnclusROF, 0);
  //
  ENCODEITSMFT(cc.chipInc, CTF::BLCchipInc, 0);
  ENCODEITSMFT(cc.chipMul, CTF::BLCchipMul, 0);
  ENCODEITSMFT(cc.row, CTF::BLCrow, 0);
  ENCODEITSMFT(cc.colInc, CTF::BLCcolInc, 0);
  ENCODEITSMFT(cc.pattID, CTF::BLCpattID, 0);
  ENCODEITSMFT(cc.pattMap, CTF::BLCpattMap, 0);
  // clang-format on
  CTF::get(buff.data())->print(getPrefix());
}

/// decode entropy-encoded clusters to standard compact clusters
template <typename VROF, typename VCLUS, typename VPAT>
void CTFCoder::decode(const CTF::base& ec, VROF& rofRecVec, VCLUS& cclusVec, VPAT& pattVec)
{
  CompressedClusters cc;
  cc.header = ec.getHeader();
  ec.print(getPrefix());
#define DECODEITSMFT(part, slot) ec.decode(part, int(slot), mCoders[int(slot)].get())
  // clang-format off
  DECODEITSMFT(cc.firstChipROF, CTF::BLCfirstChipROF);
  DECODEITSMFT(cc.bcIncROF,     CTF::BLCbcIncROF);
  DECODEITSMFT(cc.orbitIncROF,  CTF::BLCorbitIncROF);
  DECODEITSMFT(cc.nclusROF,     CTF::BLCnclusROF);
  //    
  DECODEITSMFT(cc.chipInc,      CTF::BLCchipInc);
  DECODEITSMFT(cc.chipMul,      CTF::BLCchipMul);
  DECODEITSMFT(cc.row,          CTF::BLCrow);
  DECODEITSMFT(cc.colInc,       CTF::BLCcolInc);
  DECODEITSMFT(cc.pattID,       CTF::BLCpattID);
  DECODEITSMFT(cc.pattMap,      CTF::BLCpattMap);
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
