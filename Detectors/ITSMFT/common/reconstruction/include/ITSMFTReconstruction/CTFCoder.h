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
#include "DataFormatsITSMFT/Digit.h"
#include "DataFormatsITSMFT/NoiseMap.h"
#include "ITSMFTReconstruction/LookUp.h"
#include "ITSMFTReconstruction/PixelData.h"
#include "ITSMFTReconstruction/Clusterer.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsBase/CTFCoderBase.h"
#include "rANS/rans.h"

//#define _CHECK_INCREMENTES_ // Uncoment this the check the incremements being non-negative

class TTree;

namespace o2
{
namespace itsmft
{

class CTFCoder : public o2::ctf::CTFCoderBase
{
 public:
  using PMatrix = std::array<std::array<bool, ClusterPattern::MaxRowSpan + 2>, ClusterPattern::MaxColSpan + 2>;
  using RowColBuff = std::vector<PixelData>;

  CTFCoder(o2::ctf::CTFCoderBase::OpType op, o2::detectors::DetID det) : o2::ctf::CTFCoderBase(op, CTF::getNBlocks(), det) {}
  ~CTFCoder() final = default;

  /// entropy-encode clusters to buffer with CTF
  template <typename VEC>
  o2::ctf::CTFIOSize encode(VEC& buff, const gsl::span<const ROFRecord>& rofRecVec, const gsl::span<const CompClusterExt>& cclusVec, const gsl::span<const unsigned char>& pattVec);

  /// entropy decode clusters from buffer with CTF
  template <typename VROF, typename VCLUS, typename VPAT>
  o2::ctf::CTFIOSize decode(const CTF::base& ec, VROF& rofRecVec, VCLUS& cclusVec, VPAT& pattVec, const NoiseMap* noiseMap, const LookUp& clPattLookup);

  /// entropy decode digits from buffer with CTF
  template <typename VROF, typename VDIG>
  o2::ctf::CTFIOSize decode(const CTF::base& ec, VROF& rofRecVec, VDIG& digVec, const NoiseMap* noiseMap, const LookUp& clPattLookup);

  void createCoders(const std::vector<char>& bufVec, o2::ctf::CTFCoderBase::OpType op) final;

 private:
  CompressedClusters decodeCompressedClusters(const CTF::base& ec, o2::ctf::CTFIOSize& sz);

  /// compres compact clusters to CompressedClusters
  void compress(CompressedClusters& compCl, const gsl::span<const ROFRecord>& rofRecVec, const gsl::span<const CompClusterExt>& cclusVec, const gsl::span<const unsigned char>& pattVec);
  size_t estimateCompressedSize(const CompressedClusters& compCl);

  /// decompress CompressedClusters to compact clusters
  template <typename VROF, typename VCLUS, typename VPAT>
  void decompress(const CompressedClusters& compCl, VROF& rofRecVec, VCLUS& cclusVec, VPAT& pattVec, const NoiseMap* noiseMap, const LookUp& clPattLookup);

  /// decompress CompressedClusters to digits
  template <typename VROF, typename VDIG>
  void decompress(const CompressedClusters& compCl, VROF& rofRecVec, VDIG& digVec, const NoiseMap* noiseMap, const LookUp& clPattLookup);

  void appendToTree(TTree& tree, CTF& ec);
  void readFromTree(TTree& tree, int entry, std::vector<ROFRecord>& rofRecVec, std::vector<CompClusterExt>& cclusVec, std::vector<unsigned char>& pattVec, const NoiseMap* noiseMap, const LookUp& clPattLookup);
};

/// entropy-encode clusters to buffer with CTF
template <typename VEC>
o2::ctf::CTFIOSize CTFCoder::encode(VEC& buff, const gsl::span<const ROFRecord>& rofRecVec, const gsl::span<const CompClusterExt>& cclusVec, const gsl::span<const unsigned char>& pattVec)
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
  CompressedClusters compCl;
  compress(compCl, rofRecVec, cclusVec, pattVec);
  // book output size with some margin
  auto szIni = estimateCompressedSize(compCl);
  buff.resize(szIni);

  auto ec = CTF::create(buff);
  using ECB = CTF::base;

  ec->setHeader(compCl.header);
  assignDictVersion(static_cast<o2::ctf::CTFDictHeader&>(ec->getHeader()));
  ec->getANSHeader().majorVersion = 0;
  ec->getANSHeader().minorVersion = 1;
  // at every encoding the buffer might be autoexpanded, so we don't work with fixed pointer ec
  o2::ctf::CTFIOSize iosize;
#define ENCODEITSMFT(part, slot, bits) CTF::get(buff.data())->encode(part, int(slot), bits, optField[int(slot)], &buff, mCoders[int(slot)].get(), getMemMarginFactor());
  // clang-format off
  iosize += ENCODEITSMFT(compCl.firstChipROF, CTF::BLCfirstChipROF, 0);
  iosize += ENCODEITSMFT(compCl.bcIncROF, CTF::BLCbcIncROF, 0);
  iosize += ENCODEITSMFT(compCl.orbitIncROF, CTF::BLCorbitIncROF, 0);
  iosize += ENCODEITSMFT(compCl.nclusROF, CTF::BLCnclusROF, 0);
  //
  iosize += ENCODEITSMFT(compCl.chipInc, CTF::BLCchipInc, 0);
  iosize += ENCODEITSMFT(compCl.chipMul, CTF::BLCchipMul, 0);
  iosize += ENCODEITSMFT(compCl.row, CTF::BLCrow, 0);
  iosize += ENCODEITSMFT(compCl.colInc, CTF::BLCcolInc, 0);
  iosize += ENCODEITSMFT(compCl.pattID, CTF::BLCpattID, 0);
  iosize += ENCODEITSMFT(compCl.pattMap, CTF::BLCpattMap, 0);
  // clang-format on
  //CTF::get(buff.data())->print(getPrefix());
  iosize.rawIn = rofRecVec.size() * sizeof(ROFRecord) + cclusVec.size() * sizeof(CompClusterExt) + pattVec.size() * sizeof(unsigned char);
  return iosize;
}

/// decode entropy-encoded clusters to standard compact clusters
template <typename VROF, typename VCLUS, typename VPAT>
o2::ctf::CTFIOSize CTFCoder::decode(const CTF::base& ec, VROF& rofRecVec, VCLUS& cclusVec, VPAT& pattVec, const NoiseMap* noiseMap, const LookUp& clPattLookup)
{
  o2::ctf::CTFIOSize iosize;
  auto compCl = decodeCompressedClusters(ec, iosize);
  decompress(compCl, rofRecVec, cclusVec, pattVec, noiseMap, clPattLookup);
  iosize.rawIn = rofRecVec.size() * sizeof(ROFRecord) + cclusVec.size() * sizeof(CompClusterExt) + pattVec.size() * sizeof(unsigned char);
  return iosize;
}

/// decode entropy-encoded clusters to digits
template <typename VROF, typename VDIG>
o2::ctf::CTFIOSize CTFCoder::decode(const CTF::base& ec, VROF& rofRecVec, VDIG& digVec, const NoiseMap* noiseMap, const LookUp& clPattLookup)
{
  o2::ctf::CTFIOSize iosize;
  auto compCl = decodeCompressedClusters(ec, iosize);
  decompress(compCl, rofRecVec, digVec, noiseMap, clPattLookup);
  iosize.rawIn += rofRecVec.size() * sizeof(ROFRecord) + digVec.size() * sizeof(Digit);
  return iosize;
}

/// decompress compressed clusters to standard compact clusters
template <typename VROF, typename VCLUS, typename VPAT>
void CTFCoder::decompress(const CompressedClusters& compCl, VROF& rofRecVec, VCLUS& cclusVec, VPAT& pattVec, const NoiseMap* noiseMap, const LookUp& clPattLookup)
{
  PMatrix pmat{};
  RowColBuff firedPixBuff{}, maskedPixBuff{};
  rofRecVec.resize(compCl.header.nROFs);
  cclusVec.clear();
  cclusVec.reserve(compCl.header.nClusters);
  pattVec.clear();
  pattVec.reserve(compCl.header.nPatternBytes);
  o2::InteractionRecord prevIR(compCl.header.firstBC, compCl.header.firstOrbit);
  uint32_t clCount = 0, chipCount = 0;
  auto pattIt = compCl.pattMap.begin();
  auto pattItStored = pattIt;

  // >> ====== Helper functions for reclusterization after masking some pixels in decoded clusters ======
  // clusterize the pmat matrix holding pixels of the single cluster after masking the noisy ones

  auto clusterize = [&](uint16_t chipID, int16_t row, int16_t col, int leftFired) {
#ifdef _ALLOW_DIAGONAL_ALPIDE_CLUSTERS_
    const std::array<int16_t, 8> walkRow = {1, -1, 0, 0, 1, 1, -1, -1};
    const std::array<int16_t, 8> walkCol = {0, 0, -1, 1, 1, -1, 1, 1};
#else
    const std::array<int16_t, 4> walkRow = {1, -1, 0, 0};
    const std::array<int16_t, 4> walkCol = {0, 0, -1, 1};
#endif
    Clusterer::BBox bbox(chipID);
    // check and add to new cluster seed fired pixels around ir1, ic1, return true if there are still fired pixels left
    std::function<bool(int16_t, int16_t)> checkPixelAndNeighbours = [&](int16_t ir1, int16_t ic1) {
      // if pixel in pmat is fired, add it to new cluster seed and adjust the BBox, decreasing the number of fired pixels left
      auto checkPixel = [&](int16_t ir1, int16_t ic1) {
        if (pmat[ir1][ic1]) {
          pmat[ir1][ic1] = false;
          uint16_t r = row + ir1 - 1, c = col + ic1 - 1;
          firedPixBuff.emplace_back(r, c);
          bbox.adjust(r, c);
          leftFired--;
          return true;
        }
        return false;
      };
      // check and add to new cluster seed fired pixels at and around ir1, ic1, return true if there are still fired pixels left
      if (checkPixel(ir1, ic1) && leftFired) {
        uint16_t iw = 0;
        while (checkPixelAndNeighbours(ir1 + walkRow[iw], ic1 + walkCol[iw]) && ++iw < walkRow.size()) {
        }
      }
      return leftFired;
    };
    // true will be returned if after incremental check of neighbours fired pixels are still left

    firedPixBuff.clear();          // start new cluster seed
    for (auto s : maskedPixBuff) { // we start checking from the holes remaining from the masked pixels
      uint16_t iw = 0;
      do {
        checkPixelAndNeighbours(s.getRowDirect() + walkRow[iw], s.getCol() + walkCol[iw]);
        if (!firedPixBuff.empty()) {
          bbox.chipID = chipID;
          Clusterer::streamCluster(firedPixBuff, nullptr, bbox, clPattLookup, &cclusVec, &pattVec, nullptr, 0);
          firedPixBuff.clear();
          bbox.clear();
        }
      } while (leftFired && ++iw < walkRow.size());
      if (!leftFired) {
        break;
      }
    }
  };

  auto reclusterize = [&]() {
    auto clus = cclusVec.back(); // original newly added cluster
    // acquire pattern
    o2::itsmft::ClusterPattern patt;
    auto pattItPrev = pattIt;
    maskedPixBuff.clear();
    int rowRef = clus.getRow(), colRef = clus.getCol();
    if (clPattLookup.size() == 0 && clus.getPatternID() != o2::itsmft::CompCluster::InvalidPatternID) {
      throw std::runtime_error("Clusters contain pattern IDs, but no dictionary is provided...");
    }
    if (clus.getPatternID() == o2::itsmft::CompCluster::InvalidPatternID) {
      patt.acquirePattern(pattIt);
    } else if (clPattLookup.isGroup(clus.getPatternID())) {
      patt.acquirePattern(pattIt);
      float xCOG = 0, zCOG = 0;
      patt.getCOG(xCOG, zCOG); // for grouped patterns the reference pixel is at COG
      rowRef -= round(xCOG);
      colRef -= round(zCOG);
    } else {
      patt = clPattLookup.getPattern(clus.getPatternID());
    }
    int rowSpan = patt.getRowSpan(), colSpan = patt.getColumnSpan(), nMasked = 0;
    if (rowSpan == 1 && colSpan == 1) {                                        // easy case: 1 pixel cluster
      if (noiseMap->isNoisy(clus.getChipID(), rowRef, colRef)) {               // just kill the cluster
        std::copy(pattItStored, pattItPrev, back_inserter(pattVec));           // save patterns from after last saved to the one before killing this
        pattItStored = pattIt;                                                 // advance to the head of the pattern iterator
        cclusVec.pop_back();
      }
      // otherwise do nothing: cluster was already added, eventual patterns will be copied in large block at next modified cluster writing
    } else {
      int rowSpan = patt.getRowSpan(), colSpan = patt.getColumnSpan(), nMasked = 0, nPixels = 0; // apply noise and fill hits matrix
      for (int ir = 0; ir < rowSpan; ir++) {
        int row = rowRef + ir;
        for (int ic = 0; ic < colSpan; ic++) {
          if (patt.isSet(ir, ic)) {
            if (noiseMap->isNoisy(clus.getChipID(), row, colRef + ic)) {
              maskedPixBuff.emplace_back(ir + 1, ic + 1);
              pmat[ir + 1][ic + 1] = false; // reset since might be left from prev cluster
              nMasked++;
            } else {
              pmat[ir + 1][ic + 1] = true;
              nPixels++;
            }
          } else {
            pmat[ir + 1][ic + 1] = false; // reset since might be left from prev cluster
          }
        }
      }

      if (nMasked) {
        cclusVec.pop_back();                                         // remove added cluster
        std::copy(pattItStored, pattItPrev, back_inserter(pattVec)); // save patterns from after last saved to the one before killing this
        pattItStored = pattIt;                                       // advance to the head of the pattern iterator
        if (nPixels) {                                               // need to reclusterize remaining pixels
          clusterize(clus.getChipID(), rowRef, colRef, nPixels);
        }
      }
    }
  };
  // << ====== Helper functions for reclusterization after masking some pixels in decoded clusters ======

  for (uint32_t irof = 0; irof < compCl.header.nROFs; irof++) {
    // restore ROFRecord
    auto& rofRec = rofRecVec[irof];
    if (compCl.orbitIncROF[irof]) {      // new orbit
      prevIR.bc = compCl.bcIncROF[irof]; // bcInc has absolute meaning
      prevIR.orbit += compCl.orbitIncROF[irof];
    } else {
      prevIR.bc += compCl.bcIncROF[irof];
    }
    rofRec.setBCData(prevIR);
    rofRec.setFirstEntry(cclusVec.size());

    // resrore chips data
    auto chipID = compCl.firstChipROF[irof];
    uint16_t col = 0;
    int inChip = 0;
    for (uint32_t icl = 0; icl < compCl.nclusROF[irof]; icl++) {
      auto& clus = cclusVec.emplace_back();
      if (inChip++ < compCl.chipMul[chipCount]) { // still the same chip
        clus.setCol((col += compCl.colInc[clCount]));
      } else { // new chip starts
        chipID += compCl.chipInc[++chipCount];
        inChip = 1;
        clus.setCol((col = compCl.colInc[clCount])); // colInc has abs. col meaning
      }
      clus.setRow(compCl.row[clCount]);
      clus.setPatternID(compCl.pattID[clCount]);
      clus.setChipID(chipID);
      if (noiseMap) { // noise masking was requested
        reclusterize();
      }
      clCount++;
    }
    if (compCl.nclusROF[irof]) {
      chipCount++; // since next chip for sure will be new and inChip will be 0...
    }
    rofRec.setNEntries(cclusVec.size() - rofRec.getFirstEntry());
  }
  if (noiseMap) {                 // reclusterization was requested
    if (pattItStored != pattIt) { // copy unsaved patterns
      std::copy(pattItStored, pattIt, back_inserter(pattVec));
    }
  } else { // copy decoded patterns as they are
    pattVec.resize(compCl.header.nPatternBytes);
    memcpy(pattVec.data(), compCl.pattMap.data(), compCl.header.nPatternBytes);
  }
  assert(chipCount == compCl.header.nChips);

  if (clCount != compCl.header.nClusters) {
    LOG(error) << "expected " << compCl.header.nClusters << " but counted " << clCount << " in ROFRecords";
    throw std::runtime_error("mismatch between expected and counter number of clusters");
  }
}

/// decompress compressed clusters to digits
template <typename VROF, typename VDIG>
void CTFCoder::decompress(const CompressedClusters& compCl, VROF& rofRecVec, VDIG& digVec, const NoiseMap* noiseMap, const LookUp& clPattLookup)
{
  rofRecVec.resize(compCl.header.nROFs);
  digVec.reserve(compCl.header.nClusters * 2);
  o2::InteractionRecord prevIR(compCl.header.firstBC, compCl.header.firstOrbit);
  uint32_t clCount = 0, chipCount = 0;
  auto pattIt = compCl.pattMap.begin();
  o2::itsmft::ClusterPattern patt;
  for (uint32_t irof = 0; irof < compCl.header.nROFs; irof++) {
    size_t chipStartNDig = digVec.size();
    // restore ROFRecord
    auto& rofRec = rofRecVec[irof];
    if (compCl.orbitIncROF[irof]) {      // new orbit
      prevIR.bc = compCl.bcIncROF[irof]; // bcInc has absolute meaning
      prevIR.orbit += compCl.orbitIncROF[irof];
    } else {
      prevIR.bc += compCl.bcIncROF[irof];
    }
    rofRec.setBCData(prevIR);
    rofRec.setFirstEntry(digVec.size());

    // resrore chips data
    uint16_t chipID = compCl.firstChipROF[irof], col = 0;
    int inChip = 0;
    for (uint32_t icl = 0; icl < compCl.nclusROF[irof]; icl++) {
      if (inChip++ < compCl.chipMul[chipCount]) { // still the same chip
        col += compCl.colInc[clCount];
      } else { // new chip starts
        // sort digits of previous chip in col/row
        auto added = digVec.size() - chipStartNDig;
        if (added > 1) { // we need to sort digits in colums and in rows within a column
          std::sort(digVec.end() - added, digVec.end(),
                    [](Digit& a, Digit& b) { return a.getColumn() < b.getColumn() || (a.getColumn() == b.getColumn() && a.getRow() < b.getRow()); });
        }
        chipStartNDig = digVec.size();
        chipID += compCl.chipInc[++chipCount];
#ifdef _CHECK_INCREMENTES_
        if (int16_t(compCl.chipInc[chipCount]) < 0) {
          LOG(warning) << "Negative chip increment " << int16_t(compCl.chipInc[chipCount]) << " -> " << chipID;
        }
#endif
        inChip = 1;
        col = compCl.colInc[clCount]; // colInc has abs. col meaning
      }
      uint16_t rowRef = compCl.row[clCount], colRef = col;
      auto pattID = compCl.pattID[clCount];
      if (pattID == o2::itsmft::CompCluster::InvalidPatternID) {
        patt.acquirePattern(pattIt);
      } else {
        if (clPattLookup.size() == 0) {
          throw std::runtime_error("Clusters contain pattern IDs, but no dictionary is provided...");
        }
        if (pattID == o2::itsmft::CompCluster::InvalidPatternID) {
          patt.acquirePattern(pattIt);
        } else if (clPattLookup.isGroup(pattID)) {
          patt.acquirePattern(pattIt);
          float xCOG = 0., zCOG = 0.;
          patt.getCOG(xCOG, zCOG); // for grouped patterns the reference pixel is at COG
          rowRef -= round(xCOG);
          colRef -= round(zCOG);
        } else {
          patt = clPattLookup.getPattern(pattID);
        }
      }
      clCount++;

      auto fillRowCol = [&digVec, chipID, rowRef, colRef, noiseMap](int r, int c) {
        r += rowRef;
        c += colRef;
        if (noiseMap && noiseMap->isNoisy(chipID, r, c)) {
          return;
        }
        digVec.emplace_back(chipID, uint16_t(r), uint16_t(c));
      };
      patt.process(fillRowCol);
    }
    auto added = digVec.size() - chipStartNDig;
    if (added > 1) { // Last chip of the ROF: we need to sort digits in colums and in rows within a column
      std::sort(digVec.end() - added, digVec.end(),
                [](Digit& a, Digit& b) { return a.getColumn() < b.getColumn() || (a.getColumn() == b.getColumn() && a.getRow() < b.getRow()); });
    }

    if (compCl.nclusROF[irof]) {
      chipCount++; // since next chip for sure will be new and incChip will be 0...
    }
    rofRec.setNEntries(digVec.size() - rofRec.getFirstEntry());
  }
  // explicit patterns
  assert(pattIt == compCl.pattMap.end());
  assert(chipCount == compCl.header.nChips);

  if (clCount != compCl.header.nClusters) {
    LOG(error) << "expected " << compCl.header.nClusters << " but counted " << clCount << " in ROFRecords";
    throw std::runtime_error("mismatch between expected and counter number of clusters");
  }
}

} // namespace itsmft
} // namespace o2

#endif // O2_ITSMFT_CTFCODER_H
