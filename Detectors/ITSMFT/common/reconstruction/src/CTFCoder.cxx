// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   CTFCoder.cxx
/// \author ruben.shahoyan@cern.ch
/// \brief class for entropy encoding/decoding of ITS/MFT compressmed clusters data

#include "ITSMFTReconstruction/CTFCoder.h"
#include "CommonUtils/StringUtils.h"
#include <TTree.h>

using namespace o2::itsmft;

///___________________________________________________________________________________
// Register encoded data in the tree (Fill is not called, will be done by caller)
void CTFCoder::appendToTree(TTree& tree, o2::detectors::DetID id, CTF& ec)
{
  ec.appendToTree(tree, id.getName());
}

///___________________________________________________________________________________
// extract and decode data from the tree
void CTFCoder::readFromTree(TTree& tree, int entry, o2::detectors::DetID id,
                            std::vector<ROFRecord>& rofRecVec, std::vector<CompClusterExt>& cclusVec, std::vector<unsigned char>& pattVec)
{
  assert(entry >= 0 && entry < tree.GetEntries());
  CTF ec;
  ec.readFromTree(tree, id.getName(), entry);
  decode(ec, rofRecVec, cclusVec, pattVec);
}

///________________________________
void CTFCoder::compress(CompressedClusters& cc,
                        const gsl::span<const ROFRecord>& rofRecVec,
                        const gsl::span<const CompClusterExt>& cclusVec,
                        const gsl::span<const unsigned char>& pattVec)
{
  // store in the header the orbit of 1st ROF
  cc.clear();
  if (!rofRecVec.size()) {
    return;
  }
  const auto& rofRec0 = rofRecVec[0];
  cc.header.nROFs = rofRecVec.size();
  cc.header.firstOrbit = rofRec0.getBCData().orbit;
  cc.header.firstBC = rofRec0.getBCData().bc;
  cc.header.nPatternBytes = pattVec.size();
  cc.header.nClusters = cclusVec.size();
  cc.header.nChips = 0; // this is the version with chipInc stored once per new chip

  cc.firstChipROF.resize(cc.header.nROFs);
  cc.bcIncROF.resize(cc.header.nROFs);
  cc.orbitIncROF.resize(cc.header.nROFs);
  cc.nclusROF.resize(cc.header.nROFs);
  //
  cc.row.resize(cc.header.nClusters);
  cc.colInc.resize(cc.header.nClusters);
  //  cc.chipInc.resize(cc.header.nClusters); // this is the version with chipInc stored for every pixel
  cc.chipInc.reserve(1000); // this is the version with chipInc stored once per new chip
  cc.chipMul.reserve(1000); // this is the version with chipInc stored once per new chip
  cc.pattID.resize(cc.header.nClusters);
  cc.pattMap.resize(cc.header.nPatternBytes);

  uint16_t prevBC = cc.header.firstBC;
  uint32_t prevOrbit = cc.header.firstOrbit;

  for (uint32_t irof = 0; irof < cc.header.nROFs; irof++) {
    const auto& rofRec = rofRecVec[irof];
    const auto& intRec = rofRec.getBCData();
    if (intRec.orbit == prevOrbit) {
      cc.orbitIncROF[irof] = 0;
      cc.bcIncROF[irof] = intRec.bc - prevBC; // store increment of BC if in the same orbit
    } else {
      cc.orbitIncROF[irof] = intRec.orbit - prevOrbit;
      cc.bcIncROF[irof] = intRec.bc; // otherwise, store absolute bc
      prevOrbit = intRec.orbit;
    }
    prevBC = intRec.bc;
    auto ncl = rofRec.getNEntries();
    cc.nclusROF[irof] = ncl;
    if (!ncl) { // no hits data for this ROF
      cc.firstChipROF[irof] = 0;
      continue;
    }
    cc.firstChipROF[irof] = cclusVec[rofRec.getFirstEntry()].getChipID();
    int icl = rofRec.getFirstEntry(), iclMax = icl + ncl;

    uint16_t prevChip = cc.firstChipROF[irof], prevCol = 0;
    if (icl != iclMax) { // there still clusters to store
      cc.chipMul.push_back(0);
      cc.chipInc.push_back(0);
    }
    for (; icl < iclMax; icl++) { // clusters within a chip are stored in increasing column number
      const auto& cl = cclusVec[icl];
      cc.row[icl] = cl.getRow(); // row is practically random, store it as it is
      cc.pattID[icl] = cl.getPatternID();
      if (cl.getChipID() == prevChip) { // for the same chip store column increment
        // cc.chipInc[icl] = 0;  // this is the version with chipInc stored for every pixel
        cc.chipMul.back()++; // this is the version with chipInc stored once per new chip
        cc.colInc[icl] = cl.getCol() - prevCol;
        prevCol = cl.getCol();
      } else { // for new chips store chipID increment and abs. column
        // cc.chipInc[icl] = cl.getChipID() - prevChip;  // this is the version with chipInc stored for every pixel
        cc.chipInc.push_back(cl.getChipID() - prevChip); // this is the version with chipInc stored once per new chip
        cc.chipMul.push_back(1);                         // this is the version with chipInc stored once per new chip
        prevCol = cc.colInc[icl] = cl.getCol();
        prevChip = cl.getChipID();
      }
    }
  }
  cc.header.nChips = cc.chipMul.size();
  // store explicit patters as they are
  memcpy(cc.pattMap.data(), pattVec.data(), cc.header.nPatternBytes); // RSTODO: do we need this?
}

///________________________________
void CTFCoder::createCoders(const std::string& dictPath, o2::detectors::DetID det, o2::ctf::CTFCoderBase::OpType op)
{
  bool mayFail = true; // RS FIXME if the dictionary file is not there, do not produce exception
  auto buff = readDictionaryFromFile<CTF>(dictPath, det, mayFail);
  if (!buff.size()) {
    if (mayFail) {
      return;
    }
    throw std::runtime_error("Failed to create CTF dictionaty");
  }
  const auto* ctf = CTF::get(buff.data());

  auto getFreq = [ctf](CTF::Slots slot) -> o2::rans::FrequencyTable {
    o2::rans::FrequencyTable ft;
    auto bl = ctf->getBlock(slot);
    auto md = ctf->getMetadata(slot);
    ft.addFrequencies(bl.getDict(), bl.getDict() + bl.getNDict(), md.min, md.max);
    return std::move(ft);
  };
  auto getProbBits = [ctf](CTF::Slots slot) -> int {
    return ctf->getMetadata(slot).probabilityBits;
  };

  CompressedClusters cc; // just to get member types
#define MAKECODER(part, slot) createCoder<decltype(part)::value_type>(op, getFreq(slot), getProbBits(slot), int(slot))
  // clang-format off
  MAKECODER(cc.firstChipROF, CTF::BLCfirstChipROF);
  MAKECODER(cc.bcIncROF,     CTF::BLCbcIncROF    );
  MAKECODER(cc.orbitIncROF,  CTF::BLCorbitIncROF );
  MAKECODER(cc.nclusROF,     CTF::BLCnclusROF    );
  //
  MAKECODER(cc.chipInc,      CTF::BLCchipInc     );
  MAKECODER(cc.chipMul,      CTF::BLCchipMul     );
  MAKECODER(cc.row,          CTF::BLCrow         );
  MAKECODER(cc.colInc,       CTF::BLCcolInc      );
  MAKECODER(cc.pattID,       CTF::BLCpattID      );
  MAKECODER(cc.pattMap,      CTF::BLCpattMap     );
  // clang-format on
}
