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

/// \file   CTFCoder.cxx
/// \author ruben.shahoyan@cern.ch
/// \brief class for entropy encoding/decoding of ITS/MFT compressmed clusters data

#include "ITSMFTReconstruction/CTFCoder.h"
#include "CommonUtils/StringUtils.h"
#include <TTree.h>

using namespace o2::itsmft;

///___________________________________________________________________________________
// Register encoded data in the tree (Fill is not called, will be done by caller)
void CTFCoder::appendToTree(TTree& tree, CTF& ec)
{
  ec.appendToTree(tree, mDet.getName());
}

///___________________________________________________________________________________
// extract and decode data from the tree
void CTFCoder::readFromTree(TTree& tree, int entry, std::vector<ROFRecord>& rofRecVec,
                            std::vector<CompClusterExt>& cclusVec, std::vector<unsigned char>& pattVec, const NoiseMap* noiseMap, const LookUp& clPattLookup)
{
  assert(entry >= 0 && entry < tree.GetEntries());
  CTF ec;
  ec.readFromTree(tree, mDet.getName(), entry);
  decode(ec, rofRecVec, cclusVec, pattVec, noiseMap, clPattLookup);
}

///________________________________
void CTFCoder::compress(CompressedClusters& cc,
                        const gsl::span<const ROFRecord>& rofRecVec,
                        const gsl::span<const CompClusterExt>& cclusVec,
                        const gsl::span<const unsigned char>& pattVec,
                        const LookUp& clPattLookup, int strobeLength)
{
  // store in the header the orbit of 1st ROF
  cc.clear();
  cc.header.det = mDet;
  if (!rofRecVec.size()) {
    return;
  }

  uint32_t firstROF = rofRecVec.size(), nrofSel = rofRecVec.size(), nClusSel = cclusVec.size();
  std::vector<bool> reject(rofRecVec.size());
  if (mIRFrameSelector.isSet()) {
    for (size_t ir = 0; ir < rofRecVec.size(); ir++) {
      auto irStart = rofRecVec[ir].getBCData();
      if (mIRFrameSelector.check({irStart, irStart + strobeLength - 1}) < 0) {
        reject[ir] = true;
        nrofSel--;
        nClusSel -= rofRecVec[ir].getNEntries();
      } else if (firstROF == rofRecVec.size()) {
        firstROF = ir;
      }
    }
  } else {
    firstROF = 0;
  }
  if (nrofSel == 0) { // nothing is selected
    return;
  }
  assert(nClusSel <= cclusVec.size());

  const auto& rofRec0 = rofRecVec[firstROF];
  cc.header.firstOrbit = rofRec0.getBCData().orbit;
  cc.header.firstBC = rofRec0.getBCData().bc;
  cc.header.nROFs = nrofSel;
  cc.header.nClusters = nClusSel;

  cc.firstChipROF.resize(nrofSel);
  cc.bcIncROF.resize(nrofSel);
  cc.orbitIncROF.resize(nrofSel);
  cc.nclusROF.resize(nrofSel);
  //
  cc.row.resize(nClusSel);
  cc.colInc.resize(nClusSel);
  //  cc.chipInc.resize(cc.header.nClusters); // this is the version with chipInc stored for every pixel
  cc.chipInc.reserve(1000); // this is the version with chipInc stored once per new chip
  cc.chipMul.reserve(1000); // this is the version with chipInc stored once per new chip
  cc.pattID.resize(nClusSel);

  bool selectPatterns = nrofSel < rofRecVec.size();
  if (!selectPatterns) { // nothing is rejected, simply copy the patterns
    cc.header.nPatternBytes = pattVec.size();
    cc.pattMap.resize(pattVec.size()); // to be resized in case of selection
    memcpy(cc.pattMap.data(), pattVec.data(), pattVec.size());
  } else {
    cc.pattMap.reserve(pattVec.size());
  }

  uint16_t prevBC = cc.header.firstBC;
  uint32_t prevOrbit = cc.header.firstOrbit;
  int irofOut = 0, iclOut = 0;
  auto pattIt = pattVec.begin(), pattIt0 = pattVec.begin();

  for (uint32_t irof = 0; irof < rofRecVec.size(); irof++) {
    const auto& rofRec = rofRecVec[irof];
    const auto& intRec = rofRec.getBCData();

    if (reject[irof]) {                          // need to skip some patterns
      if (selectPatterns && pattIt != pattIt0) { // copy what was already selected
        cc.pattMap.insert(cc.pattMap.end(), pattIt0, pattIt);
        pattIt0 = pattIt;
      }
      for (int icl = rofRec.getFirstEntry(); icl < rofRec.getFirstEntry() + rofRec.getNEntries(); icl++) {
        const auto& clus = cclusVec[icl];
        if (clus.getPatternID() == o2::itsmft::CompCluster::InvalidPatternID || clPattLookup.isGroup(clus.getPatternID())) {
          o2::itsmft::ClusterPattern::skipPattern(pattIt);
        }
      }
      continue;
    }
    if (intRec.orbit == prevOrbit) {
      cc.orbitIncROF[irofOut] = 0;
#ifdef _CHECK_INCREMENTES_
      if (intRec.bc < prevBC) {
        LOG(warning) << "Negative BC increment " << intRec.bc << " -> " << prevBC;
      }
#endif
      cc.bcIncROF[irofOut] = intRec.bc - prevBC; // store increment of BC if in the same orbit
    } else {
      cc.orbitIncROF[irofOut] = intRec.orbit - prevOrbit;
#ifdef _CHECK_INCREMENTES_
      if (intRec.orbit < prevOrbit) {
        LOG(warning) << "Negative Orbit increment " << intRec.orbit << " -> " << prevOrbit;
      }
#endif
      cc.bcIncROF[irofOut] = intRec.bc; // otherwise, store absolute bc
      prevOrbit = intRec.orbit;
    }
    prevBC = intRec.bc;
    auto ncl = rofRec.getNEntries();
    cc.nclusROF[irofOut] = ncl;
    if (!ncl) { // no hits data for this ROF
      cc.firstChipROF[irofOut] = 0;
      irofOut++;
      continue;
    }
    cc.firstChipROF[irofOut] = cclusVec[rofRec.getFirstEntry()].getChipID();
    int icl = rofRec.getFirstEntry(), iclMax = icl + ncl;

    uint16_t prevChip = cc.firstChipROF[irofOut], prevCol = 0;
    if (icl != iclMax) { // there are still clusters to store
      cc.chipMul.push_back(0);
      cc.chipInc.push_back(0);
    }
    for (; icl < iclMax; icl++) { // clusters within a chip are stored in increasing column number
      const auto& cl = cclusVec[icl];
      cc.row[iclOut] = cl.getRow(); // row is practically random, store it as it is
      cc.pattID[iclOut] = cl.getPatternID();
      if (cl.getChipID() == prevChip) { // for the same chip store column increment
        // cc.chipInc[iclOut] = 0;  // this is the version with chipInc stored for every pixel
        cc.chipMul.back()++; // this is the version with chipInc stored once per new chip
        cc.colInc[iclOut] = int16_t(cl.getCol()) - prevCol;
        prevCol = cl.getCol();
      } else { // for new chips store chipID increment and abs. column
        // cc.chipInc[iclOut] = cl.getChipID() - prevChip;  // this is the version with chipInc stored for every pixel
        cc.chipInc.push_back(cl.getChipID() - prevChip); // this is the version with chipInc stored once per new chip
#ifdef _CHECK_INCREMENTES_
        if (cl.getChipID() < prevChip) {
          LOG(warning) << "Negative Chip increment " << cl.getChipID() << " -> " << prevChip;
        }
#endif
        cc.chipMul.push_back(1);                         // this is the version with chipInc stored once per new chip
        prevCol = cc.colInc[iclOut] = cl.getCol();
        prevChip = cl.getChipID();
      }
      iclOut++;
    }
    irofOut++;
  }
  if (selectPatterns && pattIt != pattIt0) { // copy leftover patterns
    cc.pattMap.insert(cc.pattMap.end(), pattIt0, pattIt);
    pattIt0 = pattIt;
  }
  cc.header.nPatternBytes = cc.pattMap.size();
  cc.header.nChips = cc.chipMul.size();
}

///________________________________
void CTFCoder::createCoders(const std::vector<char>& bufVec, o2::ctf::CTFCoderBase::OpType op)
{
  const auto ctf = CTF::getImage(bufVec.data());
  CompressedClusters cc; // just to get member types
#define MAKECODER(part, slot) createCoder<decltype(part)::value_type>(op, ctf.getFrequencyTable(slot), int(slot))
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

///________________________________
size_t CTFCoder::estimateCompressedSize(const CompressedClusters& cc)
{
  size_t sz = 0;
  // clang-format off
  // RS FIXME this is very crude estimate, instead, an empirical values should be used
#define VTP(vec) typename std::remove_reference<decltype(vec)>::type::value_type
#define ESTSIZE(vec, slot) mCoders[int(slot)] ?                         \
  rans::calculateMaxBufferSize(vec.size(), reinterpret_cast<const o2::rans::LiteralEncoder64<VTP(vec)>*>(mCoders[int(slot)].get())->getAlphabetRangeBits(), sizeof(VTP(vec)) ) : vec.size()*sizeof(VTP(vec))
  sz += ESTSIZE(cc.firstChipROF, CTF::BLCfirstChipROF);
  sz += ESTSIZE(cc.bcIncROF,     CTF::BLCbcIncROF    );
  sz += ESTSIZE(cc.orbitIncROF,  CTF::BLCorbitIncROF );
  sz += ESTSIZE(cc.nclusROF,     CTF::BLCnclusROF    );
  //
  sz += ESTSIZE(cc.chipInc,      CTF::BLCchipInc     );
  sz += ESTSIZE(cc.chipMul,      CTF::BLCchipMul     );
  sz += ESTSIZE(cc.row,          CTF::BLCrow         );
  sz += ESTSIZE(cc.colInc,       CTF::BLCcolInc      );
  sz += ESTSIZE(cc.pattID,       CTF::BLCpattID      );
  sz += ESTSIZE(cc.pattMap,      CTF::BLCpattMap     );

  // clang-format on
  sz *= 2. / 3; // if needed, will be autoexpanded
  LOG(info) << "Estimated output size is " << sz << " bytes";
  return sz;
}

///________________________________
CompressedClusters CTFCoder::decodeCompressedClusters(const CTF::base& ec, o2::ctf::CTFIOSize& iosize)
{
  CompressedClusters cc;
  cc.header = ec.getHeader();
  checkDictVersion(static_cast<const o2::ctf::CTFDictHeader&>(cc.header));
  ec.print(getPrefix(), mVerbosity);
#define DECODEITSMFT(part, slot) ec.decode(part, int(slot), mCoders[int(slot)].get())
  // clang-format off
  iosize += DECODEITSMFT(cc.firstChipROF, CTF::BLCfirstChipROF);
  iosize += DECODEITSMFT(cc.bcIncROF,     CTF::BLCbcIncROF);
  iosize += DECODEITSMFT(cc.orbitIncROF,  CTF::BLCorbitIncROF);
  iosize += DECODEITSMFT(cc.nclusROF,     CTF::BLCnclusROF);
  //
  iosize += DECODEITSMFT(cc.chipInc,      CTF::BLCchipInc);
  iosize += DECODEITSMFT(cc.chipMul,      CTF::BLCchipMul);
  iosize += DECODEITSMFT(cc.row,          CTF::BLCrow);
  iosize += DECODEITSMFT(cc.colInc,       CTF::BLCcolInc);
  iosize += DECODEITSMFT(cc.pattID,       CTF::BLCpattID);
  iosize += DECODEITSMFT(cc.pattMap,      CTF::BLCpattMap);
  // clang-format on
  return cc;
}
