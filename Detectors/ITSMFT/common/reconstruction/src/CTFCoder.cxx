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
                        const gsl::span<const unsigned char>& pattVec)
{
  // store in the header the orbit of 1st ROF
  cc.clear();
  if (!rofRecVec.size()) {
    return;
  }
  const auto& rofRec0 = rofRecVec[0];
  cc.header.det = mDet;
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
#ifdef _CHECK_INCREMENTES_
      if (intRec.bc < prevBC) {
        LOG(warning) << "Negative BC increment " << intRec.bc << " -> " << prevBC;
      }
#endif
      cc.bcIncROF[irof] = intRec.bc - prevBC; // store increment of BC if in the same orbit
    } else {
      cc.orbitIncROF[irof] = intRec.orbit - prevOrbit;
#ifdef _CHECK_INCREMENTES_
      if (intRec.orbit < prevOrbit) {
        LOG(warning) << "Negative Orbit increment " << intRec.orbit << " -> " << prevOrbit;
      }
#endif
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
    if (icl != iclMax) { // there are still clusters to store
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
        cc.colInc[icl] = int16_t(cl.getCol()) - prevCol;
        prevCol = cl.getCol();
      } else { // for new chips store chipID increment and abs. column
        // cc.chipInc[icl] = cl.getChipID() - prevChip;  // this is the version with chipInc stored for every pixel
        cc.chipInc.push_back(cl.getChipID() - prevChip); // this is the version with chipInc stored once per new chip
#ifdef _CHECK_INCREMENTES_
        if (cl.getChipID() < prevChip) {
          LOG(warning) << "Negative Chip increment " << cl.getChipID() << " -> " << prevChip;
        }
#endif
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
