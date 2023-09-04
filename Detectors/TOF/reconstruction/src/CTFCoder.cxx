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
/// \author fnoferin@cern.ch
/// \brief class for entropy encoding/decoding of TOF compressed digits data

#include "TOFReconstruction/CTFCoder.h"
#include "CommonUtils/StringUtils.h"
#include <TTree.h>

using namespace o2::tof;

///___________________________________________________________________________________
// Register encoded data in the tree (Fill is not called, will be done by caller)
void CTFCoder::appendToTree(TTree& tree, CTF& ec)
{
  ec.appendToTree(tree, mDet.getName());
}

///___________________________________________________________________________________
// extract and decode data from the tree
void CTFCoder::readFromTree(TTree& tree, int entry, std::vector<ReadoutWindowData>& rofRecVec, std::vector<Digit>& cdigVec, std::vector<uint8_t>& pattVec)
{
  assert(entry >= 0 && entry < tree.GetEntries());
  CTF ec;
  ec.readFromTree(tree, mDet.getName(), entry);
  decode(ec, rofRecVec, cdigVec, pattVec);
}

///________________________________
void CTFCoder::compress(CompressedInfos& cc,
                        const gsl::span<const ReadoutWindowData>& rofRecVec,
                        const gsl::span<const Digit>& cdigVec,
                        const gsl::span<const uint8_t>& pattVec)
{
  // store in the header the orbit of 1st ROF
  cc.clear();
  cc.header.det = mDet;
  if (!rofRecVec.size()) {
    return;
  }
  std::vector<Digit> digCopy;
  int ndigTot = 0, nrofTot = 0, nrofIni = rofRecVec.size(), ndigIni = cdigVec.size();
  uint16_t prevBC = 0;
  uint32_t prevOrbit = 0;
  LOGF(debug, "TOF compress %d ReadoutWindow with %ld digits", nrofIni, ndigIni);
  if (!mIRFrameSelector.isSet()) {
    cc.bcIncROF.reserve(nrofIni);
    cc.orbitIncROF.reserve(nrofIni);
    cc.ndigROF.reserve(nrofIni);
    cc.ndiaROF.reserve(nrofIni);
    cc.ndiaCrate.reserve(nrofIni * 72);

    cc.timeFrameInc.reserve(ndigIni);
    cc.timeTDCInc.reserve(ndigIni);
    cc.stripID.reserve(ndigIni);
    cc.chanInStrip.reserve(ndigIni);
    cc.tot.reserve(ndigIni);
  }

  for (int rof0 = 0; rof0 < nrofIni; rof0++) {
    const auto& rofRec = rofRecVec[rof0];
    const auto ir = rofRec.getBCData();
    if (mIRFrameSelector.isSet() && mIRFrameSelector.check(o2::dataformats::IRFrame{ir, ir + (o2::constants::lhc::LHCMaxBunches / 3 - 1)}) < 0) {
      continue;
    }
    int64_t rofInBC = ir.toLong();
    digCopy.clear(); // make a copy of digits
    int ndig = rofRec.size(), idigMin = rofRec.first(), idigMax = idigMin + ndig;
    for (int idig = idigMin; idig < idigMax; idig++) {
      digCopy.push_back(cdigVec[idig]);
    }
    // sort digits according to time (ascending order)
    std::sort(digCopy.begin(), digCopy.end(), [](o2::tof::Digit a, o2::tof::Digit b) { return (a.getBC() == b.getBC()) ? (a.getTDC() < b.getTDC()) : a.getBC() < b.getBC(); });

    int timeframe = 0, tdc = 0, ndigAcc = 0;
    for (int idig = idigMin; idig < idigMax; idig++) {
      const auto& dig = digCopy[idig - idigMin];
      if (mIRFrameSelector.isSet() && mIRFrameSelector.check(dig.getIR()) < 0) {
        continue;
      }
      ndigAcc++;
      int deltaBC = dig.getBC() - rofInBC;
      int ctimeframe = deltaBC / 64;
      int cTDC = (deltaBC % 64) * 1024 + dig.getTDC();
      if (ctimeframe == timeframe) {
        cc.timeFrameInc.push_back(0);
        cc.timeTDCInc.push_back(cTDC - tdc);
      } else {
        cc.timeFrameInc.push_back(ctimeframe - timeframe);
        cc.timeTDCInc.push_back(cTDC);
        timeframe = ctimeframe;
      }
      tdc = cTDC;

      int chan = dig.getChannel();
      cc.stripID.push_back(chan / Geo::NPADS);
      cc.chanInStrip.push_back(chan % Geo::NPADS);
      cc.tot.push_back(dig.getTOT());
      LOGF(debug, "%d) TOFBC = %d, deltaBC = %d, TDC = %d, CH=%d", nrofTot, rofInBC, deltaBC, cTDC, chan);
      LOGF(debug, "%d) TF=%d, TDC=%d, STRIP=%d, CH=%d, TOT=%d", idig, cc.timeFrameInc[idig], cc.timeTDCInc[idig], cc.stripID[idig], cc.chanInStrip[idig], cc.tot[idig]);
    }
    ndigTot += ndigAcc;
    if (ndigAcc || !mIRFrameSelector.isSet()) {
      if (nrofTot == 0) { // very 1st ROF
        prevOrbit = cc.header.firstOrbit = ir.orbit;
        prevBC = cc.header.firstBC = ir.bc;
      }
      if (ir.orbit == prevOrbit) {
        cc.orbitIncROF.push_back(0);
        cc.bcIncROF.push_back(ir.bc - prevBC); // store increment of BC if in the same orbit
      } else {
        cc.orbitIncROF.push_back(ir.orbit - prevOrbit);
        cc.bcIncROF.push_back(ir.bc); // otherwise, store absolute bc
        prevOrbit = ir.orbit;
      }
      cc.ndigROF.push_back(ndigAcc);
      cc.ndiaROF.push_back(rofRec.sizeDia());
      cc.ndiaCrate.reserve(cc.ndiaCrate.size() + 72);
      for (int icrate = 0; icrate < 72; icrate++) {
        if (rofRec.isEmptyCrate(icrate)) {
          cc.ndiaCrate.push_back(0);
        } else {
          cc.ndiaCrate.push_back(rofRec.getDiagnosticInCrate(icrate) + 1); // shifted by one since -1 means crate not available (then to get unsigned int)
        }
      }
      prevBC = ir.bc;
      nrofTot++;
    }
  } // loop over ROFs
  if (nrofTot) {
    cc.header.nROFs = nrofTot;
    cc.header.nDigits = ndigTot;
    cc.header.nPatternBytes = pattVec.size();
    cc.pattMap.resize(cc.header.nPatternBytes);
    memcpy(cc.pattMap.data(), pattVec.data(), cc.header.nPatternBytes); // RSTODO: do we need this?
  }
}

///________________________________
void CTFCoder::createCoders(const std::vector<char>& bufVec, o2::ctf::CTFCoderBase::OpType op)
{
  const auto ctf = CTF::getImage(bufVec.data());
  CompressedInfos cc; // just to get member types
#define MAKECODER(part, slot) createCoder(op, std::get<rans::RenormedDenseHistogram<decltype(part)::value_type>>(ctf.getDictionary<decltype(part)::value_type>(slot, mANSVersion)), int(slot))
  // clang-format off
  MAKECODER(cc.bcIncROF,     CTF::BLCbcIncROF);
  MAKECODER(cc.orbitIncROF,  CTF::BLCorbitIncROF);
  MAKECODER(cc.ndigROF,      CTF::BLCndigROF);
  MAKECODER(cc.ndiaROF,      CTF::BLCndiaROF);
  MAKECODER(cc.ndiaCrate,    CTF::BLCndiaCrate);

  MAKECODER(cc.timeFrameInc, CTF::BLCtimeFrameInc);
  MAKECODER(cc.timeTDCInc,   CTF::BLCtimeTDCInc);
  MAKECODER(cc.stripID,      CTF::BLCstripID);
  MAKECODER(cc.chanInStrip,  CTF::BLCchanInStrip);
  MAKECODER(cc.tot,          CTF::BLCtot);
  MAKECODER(cc.pattMap,      CTF::BLCpattMap);
  // clang-format on
}

///________________________________
size_t CTFCoder::estimateCompressedSize(const CompressedInfos& cc)
{
  size_t sz = 0;
  // RS FIXME this is very crude estimate, instead, an empirical values should be used

  sz += estimateBufferSize(static_cast<int>(CTF::BLCbcIncROF), cc.bcIncROF);
  sz += estimateBufferSize(static_cast<int>(CTF::BLCorbitIncROF), cc.orbitIncROF);
  sz += estimateBufferSize(static_cast<int>(CTF::BLCndigROF), cc.ndigROF);
  sz += estimateBufferSize(static_cast<int>(CTF::BLCndiaROF), cc.ndiaROF);
  sz += estimateBufferSize(static_cast<int>(CTF::BLCndiaCrate), cc.ndiaCrate);
  sz += estimateBufferSize(static_cast<int>(CTF::BLCtimeFrameInc), cc.timeFrameInc);
  sz += estimateBufferSize(static_cast<int>(CTF::BLCtimeTDCInc), cc.timeTDCInc);
  sz += estimateBufferSize(static_cast<int>(CTF::BLCstripID), cc.stripID);
  sz += estimateBufferSize(static_cast<int>(CTF::BLCchanInStrip), cc.chanInStrip);
  sz += estimateBufferSize(static_cast<int>(CTF::BLCtot), cc.tot);
  sz += estimateBufferSize(static_cast<int>(CTF::BLCpattMap), cc.pattMap);
  sz *= 2. / 3; // if needed, will be autoexpanded
  LOG(debug) << "Estimated output size is " << sz << " bytes";
  return sz;
}