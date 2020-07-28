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
/// \author fnoferin@cern.ch
/// \brief class for entropy encoding/decoding of TOF compressed digits data

#include "TOFReconstruction/CTFCoder.h"
#include "CommonUtils/StringUtils.h"
#include <TTree.h>

using namespace o2::tof;

///___________________________________________________________________________________
// Register encoded data in the tree (Fill is not called, will be done by caller)
void CTFCoder::appendToTree(TTree& tree, o2::detectors::DetID id, CTF& ec)
{
  ec.appendToTree(tree, id.getName());
}

///___________________________________________________________________________________
// extract and decode data from the tree
void CTFCoder::readFromTree(TTree& tree, int entry, o2::detectors::DetID id,
                            std::vector<ReadoutWindowData>& rofRecVec, std::vector<Digit>& cdigVec, std::vector<uint32_t>& pattVec)
{
  assert(entry >= 0 && entry < tree.GetEntries());
  CTF ec;
  ec.readFromTree(tree, id.getName(), entry);
  decode(ec, rofRecVec, cdigVec, pattVec);
}

///________________________________
void CTFCoder::compress(CompressedInfos& cc,
                        const gsl::span<const ReadoutWindowData>& rofRecVec,
                        const gsl::span<const Digit>& cdigVec,
                        const gsl::span<const uint32_t>& pattVec)
{
  // store in the header the orbit of 1st ROF
  cc.clear();
  if (!rofRecVec.size()) {
    return;
  }
  const auto& rofRec0 = rofRecVec[0];
  int nrof = rofRecVec.size();

  cc.header.nROFs = nrof;
  cc.header.firstOrbit = rofRec0.getBCData().orbit;
  cc.header.firstBC = rofRec0.getBCData().bc;
  cc.header.nPatternBytes = pattVec.size();
  cc.header.nDigits = cdigVec.size();

  cc.bcIncROF.resize(cc.header.nROFs);
  cc.orbitIncROF.resize(cc.header.nROFs);
  cc.ndigROF.resize(cc.header.nROFs);
  cc.ndiaROF.resize(cc.header.nROFs);
  //
  cc.timeFrameInc.resize(cc.header.nDigits);
  cc.timeTDCInc.resize(cc.header.nDigits);
  cc.stripID.resize(cc.header.nDigits);
  cc.chanInStrip.resize(cc.header.nDigits);
  cc.tot.resize(cc.header.nDigits);
  cc.pattMap.resize(cc.header.nPatternBytes);

  uint16_t prevBC = cc.header.firstBC;
  uint32_t prevOrbit = cc.header.firstOrbit;

  std::vector<Digit> digCopy;

  for (uint32_t irof = 0; irof < rofRecVec.size(); irof++) {
    const auto& rofRec = rofRecVec[irof];

    const auto& intRec = rofRec.getBCData();
    int rofInBC = intRec.toLong();
    // define interaction record
    if (intRec.orbit == prevOrbit) {
      cc.orbitIncROF[irof] = 0;
      cc.bcIncROF[irof] = intRec.bc - prevBC; // store increment of BC if in the same orbit
    } else {
      cc.orbitIncROF[irof] = intRec.orbit - prevOrbit;
      cc.bcIncROF[irof] = intRec.bc; // otherwise, store absolute bc
      prevOrbit = intRec.orbit;
    }
    prevBC = intRec.bc;
    auto ndig = rofRec.size();
    cc.ndigROF[irof] = ndig;
    cc.ndiaROF[irof] = rofRec.sizeDia();

    if (!ndig) { // no hits data for this ROF --> not fill
      continue;
    }

    int idig = rofRec.first(), idigMax = idig + ndig;

    // make a copy of digits
    digCopy.clear();
    for (; idig < idigMax; idig++) {
      digCopy.emplace_back(cdigVec[idig]);
    }

    // sort digits according to time (ascending order)
    std::sort(digCopy.begin(), digCopy.end(),
              [](o2::tof::Digit a, o2::tof::Digit b) {
                if (a.getBC() == b.getBC())
                  return a.getTDC() < b.getTDC();
                else
                  return a.getBC() < b.getBC();
              });

    int timeframe = 0;
    int tdc = 0;
    idig = 0;
    for (; idig < ndig; idig++) {
      const auto& dig = digCopy[idig];
      int deltaBC = dig.getBC() - rofInBC;
      int ctimeframe = deltaBC / 64;
      int cTDC = (deltaBC % 64) * 1024 + dig.getTDC();
      if (ctimeframe == timeframe) {
        cc.timeFrameInc[idig] = 0;
        cc.timeTDCInc[idig] = cTDC - tdc;
      } else {
        cc.timeFrameInc[idig] = ctimeframe - timeframe;
        cc.timeTDCInc[idig] = cTDC;
        timeframe = ctimeframe;
      }
      tdc = cTDC;

      int chan = dig.getChannel();
      cc.stripID[idig] = chan / Geo::NPADS;
      cc.chanInStrip[idig] = chan % Geo::NPADS;
      cc.tot[idig] = dig.getTOT();
    }
  }
  // store explicit patters as they are
  memcpy(cc.pattMap.data(), pattVec.data(), cc.header.nPatternBytes); // RSTODO: do we need this?
}
