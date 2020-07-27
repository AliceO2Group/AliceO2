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
                            std::vector<ReadoutWindowData>& rofRecVec, std::vector<Digit>& cdigVec, std::vector<unsigned char>& pattVec)
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
                        const gsl::span<const unsigned char>& pattVec)
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
///___________________________________________________________________________________
/// entropy-encode digits to buffer with CTF
template <typename VEC>
void CTFCoder::encode(VEC& buff, const gsl::span<const ReadoutWindowData>& rofRecVec, const gsl::span<const Digit>& cdigVec, const gsl::span<const unsigned char>& pattVec)
{
  using MD = o2::ctf::Metadata::OptStore;
  // what to do which each field: see o2::ctd::Metadata explanation
  constexpr MD optField[CTF::getNBlocks()] = {
    MD::EENCODE, //BLCbcIncROF
    MD::EENCODE, //BLCorbitIncROF
    MD::EENCODE, //BLCndigROF
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

  o2::InteractionRecord prevIR(cc.header.firstBC, cc.header.firstOrbit);
  uint32_t firstEntry = 0, digCount = 0, stripCount = 0;
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
    firstEntry += cc.ndigROF[irof];

    // restore hit data
    uint ctimeframe = 0;
    uint ctdc = 0;

    int firstDig = digCount;

    for (uint32_t idig = 0; idig < cc.ndigROF[irof]; idig++) {
      auto& digit = cdigVec[digCount];
      if (cc.timeFrameInc[digCount]) { // new time frame
        ctdc = cc.timeTDCInc[digCount];
        ctimeframe += cc.timeFrameInc[digCount];
      } else {
        ctdc += cc.timeTDCInc[digCount];
      }

      digit.setBC(uint32_t(ctimeframe) * 64 + ctdc / 1024);
      digit.setTDC(ctdc % 1024);
      digit.setTOT(cc.tot[digCount]);
      digit.setChannel(uint32_t(cc.stripID[digCount]) * 96 + cc.chanInStrip[digCount]);

      digCount++;
    }

    // sort digits according to strip number within the ROF
    if (digCount > firstDig) {
      std::partial_sort(cdigVec.begin() + firstDig, cdigVec.begin() + digCount - 1, cdigVec.end(),
                        [](o2::tof::Digit a, o2::tof::Digit b) {
                          int str1 = a.getChannel() / 1600;
                          int str2 = b.getChannel() / 1600;
                          return (str1 <= str2);
                        });
    }
  }
  // explicit patterns
  memcpy(pattVec.data(), cc.pattMap.data(), cc.header.nPatternBytes); // RSTODO use swap?
  assert(digCount == cc.header.nDigits);

  if (digCount != cc.header.nDigits) {
    LOG(ERROR) << "expected " << cc.header.nDigits << " but counted " << digCount << " in ROFRecords";
    throw std::runtime_error("mismatch between expected and counter number of digits");
  }
}
