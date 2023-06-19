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
/// \brief class for entropy encoding/decoding of FT0 digits data

#ifndef O2_FT0_CTFCODER_H
#define O2_FT0_CTFCODER_H

#include <algorithm>
#include <iterator>
#include <string>
#include "FT0Base/Geometry.h"
#include "DataFormatsFT0/CTF.h"
#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/ChannelData.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "FT0Base/FT0DigParam.h"
#include "DetectorsBase/CTFCoderBase.h"

class TTree;

namespace o2
{
namespace ft0
{

class CTFCoder : public o2::ctf::CTFCoderBase
{
 public:
  CTFCoder(o2::ctf::CTFCoderBase::OpType op) : o2::ctf::CTFCoderBase(op, CTF::getNBlocks(), o2::detectors::DetID::FT0) {}
  ~CTFCoder() final = default;

  /// entropy-encode digits to buffer with CTF
  template <typename VEC>
  o2::ctf::CTFIOSize encode(VEC& buff, const gsl::span<const Digit>& digitVec, const gsl::span<const ChannelData>& channelVec);

  /// entropy decode clusters from buffer with CTF
  template <typename VDIG, typename VCHAN>
  o2::ctf::CTFIOSize decode(const CTF::base& ec, VDIG& digitVec, VCHAN& channelVec);

  void createCoders(const std::vector<char>& bufVec, o2::ctf::CTFCoderBase::OpType op) final;

 private:
  /// compres digits clusters to CompressedDigits
  template <int MAJOR_VERSION, int MINOR_VERSION>
  void compress(CompressedDigits& cd, const gsl::span<const Digit>& digitVec, const gsl::span<const ChannelData>& channelVec);
  size_t estimateCompressedSize(const CompressedDigits& cc);

  /// decompress CompressedDigits to digits
  template <int MAJOR_VERSION, int MINOR_VERSION, typename VDIG, typename VCHAN>
  void decompress(const CompressedDigits& cd, VDIG& digitVec, VCHAN& channelVec);

  void appendToTree(TTree& tree, CTF& ec);
  void readFromTree(TTree& tree, int entry, std::vector<Digit>& digitVec, std::vector<ChannelData>& channelVec);
  void assignDictVersion(o2::ctf::CTFDictHeader& h) const final;
};

/// entropy-encode clusters to buffer with CTF
template <typename VEC>
o2::ctf::CTFIOSize CTFCoder::encode(VEC& buff, const gsl::span<const Digit>& digitVec, const gsl::span<const ChannelData>& channelVec)
{
  using MD = o2::ctf::Metadata::OptStore;
  // what to do which each field: see o2::ctd::Metadata explanation
  constexpr MD optField[CTF::getNBlocks()] = {
    MD::EENCODE_OR_PACK, // BLC_trigger
    MD::EENCODE_OR_PACK, // BLC_bcInc
    MD::EENCODE_OR_PACK, // BLC_orbitInc
    MD::EENCODE_OR_PACK, // BLC_nChan
    MD::EENCODE_OR_PACK, // BLC_status
    MD::EENCODE_OR_PACK, // BLC_idChan
    MD::EENCODE_OR_PACK, // BLC_qtcChain
    MD::EENCODE_OR_PACK, // BLC_cfdTime
    MD::EENCODE_OR_PACK  // BLC_qtcAmpl
  };
  CompressedDigits cd;
  if (mExtHeader.isValidDictTimeStamp()) {
    if (mExtHeader.minorVersion == 0 && mExtHeader.majorVersion == 1) {
      compress<1, 0>(cd, digitVec, channelVec);
    } else {
      compress<1, 1>(cd, digitVec, channelVec);
    }
  } else {
    compress<1, 1>(cd, digitVec, channelVec);
  }
  // book output size with some margin
  auto szIni = estimateCompressedSize(cd);
  buff.resize(szIni);

  auto ec = CTF::create(buff);
  using ECB = CTF::base;

  ec->setHeader(cd.header);
  assignDictVersion(static_cast<o2::ctf::CTFDictHeader&>(ec->getHeader()));
  ec->setANSHeader(mANSVersion);
  // at every encoding the buffer might be autoexpanded, so we don't work with fixed pointer ec
  o2::ctf::CTFIOSize iosize;
#define ENCODEFT0(part, slot, bits) CTF::get(buff.data())->encode(part, int(slot), bits, optField[int(slot)], &buff, mCoders[int(slot)], getMemMarginFactor());
  // clang-format off
  iosize += ENCODEFT0(cd.trigger,     CTF::BLC_trigger,  0);
  iosize += ENCODEFT0(cd.bcInc,       CTF::BLC_bcInc,    0);
  iosize += ENCODEFT0(cd.orbitInc,    CTF::BLC_orbitInc, 0);
  iosize += ENCODEFT0(cd.nChan,       CTF::BLC_nChan,    0);
  iosize += ENCODEFT0(cd.eventStatus, CTF::BLC_status,   0);
  iosize += ENCODEFT0(cd.idChan ,     CTF::BLC_idChan,   0);
  iosize += ENCODEFT0(cd.qtcChain,    CTF::BLC_qtcChain, 0);
  iosize += ENCODEFT0(cd.cfdTime,     CTF::BLC_cfdTime,  0);
  iosize += ENCODEFT0(cd.qtcAmpl,     CTF::BLC_qtcAmpl,  0);
  // clang-format on
  CTF::get(buff.data())->print(getPrefix(), mVerbosity);
  finaliseCTFOutput<CTF>(buff);
  iosize.rawIn = sizeof(Digit) * digitVec.size() + sizeof(ChannelData) * channelVec.size();
  return iosize;
}

/// decode entropy-encoded clusters to standard compact clusters
template <typename VDIG, typename VCHAN>
o2::ctf::CTFIOSize CTFCoder::decode(const CTF::base& ec, VDIG& digitVec, VCHAN& channelVec)
{
  CompressedDigits cd;
  cd.header = ec.getHeader();
  const auto& hd = static_cast<const o2::ctf::CTFDictHeader&>(cd.header);
  checkDictVersion(hd);
  ec.print(getPrefix(), mVerbosity);
  o2::ctf::CTFIOSize iosize;
#define DECODEFT0(part, slot) ec.decode(part, int(slot), mCoders[int(slot)])
  // clang-format off
  iosize += DECODEFT0(cd.trigger,     CTF::BLC_trigger);
  iosize += DECODEFT0(cd.bcInc,       CTF::BLC_bcInc);
  iosize += DECODEFT0(cd.orbitInc,    CTF::BLC_orbitInc);
  iosize += DECODEFT0(cd.nChan,       CTF::BLC_nChan);
  iosize += DECODEFT0(cd.eventStatus, CTF::BLC_status);
  iosize += DECODEFT0(cd.idChan,      CTF::BLC_idChan);
  iosize += DECODEFT0(cd.qtcChain,    CTF::BLC_qtcChain);
  iosize += DECODEFT0(cd.cfdTime,     CTF::BLC_cfdTime);
  iosize += DECODEFT0(cd.qtcAmpl,     CTF::BLC_qtcAmpl);
  // clang-format on
  //
  if (hd.minorVersion == 0 && hd.majorVersion == 1) {
    decompress<1, 0>(cd, digitVec, channelVec);
  } else {
    decompress<1, 1>(cd, digitVec, channelVec);
  }
  iosize.rawIn = sizeof(Digit) * digitVec.size() + sizeof(ChannelData) * channelVec.size();
  return iosize;
}

/// decompress compressed digits to standard digits
template <int MAJOR_VERSION, int MINOR_VERSION, typename VDIG, typename VCHAN>
void CTFCoder::decompress(const CompressedDigits& cd, VDIG& digitVec, VCHAN& channelVec)
{
  digitVec.clear();
  channelVec.clear();
  digitVec.reserve(cd.header.nTriggers);
  channelVec.reserve(cd.idChan.size());

  uint32_t firstEntry = 0, clCount = 0, chipCount = 0;
  o2::InteractionRecord ir(cd.header.firstBC, cd.header.firstOrbit);

  for (uint32_t idig = 0; idig < cd.header.nTriggers; idig++) {
    // restore ROFRecord
    if (cd.orbitInc[idig]) {  // non-0 increment => new orbit
      ir.bc = cd.bcInc[idig]; // bcInc has absolute meaning
      ir.orbit += cd.orbitInc[idig];
    } else {
      ir.bc += cd.bcInc[idig];
    }
    const auto& params = FT0DigParam::Instance();
    int triggerGate = params.mTime_trg_gate;
    firstEntry = channelVec.size();
    uint8_t chID = 0;
    int8_t nChanA = 0, nChanC = 0;
    int32_t amplA = 0, amplC = 0;
    int16_t timeA = 0, timeC = 0;
    for (uint8_t ic = 0; ic < cd.nChan[idig]; ic++) {
      auto icc = channelVec.size();
      if constexpr (MINOR_VERSION == 0 && MAJOR_VERSION == 1) {
        // Old decoding procedure, mostly for Pilot Beam in October 2021
        chID += cd.idChan[icc];
      } else {
        // New decoding procedure, w/o sorted ChID requriment
        chID = cd.idChan[icc];
      }
      const auto& chan = channelVec.emplace_back(chID, cd.cfdTime[icc], cd.qtcAmpl[icc], cd.qtcChain[icc]);
      if (std::abs(chan.CFDTime) < triggerGate) {
        if (chan.ChId < 4 * uint8_t(Geometry::NCellsA)) { // A side
          amplA += chan.QTCAmpl;
          timeA += chan.CFDTime;
          nChanA++;
        } else {
          amplC += chan.QTCAmpl;
          timeC += chan.CFDTime;
          nChanC++;
        }
      }
    }
    if (nChanA) {
      timeA /= nChanA;
      amplA *= 0.125;
    } else {
      timeA = Triggers::DEFAULT_TIME;
      amplA = Triggers::DEFAULT_AMP;
    }
    if (nChanC) {
      timeC /= nChanC;
      amplC *= 0.125;
    } else {
      timeC = Triggers::DEFAULT_TIME;
      amplC = Triggers::DEFAULT_AMP;
    }
    Triggers trig;
    trig.setTriggers(cd.trigger[idig], nChanA, nChanC, amplA, amplC, timeA, timeC);
    auto& d = digitVec.emplace_back(firstEntry, cd.nChan[idig], ir, trig, idig);
    d.setEventStatus(cd.eventStatus[idig]);
  }
}

///________________________________
template <int MAJOR_VERSION, int MINOR_VERSION>
void CTFCoder::compress(CompressedDigits& cd, const gsl::span<const Digit>& digitVec, const gsl::span<const ChannelData>& channelVec)
{
  // convert digits/channel to their compressed version
  cd.clear();
  cd.header.det = mDet;
  if (!digitVec.size()) {
    return;
  }
  uint32_t firstDig = digitVec.size(), nDigSel = digitVec.size(), nChanSel = channelVec.size();
  std::vector<bool> reject(digitVec.size());
  if (mIRFrameSelector.isSet()) {
    for (size_t id = 0; id < digitVec.size(); id++) {
      if (mIRFrameSelector.check(digitVec[id].mIntRecord) < 0) {
        reject[id] = true;
        nDigSel--;
        nChanSel -= digitVec[id].ref.getEntries();
      } else if (firstDig == digitVec.size()) {
        firstDig = id;
      }
    }
  } else {
    firstDig = 0;
  }
  if (nDigSel == 0) { // nothing is selected
    return;
  }

  const auto& dig0 = digitVec[firstDig];
  cd.header.nTriggers = nDigSel;
  cd.header.firstOrbit = dig0.getOrbit();
  cd.header.firstBC = dig0.getBC();
  cd.header.triggerGate = FT0DigParam::Instance().mTime_trg_gate;

  cd.trigger.resize(cd.header.nTriggers);
  cd.bcInc.resize(cd.header.nTriggers);
  cd.orbitInc.resize(cd.header.nTriggers);
  cd.eventStatus.resize(cd.header.nTriggers);
  cd.nChan.resize(cd.header.nTriggers);

  cd.idChan.resize(nChanSel);
  cd.qtcChain.resize(nChanSel);
  cd.cfdTime.resize(nChanSel);
  cd.qtcAmpl.resize(nChanSel);

  uint16_t prevBC = cd.header.firstBC;
  uint32_t prevOrbit = cd.header.firstOrbit;
  uint32_t ccount = 0, dcount = 0;
  for (uint32_t idig = 0; idig < digitVec.size(); idig++) {
    if (reject[idig]) {
      continue;
    }
    const auto& digit = digitVec[idig];
    const auto chanels = digit.getBunchChannelData(channelVec); // we assume the channels are sorted

    // fill trigger info
    cd.trigger[dcount] = digit.getTriggers().getTriggersignals();
    cd.eventStatus[dcount] = digit.getEventStatusWord();
    if (prevOrbit == digit.getOrbit()) {
      cd.bcInc[dcount] = digit.getBC() - prevBC;
      cd.orbitInc[dcount] = 0;
    } else {
      cd.bcInc[dcount] = digit.getBC();
      cd.orbitInc[dcount] = digit.getOrbit() - prevOrbit;
    }
    prevBC = digit.getBC();
    prevOrbit = digit.getOrbit();
    // fill channels info
    cd.nChan[dcount] = chanels.size();
    if (!cd.nChan[dcount]) {
      LOG(debug) << "Digits with no channels";
      dcount++;
      continue;
    }
    uint8_t prevChan = 0;
    for (uint8_t ic = 0; ic < cd.nChan[dcount]; ic++) {
      if constexpr (MINOR_VERSION == 0 && MAJOR_VERSION == 1) {
        cd.idChan[ccount] = chanels[ic].ChId - prevChan; // Old method, lets keep it for a while
      } else {
        cd.idChan[ccount] = chanels[ic].ChId;
      }
      cd.qtcChain[ccount] = chanels[ic].ChainQTC;
      cd.cfdTime[ccount] = chanels[ic].CFDTime;
      cd.qtcAmpl[ccount] = chanels[ic].QTCAmpl;
      prevChan = chanels[ic].ChId;
      ccount++;
    }
    dcount++;
  }
}

} // namespace ft0
} // namespace o2

#endif // O2_FT0_CTFCODER_H
