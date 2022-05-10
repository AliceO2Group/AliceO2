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
#include "rANS/rans.h"

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
  void encode(VEC& buff, const gsl::span<const Digit>& digitVec, const gsl::span<const ChannelData>& channelVec);

  /// entropy decode clusters from buffer with CTF
  template <typename VDIG, typename VCHAN>
  void decode(const CTF::base& ec, VDIG& digitVec, VCHAN& channelVec);

  void createCoders(const std::vector<char>& bufVec, o2::ctf::CTFCoderBase::OpType op) final;

 private:
  /// compres digits clusters to CompressedDigits
  void compress(CompressedDigits& cd, const gsl::span<const Digit>& digitVec, const gsl::span<const ChannelData>& channelVec);
  size_t estimateCompressedSize(const CompressedDigits& cc);

  /// decompress CompressedDigits to digits
  template <typename VDIG, typename VCHAN>
  void decompress(const CompressedDigits& cd, VDIG& digitVec, VCHAN& channelVec);

  void appendToTree(TTree& tree, CTF& ec);
  void readFromTree(TTree& tree, int entry, std::vector<Digit>& digitVec, std::vector<ChannelData>& channelVec);
};

/// entropy-encode clusters to buffer with CTF
template <typename VEC>
void CTFCoder::encode(VEC& buff, const gsl::span<const Digit>& digitVec, const gsl::span<const ChannelData>& channelVec)
{
  using MD = o2::ctf::Metadata::OptStore;
  // what to do which each field: see o2::ctd::Metadata explanation
  constexpr MD optField[CTF::getNBlocks()] = {
    MD::EENCODE, // BLC_trigger
    MD::EENCODE, // BLC_bcInc
    MD::EENCODE, // BLC_orbitInc
    MD::EENCODE, // BLC_nChan
    MD::EENCODE, // BLC_status
    MD::EENCODE, // BLC_idChan
    MD::EENCODE, // BLC_qtcChain
    MD::EENCODE, // BLC_cfdTime
    MD::EENCODE  // BLC_qtcAmpl
  };
  CompressedDigits cd;
  compress(cd, digitVec, channelVec);

  // book output size with some margin
  auto szIni = estimateCompressedSize(cd);
  buff.resize(szIni);

  auto ec = CTF::create(buff);
  using ECB = CTF::base;

  ec->setHeader(cd.header);
  assignDictVersion(static_cast<o2::ctf::CTFDictHeader&>(ec->getHeader()));
  ec->getANSHeader().majorVersion = 0;
  ec->getANSHeader().minorVersion = 1;
  // at every encoding the buffer might be autoexpanded, so we don't work with fixed pointer ec
#define ENCODEFT0(part, slot, bits) CTF::get(buff.data())->encode(part, int(slot), bits, optField[int(slot)], &buff, mCoders[int(slot)].get(), getMemMarginFactor());
  // clang-format off
  ENCODEFT0(cd.trigger,     CTF::BLC_trigger,  0);
  ENCODEFT0(cd.bcInc,       CTF::BLC_bcInc,    0);
  ENCODEFT0(cd.orbitInc,    CTF::BLC_orbitInc, 0);
  ENCODEFT0(cd.nChan,       CTF::BLC_nChan,    0);
  ENCODEFT0(cd.eventStatus, CTF::BLC_status,   0);
  ENCODEFT0(cd.idChan ,     CTF::BLC_idChan,   0);
  ENCODEFT0(cd.qtcChain,    CTF::BLC_qtcChain, 0);
  ENCODEFT0(cd.cfdTime,     CTF::BLC_cfdTime,  0);
  ENCODEFT0(cd.qtcAmpl,     CTF::BLC_qtcAmpl,  0);
  // clang-format on
  CTF::get(buff.data())->print(getPrefix(), mVerbosity);
}

/// decode entropy-encoded clusters to standard compact clusters
template <typename VDIG, typename VCHAN>
void CTFCoder::decode(const CTF::base& ec, VDIG& digitVec, VCHAN& channelVec)
{
  CompressedDigits cd;
  cd.header = ec.getHeader();
  checkDictVersion(static_cast<const o2::ctf::CTFDictHeader&>(cd.header));
  ec.print(getPrefix(), mVerbosity);
#define DECODEFT0(part, slot) ec.decode(part, int(slot), mCoders[int(slot)].get())
  // clang-format off
  DECODEFT0(cd.trigger,     CTF::BLC_trigger);
  DECODEFT0(cd.bcInc,       CTF::BLC_bcInc);
  DECODEFT0(cd.orbitInc,    CTF::BLC_orbitInc);
  DECODEFT0(cd.nChan,       CTF::BLC_nChan);
  DECODEFT0(cd.eventStatus, CTF::BLC_status);
  DECODEFT0(cd.idChan,      CTF::BLC_idChan);
  DECODEFT0(cd.qtcChain,    CTF::BLC_qtcChain);
  DECODEFT0(cd.cfdTime,     CTF::BLC_cfdTime);
  DECODEFT0(cd.qtcAmpl,     CTF::BLC_qtcAmpl);
  // clang-format on
  //
  decompress(cd, digitVec, channelVec);
}

/// decompress compressed digits to standard digits
template <typename VDIG, typename VCHAN>
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
      const auto& chan = channelVec.emplace_back((chID += cd.idChan[icc]), cd.cfdTime[icc], cd.qtcAmpl[icc], cd.qtcChain[icc]);
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

} // namespace ft0
} // namespace o2

#endif // O2_FT0_CTFCODER_H
