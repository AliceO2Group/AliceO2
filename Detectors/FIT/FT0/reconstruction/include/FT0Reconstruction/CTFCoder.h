// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "rANS/rans.h"

class TTree;

namespace o2
{
namespace ft0
{

class CTFCoder
{
 public:
  /// entropy-encode digits to buffer with CTF
  template <typename VEC>
  static void encode(VEC& buff, const gsl::span<const Digit>& digitVec, const gsl::span<const ChannelData>& channelVec);

  /// entropy decode clusters from buffer with CTF
  template <typename VDIG, typename VCHAN>
  static void decode(const CTF::base& ec, VDIG& digitVec, VCHAN& channelVec);

 private:
  /// compres digits clusters to CompressedDigits
  static void compress(CompressedDigits& cd, const gsl::span<const Digit>& digitVec, const gsl::span<const ChannelData>& channelVec);

  /// decompress CompressedDigits to digits
  template <typename VDIG, typename VCHAN>
  static void decompress(const CompressedDigits& cd, VDIG& digitVec, VCHAN& channelVec);

  static void appendToTree(TTree& tree, CTF& ec);
  static void readFromTree(TTree& tree, int entry, std::vector<Digit>& digitVec, std::vector<ChannelData>& channelVec);

  ClassDefNV(CTFCoder, 1);
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
    //  MD::EENCODE, // BLC_flags
    MD::EENCODE, // BLC_idChan
    MD::EENCODE, // BLC_qtcChain
    MD::EENCODE, // BLC_cfdTime
    MD::EENCODE  // BLC_qtcAmpl
  };
  CompressedDigits cd;
  compress(cd, digitVec, channelVec);
  auto ec = CTF::create(buff);
  using ECB = CTF::base;

  ec->setHeader(cd.header);
  ec->getANSHeader().majorVersion = 0;
  ec->getANSHeader().minorVersion = 1;
  // at every encoding the buffer might be autoexpanded, so we don't work with fixed pointer ec
#define ENCODE CTF::get(buff.data())->encode
  // clang-format off
  ENCODE(cd.trigger,   CTF::BLC_trigger,  o2::rans::ProbabilityBits16Bit, optField[CTF::BLC_trigger],  &buff);
  ENCODE(cd.bcInc,     CTF::BLC_bcInc,    o2::rans::ProbabilityBits16Bit, optField[CTF::BLC_bcInc],    &buff);
  ENCODE(cd.orbitInc,  CTF::BLC_orbitInc, o2::rans::ProbabilityBits16Bit, optField[CTF::BLC_orbitInc], &buff);
  ENCODE(cd.nChan,     CTF::BLC_nChan,    o2::rans::ProbabilityBits8Bit,  optField[CTF::BLC_nChan],    &buff);
  //  ENCODE(cd.eventFlags, CTF::BLC_flags,    o2::rans::ProbabilityBits8Bit,  optField[CTF::BLC_flags],    &buff);
  ENCODE(cd.idChan ,   CTF::BLC_idChan,   o2::rans::ProbabilityBits8Bit,  optField[CTF::BLC_idChan],   &buff);
  ENCODE(cd.qtcChain,  CTF::BLC_qtcChain,      o2::rans::ProbabilityBits8Bit,  optField[CTF::BLC_qtcChain],      &buff);
  ENCODE(cd.cfdTime,   CTF::BLC_cfdTime,  o2::rans::ProbabilityBits16Bit, optField[CTF::BLC_cfdTime],  &buff);
  ENCODE(cd.qtcAmpl,   CTF::BLC_qtcAmpl,  o2::rans::ProbabilityBits25Bit, optField[CTF::BLC_qtcAmpl],  &buff);
  // clang-format on
}

/// decode entropy-encoded clusters to standard compact clusters
template <typename VDIG, typename VCHAN>
void CTFCoder::decode(const CTF::base& ec, VDIG& digitVec, VCHAN& channelVec)
{
  CompressedDigits cd;
  cd.header = ec.getHeader();
  // clang-format off
  ec.decode(cd.trigger,   CTF::BLC_trigger);
  ec.decode(cd.bcInc,     CTF::BLC_bcInc); 
  ec.decode(cd.orbitInc,  CTF::BLC_orbitInc);
  ec.decode(cd.nChan,     CTF::BLC_nChan);
  //  ec.decode(cd.eventFlags,     CTF::BLC_flags);
  ec.decode(cd.idChan,    CTF::BLC_idChan);
  ec.decode(cd.qtcChain,  CTF::BLC_qtcChain);
  ec.decode(cd.cfdTime,   CTF::BLC_cfdTime);
  ec.decode(cd.qtcAmpl,   CTF::BLC_qtcAmpl);
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
    Triggers trig;
    trig.triggersignals = cd.trigger[idig];

    firstEntry = channelVec.size();
    uint8_t chID = 0;
    int amplA = 0, amplC = 0, timeA = 0, timeC = 0;
    //  int mTime_trg_gate = 192; // #channels
    for (uint8_t ic = 0; ic < cd.nChan[idig]; ic++) {
      auto icc = channelVec.size();
      const auto& chan = channelVec.emplace_back((chID += cd.idChan[icc]), cd.cfdTime[icc], cd.qtcAmpl[icc], cd.qtcChain[icc]);
      //
      // rebuild digit
      if (std::abs(chan.CFDTime) < Geometry::mTime_trg_gate) {
        if (chan.ChId < 4 * uint8_t(Geometry::NCellsA)) { // A side
          amplA += chan.QTCAmpl;
          timeA += chan.CFDTime;
          trig.nChanA++;

        } else {
          amplC += chan.QTCAmpl;
          timeC += chan.CFDTime;
          trig.nChanC++;
        }
      }
    }
    if (trig.nChanA) {
      trig.timeA = timeA / trig.nChanA;
      trig.amplA = amplA / 8;
    }
    if (trig.nChanC) {
      trig.timeC = timeC / trig.nChanC;
      trig.amplC = amplC / 8;
    }
    digitVec.emplace_back(firstEntry, cd.nChan[idig], ir, trig, idig);
  }
}

} // namespace ft0
} // namespace o2

#endif // O2_FT0_CTFCODER_H
