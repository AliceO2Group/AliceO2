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
/// \brief class for entropy encoding/decoding of FDD digits data

#ifndef O2_FDD_CTFCODER_H
#define O2_FDD_CTFCODER_H

#include <algorithm>
#include <iterator>
#include <string>
#include "FDDBase/Geometry.h"
#include "DataFormatsFDD/CTF.h"
#include "DataFormatsFDD/Digit.h"
#include "DataFormatsFDD/ChannelData.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsBase/CTFCoderBase.h"
#include "rANS/rans.h"

class TTree;

namespace o2
{
namespace fdd
{

class CTFCoder : public o2::ctf::CTFCoderBase
{
 public:
  CTFCoder() : o2::ctf::CTFCoderBase(CTF::getNBlocks(), o2::detectors::DetID::FDD) {}
  ~CTFCoder() = default;

  /// entropy-encode digits to buffer with CTF
  template <typename VEC>
  void encode(VEC& buff, const gsl::span<const Digit>& digitVec, const gsl::span<const ChannelData>& channelVec);

  /// entropy decode clusters from buffer with CTF
  template <typename VDIG, typename VCHAN>
  void decode(const CTF::base& ec, VDIG& digitVec, VCHAN& channelVec);

  void createCoders(const std::string& dictPath, o2::ctf::CTFCoderBase::OpType op);

 private:
  /// compres digits clusters to CompressedDigits
  void compress(CompressedDigits& cd, const gsl::span<const Digit>& digitVec, const gsl::span<const ChannelData>& channelVec);
  size_t estimateCompressedSize(const CompressedDigits& cc);

  /// decompress CompressedDigits to digits
  template <typename VDIG, typename VCHAN>
  void decompress(const CompressedDigits& cd, VDIG& digitVec, VCHAN& channelVec);

  void appendToTree(TTree& tree, CTF& ec);
  void readFromTree(TTree& tree, int entry, std::vector<Digit>& digitVec, std::vector<ChannelData>& channelVec);

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

    MD::EENCODE, // BLC_idChan
    MD::EENCODE, // BLC_time
    MD::EENCODE, // BLC_charge
    MD::EENCODE  // BLC_feeBits
  };
  CompressedDigits cd;
  compress(cd, digitVec, channelVec);

  // book output size with some margin
  auto szIni = estimateCompressedSize(cd);
  buff.resize(szIni);

  auto ec = CTF::create(buff);
  using ECB = CTF::base;

  ec->setHeader(cd.header);
  ec->getANSHeader().majorVersion = 0;
  ec->getANSHeader().minorVersion = 1;
  // at every encoding the buffer might be autoexpanded, so we don't work with fixed pointer ec
#define ENCODEFDD(part, slot, bits) CTF::get(buff.data())->encode(part, int(slot), bits, optField[int(slot)], &buff, mCoders[int(slot)].get());
  // clang-format off
  ENCODEFDD(cd.trigger,   CTF::BLC_trigger,  0);
  ENCODEFDD(cd.bcInc,     CTF::BLC_bcInc,    0);
  ENCODEFDD(cd.orbitInc,  CTF::BLC_orbitInc, 0);
  ENCODEFDD(cd.nChan,     CTF::BLC_nChan,    0);

  ENCODEFDD(cd.idChan ,   CTF::BLC_idChan,   0);
  ENCODEFDD(cd.time,      CTF::BLC_time,     0);
  ENCODEFDD(cd.charge,    CTF::BLC_charge,   0);
  ENCODEFDD(cd.feeBits,   CTF::BLC_feeBits,  0);
  // clang-format on
  CTF::get(buff.data())->print(getPrefix());
}

/// decode entropy-encoded clusters to standard compact clusters
template <typename VDIG, typename VCHAN>
void CTFCoder::decode(const CTF::base& ec, VDIG& digitVec, VCHAN& channelVec)
{
  CompressedDigits cd;
  cd.header = ec.getHeader();
  ec.print(getPrefix());
#define DECODEFDD(part, slot) ec.decode(part, int(slot), mCoders[int(slot)].get())
  // clang-format off
  DECODEFDD(cd.trigger,   CTF::BLC_trigger);
  DECODEFDD(cd.bcInc,     CTF::BLC_bcInc); 
  DECODEFDD(cd.orbitInc,  CTF::BLC_orbitInc);
  DECODEFDD(cd.nChan,     CTF::BLC_nChan);

  DECODEFDD(cd.idChan,    CTF::BLC_idChan);
  DECODEFDD(cd.time,      CTF::BLC_time);
  DECODEFDD(cd.charge,    CTF::BLC_charge);
  DECODEFDD(cd.feeBits,   CTF::BLC_feeBits);
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
    for (uint8_t ic = 0; ic < cd.nChan[idig]; ic++) {
      auto icc = channelVec.size();
      const auto& chan = channelVec.emplace_back((chID += cd.idChan[icc]), cd.time[icc], cd.charge[icc], cd.feeBits[icc]);
      //
      // rebuild digit
      if (chan.mPMNumber > 7) { // A side
        amplA += chan.mChargeADC;
        timeA += chan.mTime;
        trig.nChanA++;

      } else {
        amplC += chan.mChargeADC;
        timeC += chan.mTime;
        trig.nChanC++;
      }
    }
    if (trig.nChanA) {
      trig.timeA = timeA / trig.nChanA;
      trig.amplA = amplA * 0.125;
    }
    if (trig.nChanC) {
      trig.timeC = timeC / trig.nChanC;
      trig.amplC = amplC * 0.125;
    }
    digitVec.emplace_back(firstEntry, cd.nChan[idig], ir, trig);
  }
}

} // namespace fdd
} // namespace o2

#endif // O2_FDD_CTFCODER_H
