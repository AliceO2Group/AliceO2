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
/// \brief class for entropy encoding/decoding of FV0 digits data

#ifndef O2_FV0_CTFCODER_H
#define O2_FV0_CTFCODER_H

#include <algorithm>
#include <iterator>
#include <string>
#include "FV0Base/Geometry.h"
#include "DataFormatsFV0/CTF.h"
#include "DataFormatsFV0/BCData.h"
#include "DataFormatsFV0/ChannelData.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsBase/CTFCoderBase.h"
#include "rANS/rans.h"

class TTree;

namespace o2
{
namespace fv0
{

class CTFCoder : public o2::ctf::CTFCoderBase
{
 public:
  CTFCoder() : o2::ctf::CTFCoderBase(CTF::getNBlocks(), o2::detectors::DetID::FV0) {}
  ~CTFCoder() = default;

  /// entropy-encode digits to buffer with CTF
  template <typename VEC>
  void encode(VEC& buff, const gsl::span<const BCData>& digitVec, const gsl::span<const ChannelData>& channelVec);

  /// entropy decode clusters from buffer with CTF
  template <typename VDIG, typename VCHAN>
  void decode(const CTF::base& ec, VDIG& digitVec, VCHAN& channelVec);

  void createCoders(const std::string& dictPath, o2::ctf::CTFCoderBase::OpType op);

 private:
  /// compres digits clusters to CompressedDigits
  void compress(CompressedDigits& cd, const gsl::span<const BCData>& digitVec, const gsl::span<const ChannelData>& channelVec);
  size_t estimateCompressedSize(const CompressedDigits& cc);

  /// decompress CompressedDigits to digits
  template <typename VDIG, typename VCHAN>
  void decompress(const CompressedDigits& cd, VDIG& digitVec, VCHAN& channelVec);

  void appendToTree(TTree& tree, CTF& ec);
  void readFromTree(TTree& tree, int entry, std::vector<BCData>& digitVec, std::vector<ChannelData>& channelVec);

  ClassDefNV(CTFCoder, 1);
};

/// entropy-encode clusters to buffer with CTF
template <typename VEC>
void CTFCoder::encode(VEC& buff, const gsl::span<const BCData>& digitVec, const gsl::span<const ChannelData>& channelVec)
{
  using MD = o2::ctf::Metadata::OptStore;
  // what to do which each field: see o2::ctd::Metadata explanation
  constexpr MD optField[CTF::getNBlocks()] = {
    MD::EENCODE, // BLC_bcInc
    MD::EENCODE, // BLC_orbitInc
    MD::EENCODE, // BLC_nChan

    MD::EENCODE, // BLC_idChan
    MD::EENCODE, // BLC_time
    MD::EENCODE  // BLC_charge
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
#define ENCODEFV0(part, slot, bits) CTF::get(buff.data())->encode(part, int(slot), bits, optField[int(slot)], &buff, mCoders[int(slot)].get());
  // clang-format off
  ENCODEFV0(cd.bcInc,     CTF::BLC_bcInc,    0);
  ENCODEFV0(cd.orbitInc,  CTF::BLC_orbitInc, 0);
  ENCODEFV0(cd.nChan,     CTF::BLC_nChan,    0);

  ENCODEFV0(cd.idChan ,   CTF::BLC_idChan,   0);
  ENCODEFV0(cd.time,      CTF::BLC_time,     0);
  ENCODEFV0(cd.charge,    CTF::BLC_charge,   0);
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
#define DECODEFV0(part, slot) ec.decode(part, int(slot), mCoders[int(slot)].get())
  // clang-format off
  DECODEFV0(cd.bcInc,     CTF::BLC_bcInc); 
  DECODEFV0(cd.orbitInc,  CTF::BLC_orbitInc);
  DECODEFV0(cd.nChan,     CTF::BLC_nChan);

  DECODEFV0(cd.idChan,    CTF::BLC_idChan);
  DECODEFV0(cd.time,      CTF::BLC_time);
  DECODEFV0(cd.charge,    CTF::BLC_charge);
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

    firstEntry = channelVec.size();
    uint8_t chID = 0;
    for (uint8_t ic = 0; ic < cd.nChan[idig]; ic++) {
      auto icc = channelVec.size();
      const auto& chan = channelVec.emplace_back((chID += cd.idChan[icc]), cd.time[icc], cd.charge[icc]);
    }
    Triggers triggers; // TODO: Actual values are not set
    digitVec.emplace_back(firstEntry, cd.nChan[idig], ir, triggers);
  }
}

} // namespace fv0
} // namespace o2

#endif // O2_FV0_CTFCODER_H
