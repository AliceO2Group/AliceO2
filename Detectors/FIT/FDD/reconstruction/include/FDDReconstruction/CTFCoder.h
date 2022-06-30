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
  CTFCoder(o2::ctf::CTFCoderBase::OpType op) : o2::ctf::CTFCoderBase(op, CTF::getNBlocks(), o2::detectors::DetID::FDD) {}
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
  ec->getANSHeader().majorVersion = 0;
  ec->getANSHeader().minorVersion = 1;
  // at every encoding the buffer might be autoexpanded, so we don't work with fixed pointer ec
  o2::ctf::CTFIOSize iosize;
#define ENCODEFDD(part, slot, bits) CTF::get(buff.data())->encode(part, int(slot), bits, optField[int(slot)], &buff, mCoders[int(slot)].get(), getMemMarginFactor());
  // clang-format off
  iosize += ENCODEFDD(cd.trigger,   CTF::BLC_trigger,  0);
  iosize += ENCODEFDD(cd.bcInc,     CTF::BLC_bcInc,    0);
  iosize += ENCODEFDD(cd.orbitInc,  CTF::BLC_orbitInc, 0);
  iosize += ENCODEFDD(cd.nChan,     CTF::BLC_nChan,    0);

  iosize += ENCODEFDD(cd.idChan ,   CTF::BLC_idChan,   0);
  iosize += ENCODEFDD(cd.time,      CTF::BLC_time,     0);
  iosize += ENCODEFDD(cd.charge,    CTF::BLC_charge,   0);
  iosize += ENCODEFDD(cd.feeBits,   CTF::BLC_feeBits,  0);
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
#define DECODEFDD(part, slot) ec.decode(part, int(slot), mCoders[int(slot)].get())
  // clang-format off
  iosize += DECODEFDD(cd.trigger,   CTF::BLC_trigger);
  iosize += DECODEFDD(cd.bcInc,     CTF::BLC_bcInc);
  iosize += DECODEFDD(cd.orbitInc,  CTF::BLC_orbitInc);
  iosize += DECODEFDD(cd.nChan,     CTF::BLC_nChan);

  iosize += DECODEFDD(cd.idChan,    CTF::BLC_idChan);
  iosize += DECODEFDD(cd.time,      CTF::BLC_time);
  iosize += DECODEFDD(cd.charge,    CTF::BLC_charge);
  iosize += DECODEFDD(cd.feeBits,   CTF::BLC_feeBits);
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
      const auto& chan = channelVec.emplace_back(chID, cd.time[icc], cd.charge[icc], cd.feeBits[icc]);
      // rebuild digit
      if (chan.mPMNumber > 7) { // A side
        amplA += chan.mChargeADC;
        timeA += chan.mTime;
        nChanA++;

      } else {
        amplC += chan.mChargeADC;
        timeC += chan.mTime;
        nChanC++;
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
    digitVec.emplace_back(firstEntry, cd.nChan[idig], ir, trig);
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
  cd.header.nTriggers = digitVec.size();
  cd.header.firstOrbit = dig0.mIntRecord.orbit;
  cd.header.firstBC = dig0.mIntRecord.bc;

  cd.trigger.resize(cd.header.nTriggers);
  cd.bcInc.resize(cd.header.nTriggers);
  cd.orbitInc.resize(cd.header.nTriggers);
  cd.nChan.resize(cd.header.nTriggers);

  cd.idChan.resize(channelVec.size());
  cd.time.resize(channelVec.size());
  cd.charge.resize(channelVec.size());
  cd.feeBits.resize(channelVec.size());

  uint16_t prevBC = cd.header.firstBC;
  uint32_t prevOrbit = cd.header.firstOrbit;
  uint32_t ccount = 0;
  for (uint32_t idig = 0; idig < cd.header.nTriggers; idig++) {
    if (reject[idig]) {
      continue;
    }
    const auto& digit = digitVec[idig];
    const auto chanels = digit.getBunchChannelData(channelVec); // we assume the channels are sorted

    // fill trigger info
    cd.trigger[idig] = digit.mTriggers.getTriggersignals();
    if (prevOrbit == digit.mIntRecord.orbit) {
      cd.bcInc[idig] = digit.mIntRecord.bc - prevBC;
      cd.orbitInc[idig] = 0;
    } else {
      cd.bcInc[idig] = digit.mIntRecord.bc;
      cd.orbitInc[idig] = digit.mIntRecord.orbit - prevOrbit;
    }
    prevBC = digit.mIntRecord.bc;
    prevOrbit = digit.mIntRecord.orbit;
    // fill channels info
    cd.nChan[idig] = chanels.size();
    if (!cd.nChan[idig]) {
      LOG(debug) << "Digits with no channels";
      continue;
    }
    uint8_t prevChan = 0;
    for (uint8_t ic = 0; ic < cd.nChan[idig]; ic++) {
      if constexpr (MINOR_VERSION == 0 && MAJOR_VERSION == 1) {
        cd.idChan[ccount] = chanels[ic].mPMNumber - prevChan; // Old method, lets keep it for a while
      } else {
        cd.idChan[ccount] = chanels[ic].mPMNumber;
      }
      cd.time[ccount] = chanels[ic].mTime;        // make sure it fits to short!!!
      cd.charge[ccount] = chanels[ic].mChargeADC; // make sure we really need short!!!
      cd.feeBits[ccount] = chanels[ic].mFEEBits;
      prevChan = chanels[ic].mPMNumber;
      ccount++;
    }
  }
}

} // namespace fdd
} // namespace o2

#endif // O2_FDD_CTFCODER_H
