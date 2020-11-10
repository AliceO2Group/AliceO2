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
/// \author ruben.shahoyan@cern.ch
/// \brief class for entropy encoding/decoding of FDD digits data

#include "FDDReconstruction/CTFCoder.h"
#include "CommonUtils/StringUtils.h"
#include <TTree.h>

using namespace o2::fdd;

///___________________________________________________________________________________
// Register encoded data in the tree (Fill is not called, will be done by caller)
void CTFCoder::appendToTree(TTree& tree, CTF& ec)
{
  ec.appendToTree(tree, mDet.getName());
}

///___________________________________________________________________________________
// extract and decode data from the tree
void CTFCoder::readFromTree(TTree& tree, int entry,
                            std::vector<Digit>& digitVec, std::vector<ChannelData>& channelVec)
{
  assert(entry >= 0 && entry < tree.GetEntries());
  CTF ec;
  ec.readFromTree(tree, mDet.getName(), entry);
  decode(ec, digitVec, channelVec);
}

///________________________________
void CTFCoder::compress(CompressedDigits& cd, const gsl::span<const Digit>& digitVec, const gsl::span<const ChannelData>& channelVec)
{
  // convert digits/channel to their compressed version
  cd.clear();
  if (!digitVec.size()) {
    return;
  }
  const auto& dig0 = digitVec[0];
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
    const auto& digit = digitVec[idig];
    const auto chanels = digit.getBunchChannelData(channelVec); // we assume the channels are sorted

    // fill trigger info
    cd.trigger[idig] = digit.mTriggers.triggersignals;
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
      LOG(ERROR) << "Digits with no channels";
      continue;
    }
    uint8_t prevChan = 0;
    for (uint8_t ic = 0; ic < cd.nChan[idig]; ic++) {
      assert(prevChan <= chanels[ic].mPMNumber);
      cd.idChan[ccount] = chanels[ic].mPMNumber - prevChan;
      cd.time[ccount] = chanels[ic].mTime;        // make sure it fits to short!!!
      cd.charge[ccount] = chanels[ic].mChargeADC; // make sure we really need short!!!
      cd.feeBits[ccount] = chanels[ic].mFEEBits;
      prevChan = chanels[ic].mPMNumber;
      ccount++;
    }
  }
}

///________________________________
void CTFCoder::createCoders(const std::string& dictPath, o2::ctf::CTFCoderBase::OpType op)
{
  bool mayFail = true; // RS FIXME if the dictionary file is not there, do not produce exception
  auto buff = readDictionaryFromFile<CTF>(dictPath, mayFail);
  if (!buff.size()) {
    if (mayFail) {
      return;
    }
    throw std::runtime_error("Failed to create CTF dictionaty");
  }
  const auto* ctf = CTF::get(buff.data());

  auto getFreq = [ctf](CTF::Slots slot) -> o2::rans::FrequencyTable {
    o2::rans::FrequencyTable ft;
    auto bl = ctf->getBlock(slot);
    auto md = ctf->getMetadata(slot);
    ft.addFrequencies(bl.getDict(), bl.getDict() + bl.getNDict(), md.min, md.max);
    return std::move(ft);
  };
  auto getProbBits = [ctf](CTF::Slots slot) -> int {
    return ctf->getMetadata(slot).probabilityBits;
  };

  CompressedDigits cd; // just to get member types
#define MAKECODER(part, slot) createCoder<decltype(part)::value_type>(op, getFreq(slot), getProbBits(slot), int(slot))
  // clang-format off
  MAKECODER(cd.trigger,   CTF::BLC_trigger);
  MAKECODER(cd.bcInc,     CTF::BLC_bcInc); 
  MAKECODER(cd.orbitInc,  CTF::BLC_orbitInc);
  MAKECODER(cd.nChan,     CTF::BLC_nChan);

  MAKECODER(cd.idChan,    CTF::BLC_idChan);
  MAKECODER(cd.time,      CTF::BLC_time);
  MAKECODER(cd.charge,    CTF::BLC_charge);
  MAKECODER(cd.feeBits,   CTF::BLC_feeBits);
  // clang-format on
}

///________________________________
size_t CTFCoder::estimateCompressedSize(const CompressedDigits& cd)
{
  size_t sz = 0;
  // clang-format off
  // RS FIXME this is very crude estimate, instead, an empirical values should be used
#define VTP(vec) typename std::remove_reference<decltype(vec)>::type::value_type
#define ESTSIZE(vec, slot) mCoders[int(slot)] ?                         \
  rans::calculateMaxBufferSize(vec.size(), reinterpret_cast<const o2::rans::LiteralEncoder64<VTP(vec)>*>(mCoders[int(slot)].get())->getAlphabetRangeBits(), sizeof(VTP(vec)) ) : vec.size()*sizeof(VTP(vec))
  sz += ESTSIZE(cd.trigger,   CTF::BLC_trigger);
  sz += ESTSIZE(cd.bcInc,     CTF::BLC_bcInc); 
  sz += ESTSIZE(cd.orbitInc,  CTF::BLC_orbitInc);
  sz += ESTSIZE(cd.nChan,     CTF::BLC_nChan);

  sz += ESTSIZE(cd.idChan,    CTF::BLC_idChan);
  sz += ESTSIZE(cd.time,      CTF::BLC_time);
  sz += ESTSIZE(cd.charge,    CTF::BLC_charge);
  sz += ESTSIZE(cd.feeBits,   CTF::BLC_feeBits);
  // clang-format on

  LOG(INFO) << "Estimated output size is " << sz << " bytes";
  return sz;
}
