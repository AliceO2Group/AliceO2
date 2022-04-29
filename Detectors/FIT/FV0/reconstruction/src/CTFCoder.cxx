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
/// \author ruben.shahoyan@cern.ch
/// \brief class for entropy encoding/decoding of FV0 digits data

#include "FV0Reconstruction/CTFCoder.h"
#include "FV0Simulation/FV0DigParam.h"
#include "CommonUtils/StringUtils.h"
#include <TTree.h>

using namespace o2::fv0;

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
  cd.header.det = mDet;
  cd.header.nTriggers = digitVec.size();
  cd.header.firstOrbit = dig0.getOrbit();
  cd.header.firstBC = dig0.getBC();
  cd.header.triggerGate = FV0DigParam::Instance().mTime_trg_gate;

  cd.trigger.resize(cd.header.nTriggers);
  cd.bcInc.resize(cd.header.nTriggers);
  cd.orbitInc.resize(cd.header.nTriggers);
  cd.nChan.resize(cd.header.nTriggers);

  cd.idChan.resize(channelVec.size());
  cd.qtcChain.resize(channelVec.size());
  cd.cfdTime.resize(channelVec.size());
  cd.qtcAmpl.resize(channelVec.size());

  uint16_t prevBC = cd.header.firstBC;
  uint32_t prevOrbit = cd.header.firstOrbit;
  uint32_t ccount = 0;
  for (uint32_t idig = 0; idig < cd.header.nTriggers; idig++) {
    const auto& digit = digitVec[idig];
    const auto chanels = digit.getBunchChannelData(channelVec); // we assume the channels are sorted

    // fill trigger info
    cd.trigger[idig] = digit.getTriggers().getTriggersignals();
    if (prevOrbit == digit.getOrbit()) {
      cd.bcInc[idig] = digit.getBC() - prevBC;
      cd.orbitInc[idig] = 0;
    } else {
      cd.bcInc[idig] = digit.getBC();
      cd.orbitInc[idig] = digit.getOrbit() - prevOrbit;
    }
    prevBC = digit.getBC();
    prevOrbit = digit.getOrbit();
    // fill channels info
    cd.nChan[idig] = chanels.size();
    if (!cd.nChan[idig]) {
      LOG(debug) << "Digits with no channels";
      continue;
    }
    uint8_t prevChan = 0;
    for (uint8_t ic = 0; ic < cd.nChan[idig]; ic++) {
      assert(prevChan <= chanels[ic].ChId);
      cd.idChan[ccount] = chanels[ic].ChId - prevChan;
      cd.qtcChain[ccount] = chanels[ic].ChainQTC;
      cd.cfdTime[ccount] = chanels[ic].CFDTime;
      cd.qtcAmpl[ccount] = chanels[ic].QTCAmpl;
      prevChan = chanels[ic].ChId;
      ccount++;
    }
  }
}

///________________________________
void CTFCoder::createCoders(const std::vector<char>& bufVec, o2::ctf::CTFCoderBase::OpType op)
{
  const auto ctf = CTF::getImage(bufVec.data());
  CompressedDigits cd; // just to get member types
#define MAKECODER(part, slot) createCoder<decltype(part)::value_type>(op, ctf.getFrequencyTable(slot), int(slot))
  // clang-format off
  MAKECODER(cd.bcInc,        CTF::BLC_bcInc);
  MAKECODER(cd.orbitInc,     CTF::BLC_orbitInc);
  MAKECODER(cd.nChan,        CTF::BLC_nChan);

  MAKECODER(cd.idChan,       CTF::BLC_idChan);
  MAKECODER(cd.cfdTime,      CTF::BLC_cfdTime);
  MAKECODER(cd.qtcAmpl,      CTF::BLC_qtcAmpl);
  //
  // extra slots were added in the end
  MAKECODER(cd.trigger,   CTF::BLC_trigger);
  MAKECODER(cd.qtcChain,  CTF::BLC_qtcChain);
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
  sz += ESTSIZE(cd.trigger,      CTF::BLC_trigger);
  sz += ESTSIZE(cd.bcInc,        CTF::BLC_bcInc);
  sz += ESTSIZE(cd.orbitInc,     CTF::BLC_orbitInc);
  sz += ESTSIZE(cd.nChan,        CTF::BLC_nChan);

  sz += ESTSIZE(cd.idChan,       CTF::BLC_idChan);
  sz += ESTSIZE(cd.qtcChain,     CTF::BLC_qtcChain);
  sz += ESTSIZE(cd.cfdTime,      CTF::BLC_cfdTime);
  sz += ESTSIZE(cd.qtcAmpl,      CTF::BLC_qtcAmpl);
  // clang-format on

  LOG(info) << "Estimated output size is " << sz << " bytes";
  return sz;
}
