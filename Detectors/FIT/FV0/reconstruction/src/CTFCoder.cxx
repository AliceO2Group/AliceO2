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
///___________________________________________________________________________________
void CTFCoder::assignDictVersion(o2::ctf::CTFDictHeader& h) const
{
  if (mExtHeader.isValidDictTimeStamp()) {
    h = mExtHeader;
  } else {
    h.majorVersion = 1;
    h.minorVersion = 1;
  }
}

///________________________________
void CTFCoder::createCoders(const std::vector<char>& bufVec, o2::ctf::CTFCoderBase::OpType op)
{
  const auto ctf = CTF::getImage(bufVec.data());
  CompressedDigits cd; // just to get member types
#define MAKECODER(part, slot) createCoder(op, ctf.getFrequencyTable<decltype(part)::value_type>(slot), int(slot))
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
  using namespace o2::ctf::ctfCoderBaseImpl;

  size_t sz = 0;
  // RS FIXME this is very crude estimate, instead, an empirical values should be used
  sz += estimateSize(mCoders[static_cast<int>(CTF::BLC_trigger)].get(), cd.trigger);
  sz += estimateSize(mCoders[static_cast<int>(CTF::BLC_bcInc)].get(), cd.bcInc);
  sz += estimateSize(mCoders[static_cast<int>(CTF::BLC_orbitInc)].get(), cd.orbitInc);
  sz += estimateSize(mCoders[static_cast<int>(CTF::BLC_nChan)].get(), cd.nChan);

  sz += estimateSize(mCoders[static_cast<int>(CTF::BLC_idChan)].get(), cd.idChan);
  sz += estimateSize(mCoders[static_cast<int>(CTF::BLC_qtcChain)].get(), cd.qtcChain);
  sz += estimateSize(mCoders[static_cast<int>(CTF::BLC_cfdTime)].get(), cd.cfdTime);
  sz += estimateSize(mCoders[static_cast<int>(CTF::BLC_qtcAmpl)].get(), cd.qtcAmpl);

  LOG(info) << "Estimated output size is " << sz << " bytes";
  return sz;
};