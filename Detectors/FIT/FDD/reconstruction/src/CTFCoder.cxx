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
#define MAKECODER(part, slot) createCoder(op, std::get<rans::RenormedDenseHistogram<typename decltype(part)::value_type>>(ctf.getDictionary<decltype(part)::value_type>(slot, mANSVersion)), int(slot))
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
  // RS FIXME this is very crude estimate, instead, an empirical values should be used
  sz += estimateBufferSize(static_cast<int>(CTF::BLC_trigger), cd.trigger);
  sz += estimateBufferSize(static_cast<int>(CTF::BLC_bcInc), cd.bcInc);
  sz += estimateBufferSize(static_cast<int>(CTF::BLC_orbitInc), cd.orbitInc);
  sz += estimateBufferSize(static_cast<int>(CTF::BLC_nChan), cd.nChan);

  sz += estimateBufferSize(static_cast<int>(CTF::BLC_idChan), cd.idChan);
  sz += estimateBufferSize(static_cast<int>(CTF::BLC_time), cd.time);
  sz += estimateBufferSize(static_cast<int>(CTF::BLC_charge), cd.charge);
  sz += estimateBufferSize(static_cast<int>(CTF::BLC_feeBits), cd.feeBits);

  LOG(info) << "Estimated output size is " << sz << " bytes";
  return sz;
}
