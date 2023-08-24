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
/// \brief class for entropy encoding/decoding of ZDC data

#include "ZDCReconstruction/CTFCoder.h"
#include "CommonUtils/StringUtils.h"
#include <TTree.h>

using namespace o2::zdc;

///___________________________________________________________________________________
// Register encoded data in the tree (Fill is not called, will be done by caller)
void CTFCoder::appendToTree(TTree& tree, CTF& ec)
{
  ec.appendToTree(tree, mDet.getName());
}

///___________________________________________________________________________________
// extract and decode data from the tree
void CTFCoder::readFromTree(TTree& tree, int entry, std::vector<BCData>& trigVec, std::vector<ChannelData>& chanVec, std::vector<OrbitData>& eodVec)
{
  assert(entry >= 0 && entry < tree.GetEntries());
  CTF ec;
  ec.readFromTree(tree, mDet.getName(), entry);
  decode(ec, trigVec, chanVec, eodVec);
}

///________________________________
void CTFCoder::createCoders(const std::vector<char>& bufVec, o2::ctf::CTFCoderBase::OpType op)
{
  const auto ctf = CTF::getImage(bufVec.data());
  // just to get types
  int16_t bcIncTrig, sclInc;
  int32_t orbitIncTrig, orbitIncEOD;
  uint16_t moduleTrig, nchanTrig, chanData, pedData, triggersHL, channelsHL;
  uint8_t extTriggers, chanID;
#define MAKECODER(part, slot) createCoder(op, std::get<rans::RenormedDenseHistogram<decltype(part)>>(ctf.getDictionary<decltype(part)>(slot, mANSVersion)), int(slot))
  // clang-format off
  MAKECODER(bcIncTrig,         CTF::BLC_bcIncTrig);
  MAKECODER(orbitIncTrig,      CTF::BLC_orbitIncTrig);
  MAKECODER(moduleTrig,        CTF::BLC_moduleTrig);
  MAKECODER(channelsHL,        CTF::BLC_channelsHL);
  MAKECODER(triggersHL,        CTF::BLC_triggersHL);
  MAKECODER(extTriggers,       CTF::BLC_extTriggers);
  MAKECODER(nchanTrig,         CTF::BLC_nchanTrig);
  //
  MAKECODER(chanID,            CTF::BLC_chanID);
  MAKECODER(chanData,          CTF::BLC_chanData);
  //
  MAKECODER(orbitIncEOD,       CTF::BLC_orbitIncEOD);
  MAKECODER(pedData,           CTF::BLC_pedData);
  MAKECODER(sclInc,            CTF::BLC_sclInc);
  // clang-format on
}
