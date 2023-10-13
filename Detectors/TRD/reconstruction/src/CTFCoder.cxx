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
/// \brief class for entropy encoding/decoding of TRD data

#include "TRDReconstruction/CTFCoder.h"
#include "CommonUtils/StringUtils.h"
#include <TTree.h>

using namespace o2::trd;

///___________________________________________________________________________________
// Register encoded data in the tree (Fill is not called, will be done by caller)
void CTFCoder::appendToTree(TTree& tree, CTF& ec)
{
  ec.appendToTree(tree, mDet.getName());
}

///___________________________________________________________________________________
// extract and decode data from the tree
void CTFCoder::readFromTree(TTree& tree, int entry, std::vector<TriggerRecord>& trigVec, std::vector<Tracklet64>& trkVec, std::vector<Digit>& digVec)
{
  assert(entry >= 0 && entry < tree.GetEntries());
  CTF ec;
  ec.readFromTree(tree, mDet.getName(), entry);
  decode(ec, trigVec, trkVec, digVec);
}

///________________________________
void CTFCoder::createCoders(const std::vector<char>& bufVec, o2::ctf::CTFCoderBase::OpType op)
{
  const auto ctf = CTF::getImage(bufVec.data());
  // just to get types
  int16_t bcInc;
  int32_t orbitInc;
  uint16_t HCIDTrk, posTrk, CIDDig, ADCDig;
  uint32_t entriesTrk, entriesDig, pidTrk;
  uint8_t padrowTrk, colTrk, slopeTrk, ROBDig, MCMDig, chanDig;

#define MAKECODER(part, slot) createCoder(op, std::get<rans::RenormedDenseHistogram<decltype(part)>>(ctf.getDictionary<decltype(part)>(slot, mANSVersion)), int(slot))
  // clang-format off
  MAKECODER(bcInc,      CTF::BLC_bcIncTrig);
  MAKECODER(orbitInc,   CTF::BLC_orbitIncTrig);
  MAKECODER(entriesTrk, CTF::BLC_entriesTrk);
  MAKECODER(entriesDig, CTF::BLC_entriesDig);

  MAKECODER(HCIDTrk,    CTF::BLC_HCIDTrk);
  MAKECODER(padrowTrk,  CTF::BLC_padrowTrk);
  MAKECODER(colTrk,     CTF::BLC_colTrk);
  MAKECODER(posTrk,     CTF::BLC_posTrk);
  MAKECODER(slopeTrk,   CTF::BLC_slopeTrk);
  MAKECODER(pidTrk,     CTF::BLC_pidTrk);

  MAKECODER(CIDDig,     CTF::BLC_CIDDig);
  MAKECODER(ROBDig,     CTF::BLC_ROBDig);
  MAKECODER(MCMDig,     CTF::BLC_MCMDig);
  MAKECODER(chanDig,    CTF::BLC_chanDig);
  MAKECODER(ADCDig,     CTF::BLC_ADCDig);
  // clang-format on
}
