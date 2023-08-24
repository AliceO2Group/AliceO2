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
/// \brief class for entropy encoding/decoding of MID  data

#include "MIDCTF/CTFCoder.h"
#include "CommonUtils/StringUtils.h"
#include <TTree.h>

using namespace o2::mid;

///___________________________________________________________________________________
// Register encoded data in the tree (Fill is not called, will be done by caller)
void CTFCoder::appendToTree(TTree& tree, CTF& ec)
{
  ec.appendToTree(tree, mDet.getName());
}

///___________________________________________________________________________________
// extract and decode data from the tree
void CTFCoder::readFromTree(TTree& tree, int entry, std::array<std::vector<ROFRecord>, NEvTypes>& rofVec,
                            std::array<std::vector<ColumnData>, NEvTypes>& colVec)
{
  assert(entry >= 0 && entry < tree.GetEntries());
  CTF ec;
  ec.readFromTree(tree, mDet.getName(), entry);
  decode(ec, rofVec, colVec);
}

///________________________________
void CTFCoder::createCoders(const std::vector<char>& bufVec, o2::ctf::CTFCoderBase::OpType op)
{
  const auto ctf = CTF::getImage(bufVec.data());
  // just to get types
  int16_t bcInc = 0;
  int32_t orbitInc = 0;
  uint16_t entries = 0, pattern = 0;
  uint8_t evType = 0, deId = 0, colId = 0;
#define MAKECODER(part, slot) createCoder(op, std::get<rans::RenormedDenseHistogram<decltype(part)>>(ctf.getDictionary<decltype(part)>(slot, mANSVersion)), int(slot))
  // clang-format off
  MAKECODER(bcInc,    CTF::BLC_bcIncROF);
  MAKECODER(orbitInc, CTF::BLC_orbitIncROF);
  MAKECODER(entries,  CTF::BLC_entriesROF);
  MAKECODER(evType,   CTF::BLC_evtypeROF);
  MAKECODER(pattern,  CTF::BLC_pattern);
  MAKECODER(deId,     CTF::BLC_deId);
  MAKECODER(colId,    CTF::BLC_colId);
  // clang-format on
}
