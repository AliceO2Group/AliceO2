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
/// \brief class for entropy encoding/decoding of EMCAL data

#include "EMCALReconstruction/CTFCoder.h"
#include "CommonUtils/StringUtils.h"
#include <TTree.h>

using namespace o2::emcal;

///___________________________________________________________________________________
// Register encoded data in the tree (Fill is not called, will be done by caller)
void CTFCoder::appendToTree(TTree& tree, CTF& ec)
{
  ec.appendToTree(tree, mDet.getName());
}

///___________________________________________________________________________________
// extract and decode data from the tree
void CTFCoder::readFromTree(TTree& tree, int entry, std::vector<TriggerRecord>& trigVec, std::vector<Cell>& cellVec)
{
  assert(entry >= 0 && entry < tree.GetEntries());
  CTF ec;
  ec.readFromTree(tree, mDet.getName(), entry);
  decode(ec, trigVec, cellVec);
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

  // just to get types
  uint16_t bcInc = 0, entries = 0, cellTime = 0, energy = 0, tower = 0, trigger = 0;
  uint32_t orbitInc = 0;
  uint8_t status = 0;
#define MAKECODER(part, slot) createCoder<decltype(part)>(op, getFreq(slot), getProbBits(slot), int(slot))
  // clang-format off
  MAKECODER(bcInc,      CTF::BLC_bcIncTrig);
  MAKECODER(orbitInc,   CTF::BLC_orbitIncTrig);
  MAKECODER(entries,    CTF::BLC_entriesTrig);
  MAKECODER(tower,      CTF::BLC_towerID);
  MAKECODER(cellTime,   CTF::BLC_time);
  MAKECODER(energy,     CTF::BLC_energy);
  MAKECODER(status,     CTF::BLC_status);
  // extra slot was added in the end
  MAKECODER(trigger,    CTF::BLC_trigger);
  // clang-format on
}
