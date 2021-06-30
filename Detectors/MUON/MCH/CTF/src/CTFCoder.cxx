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
/// \brief class for entropy encoding/decoding of MCH  data

#include "MCHCTF/CTFCoder.h"
#include "CommonUtils/StringUtils.h"
#include <TTree.h>

using namespace o2::mch;

///___________________________________________________________________________________
// Register encoded data in the tree (Fill is not called, will be done by caller)
void CTFCoder::appendToTree(TTree& tree, CTF& ec)
{
  ec.appendToTree(tree, mDet.getName());
}

///___________________________________________________________________________________
// extract and decode data from the tree
void CTFCoder::readFromTree(TTree& tree, int entry, std::vector<ROFRecord>& rofVec, std::vector<Digit>& digVec)
{
  assert(entry >= 0 && entry < tree.GetEntries());
  CTF ec;
  ec.readFromTree(tree, mDet.getName(), entry);
  decode(ec, rofVec, digVec);
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
  uint16_t bcInc, nDigits, nSamples;
  uint32_t orbitInc, ADC;
  int32_t tfTime;
  int16_t detID, padID;
  uint8_t isSaturated;
#define MAKECODER(part, slot) createCoder<decltype(part)>(op, getFreq(slot), getProbBits(slot), int(slot))

  // clang-format off
  MAKECODER(bcInc,       CTF::BLC_bcIncROF); 
  MAKECODER(orbitInc,    CTF::BLC_orbitIncROF);
  MAKECODER(nDigits,     CTF::BLC_nDigitsROF);
  MAKECODER(tfTime,      CTF::BLC_tfTime);
  MAKECODER(nSamples,    CTF::BLC_nSamples);
  MAKECODER(isSaturated, CTF::BLC_isSaturated);
  MAKECODER(detID,       CTF::BLC_detID);
  MAKECODER(padID,       CTF::BLC_padID);
  MAKECODER(ADC,         CTF::BLC_ADC);
  // clang-format on
}
