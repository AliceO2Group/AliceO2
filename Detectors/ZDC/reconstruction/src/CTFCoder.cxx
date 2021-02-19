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
void CTFCoder::readFromTree(TTree& tree, int entry, std::vector<BCData>& trigVec, std::vector<ChannelData>& chanVec, std::vector<PedestalData>& pedVec)
{
  assert(entry >= 0 && entry < tree.GetEntries());
  CTF ec;
  ec.readFromTree(tree, mDet.getName(), entry);
  decode(ec, trigVec, chanVec, pedVec);
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
  uint16_t bcIncTrig, moduleTrig, nchanTrig, chanData, pedData, triggersHL, channelsHL;
  uint32_t orbitIncTrig, orbitIncPed;
  uint8_t extTriggers, chanID;
#define MAKECODER(part, slot) createCoder<decltype(part)>(op, getFreq(slot), getProbBits(slot), int(slot))
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
  MAKECODER(orbitIncPed,       CTF::BLC_orbitIncPed);
  MAKECODER(pedData,           CTF::BLC_pedData);
  // clang-format on
}
