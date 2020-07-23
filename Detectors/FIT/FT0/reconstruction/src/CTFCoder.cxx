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
/// \brief class for entropy encoding/decoding of FT0 digits data

#include "FT0Reconstruction/CTFCoder.h"
#include "CommonUtils/StringUtils.h"
#include <TTree.h>

using namespace o2::ft0;

///___________________________________________________________________________________
// Register encoded data in the tree (Fill is not called, will be done by caller)
void CTFCoder::appendToTree(TTree& tree, CTF& ec)
{
  ec.appendToTree(tree, o2::detectors::DetID::getName(o2::detectors::DetID::FT0));
}

///___________________________________________________________________________________
// extract and decode data from the tree
void CTFCoder::readFromTree(TTree& tree, int entry,
                            std::vector<Digit>& digitVec, std::vector<ChannelData>& channelVec)
{
  assert(entry >= 0 && entry < tree.GetEntries());
  CTF ec;
  ec.readFromTree(tree, o2::detectors::DetID::getName(o2::detectors::DetID::FT0), entry);
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
  cd.header.firstOrbit = dig0.getOrbit();
  cd.header.firstBC = dig0.getBC();

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
    cd.trigger[idig] = digit.getTriggers().triggersignals;
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
      LOG(ERROR) << "Digits with no channels";
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
