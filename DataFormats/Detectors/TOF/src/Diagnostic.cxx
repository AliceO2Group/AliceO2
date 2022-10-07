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

/// \file Diagnostic.cxx
/// \brief Implementation of the TOF cluster

#include "DataFormatsTOF/Diagnostic.h"
#include <iostream>
#include "Framework/Logger.h"

using namespace o2::tof;

ClassImp(Diagnostic);

int Diagnostic::fill(ULong64_t pattern)
{
  int frequency = 1;

  auto pairC = mVector.find(pattern);

  if (pairC != mVector.end()) {
    frequency = (pairC->second)++;
  } else {
    mVector.emplace(std::make_pair(pattern, 1));
  }

  return frequency;
}

int Diagnostic::fill(ULong64_t pattern, int frequency)
{
  auto pairC = mVector.find(pattern);

  if (pairC != mVector.end()) {
    (pairC->second) += frequency;
    frequency = (pairC->second);
  } else {
    mVector.emplace(std::make_pair(pattern, frequency));
  }

  return frequency;
}

int Diagnostic::getFrequency(ULong64_t pattern) const
{
  auto pairC = mVector.find(pattern);
  if (pairC != mVector.end()) {
    return (pairC->second);
  }

  return 0;
}

void Diagnostic::print(bool longFormat) const
{
  LOG(info) << "Diagnostic patterns, entries = " << mVector.size();

  if (!longFormat) {
    return;
  }

  for (const auto& [key, value] : mVector) {
    std::cout << key << " = " << value << "; ";
  }
  std::cout << std::endl;
}

ULong64_t Diagnostic::getEmptyCrateKey(int crate)
{
  ULong64_t key = (ULong64_t(13) << 32) + (ULong64_t(crate) << 36); // slot=13 means empty crate
  return key;
}

ULong64_t Diagnostic::getNoisyChannelKey(int channel)
{
  ULong64_t key = (ULong64_t(14) << 32) + channel; // slot=14 means noisy channels
  return key;
}

ULong64_t Diagnostic::getTRMKey(int crate, int trm)
{
  ULong64_t key = (ULong64_t(trm) << 32) + (ULong64_t(crate) << 36);
  return key;
}

void Diagnostic::fill(const Diagnostic& diag)
{
  LOG(debug) << "Filling diagnostic word";
  for (auto const& el : diag.mVector) {
    LOG(debug) << "Filling diagnostic pattern " << el.first << " adding " << el.second << " to " << getFrequency(el.first) << " --> " << el.second + getFrequency(el.first);
    fill(el.first, el.second);
  }
}

void Diagnostic::merge(const Diagnostic* prev)
{
  LOG(debug) << "Merging diagnostic words";
  for (auto const& el : prev->mVector) {
    fill(el.first, el.second + getFrequency(el.first));
  }
}

void Diagnostic::getNoisyLevelMap(Char_t* output) const
{
  // set true in output channel array
  for (auto pair : mVector) {
    auto key = pair.first;
    int slot = getSlot(key);

    if (slot != 14) {
      continue;
    }

    output[getChannel(key)] = getNoisyLevel(key);
  }
}

void Diagnostic::getNoisyMap(Bool_t* output, int noisyThr) const
{
  // set true in output channel array
  for (auto pair : mVector) {
    auto key = pair.first;
    int slot = getSlot(key);

    if (slot != 14) {
      continue;
    }

    if (getNoisyLevel(key) >= noisyThr) {
      output[getChannel(key)] = true;
    }
  }
}

bool Diagnostic::isNoisyChannel(int channel, int thr) const
{
  static const ULong64_t addMask[3] = {0, 1 << 19, 3 << 19};
  ULong64_t mask = getNoisyChannelKey(channel);
  for (int i = thr; i <= 2; i++) {
    if (getFrequency(mask + addMask[i])) {
      return true;
    }
  }

  return false;
}
