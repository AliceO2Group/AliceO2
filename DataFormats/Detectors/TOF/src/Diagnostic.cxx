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

int Diagnostic::getFrequency(ULong64_t pattern)
{
  auto pairC = mVector.find(pattern);
  if (pairC != mVector.end()) {
    return (pairC->second);
  }

  return 0;
}

void Diagnostic::print()
{
  std::cout << "Diagnosti patterns" << std::endl;
  for (const auto& [key, value] : mVector) {
    std::cout << key << " = " << value << "; ";
  }
  std::cout << std::endl;
}

ULong64_t Diagnostic::getEmptyCrateKey(int crate)
{
  ULong64_t key = (ULong64_t(11) << 32) + (ULong64_t(crate) << 36); // slot=11 means empty crate
  return key;
}
