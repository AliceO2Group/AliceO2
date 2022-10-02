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

/// \file DigitReader.cxx
/// \brief Implementation of EMCAL cell/digit reader

#include "CommonUtils/RootChain.h"
#include "EMCALReconstruction/DigitReader.h"
#include <fairlogger/Logger.h> // for LOG

using namespace o2::emcal;
using o2::emcal::Cell;
using o2::emcal::Digit;

//______________________________________________________________________________
template <class InputType>
void DigitReader<InputType>::openInput(const std::string fileName)
{
  clear();
  if (!(mInputTree = o2::utils::RootChain::load("o2sim", fileName))) {
    LOG(fatal) << "Failed to load cells/digits tree from " << fileName;
  }

  if constexpr (std::is_same<InputType, Digit>::value) {
    mInputTree->SetBranchAddress("EMCALDigit", &mInputArray);
    if (!mInputArray) {
      LOG(fatal) << "Failed to find EMCALDigit branch in the " << mInputTree->GetName()
                 << " from file " << fileName;
    }
    mInputTree->SetBranchAddress("EMCALDigitTRGR", &mTriggerArray);
    if (!mTriggerArray) {
      LOG(fatal) << "Failed to find TriggerRecords branch in the " << mInputTree->GetName()
                 << " from file " << fileName;
    }
  } else if constexpr (std::is_same<InputType, Cell>::value) {
    mInputTree->SetBranchAddress("EMCALCell", &mInputArray);
    if (!mInputArray) {
      LOG(fatal) << "Failed to find EMCALCell branch in the " << mInputTree->GetName()
                 << " from file " << fileName;
    }
    mInputTree->SetBranchAddress("EMCALCellTRGR", &mTriggerArray);
    if (!mTriggerArray) {
      LOG(fatal) << "Failed to find TriggerRecords branch in the " << mInputTree->GetName()
                 << " from file " << fileName;
    }
  }
}

//______________________________________________________________________________
template <class InputType>
bool DigitReader<InputType>::readNextEntry()
{
  // Load next entry from the self-managed input

  if (mCurrentEntry >= mInputTree->GetEntriesFast()) {
    return false;
  }

  mInputTree->GetEntry(mCurrentEntry);
  mCurrentEntry++;
  return true;
}

//______________________________________________________________________________
template <class InputType>
void DigitReader<InputType>::clear()
{
  // clear data structures
  mInputTree.reset(); // here we reset the unique ptr, not the tree!
}

template class o2::emcal::DigitReader<o2::emcal::Cell>;
template class o2::emcal::DigitReader<o2::emcal::Digit>;
