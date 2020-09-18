// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigitReader.cxx
/// \brief Implementation of EMCAL cell/digit reader

#include "CommonUtils/RootChain.h"
#include "EMCALReconstruction/DigitReader.h"
#include "FairLogger.h" // for LOG

using namespace o2::emcal;
using o2::emcal::Cell;
using o2::emcal::Digit;

//______________________________________________________________________________
template <class InputType>
void DigitReader<InputType>::openInput(const std::string fileName)
{
  clear();
  if (!(mInputTree = o2::utils::RootChain::load("o2sim", fileName))) {
    LOG(FATAL) << "Failed to load cells/digits tree from " << fileName;
  }

  if constexpr (std::is_same<InputType, Digit>::value) {
    mInputTree->SetBranchAddress("EMCALDigit", &mInputArray);
    if (!mInputArray) {
      LOG(FATAL) << "Failed to find EMCALDigit branch in the " << mInputTree->GetName()
                 << " from file " << fileName;
    }
    mInputTree->SetBranchAddress("EMCALDigitTRGR", &mTriggerArray);
    if (!mTriggerArray) {
      LOG(FATAL) << "Failed to find TriggerRecords branch in the " << mInputTree->GetName()
                 << " from file " << fileName;
    }
  } else if constexpr (std::is_same<InputType, Cell>::value) {
    mInputTree->SetBranchAddress("EMCALCell", &mInputArray);
    if (!mInputArray) {
      LOG(FATAL) << "Failed to find EMCALCell branch in the " << mInputTree->GetName()
                 << " from file " << fileName;
    }
    mInputTree->SetBranchAddress("EMCALCellTRGR", &mTriggerArray);
    if (!mTriggerArray) {
      LOG(FATAL) << "Failed to find TriggerRecords branch in the " << mInputTree->GetName()
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
