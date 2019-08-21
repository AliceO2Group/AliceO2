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
/// \brief Implementation of EMCAL digit reader

#include "CommonUtils/RootChain.h"
#include "EMCALReconstruction/DigitReader.h"
#include "FairLogger.h" // for LOG

using namespace o2::emcal;
using o2::emcal::Digit;

//______________________________________________________________________________
void DigitReader::openInput(const std::string fileName)
{
  clear();
  if (!(mInputTree = o2::utils::RootChain::load("o2sim", fileName))) {
    LOG(FATAL) << "Failed to load digits tree from " << fileName;
  }
  mInputTree->SetBranchAddress("EMCALDigit", &mDigitArray);
  if (!mDigitArray) {
    LOG(FATAL) << "Failed to find EMCALDigit branch in the " << mInputTree->GetName()
               << " from file " << fileName;
  }
}

//______________________________________________________________________________
bool DigitReader::readNextEntry()
{
  // Load next entry from the self-managed input

  if (mCurrentEntry >= mInputTree->GetEntriesFast())
    return false;

  mInputTree->GetEntry(mCurrentEntry);
  mCurrentEntry++;
  return true;

  /*
  mCurrentEntry
  auto nev = mInputTree->GetEntries();
  if (nev != 1) {
    LOG(FATAL) << "In the self-managed mode the digits trees must have 1 entry only";
  }
  auto evID = mInputTree->GetReadEntry();
  if (evID < -1)
    evID = -1;
  if (++evID < nev) {
    mInputTree->GetEntry(evID);
    return true;
  } else {
    return false;
  }
  */
}

//______________________________________________________________________________
void DigitReader::clear()
{
  // clear data structures
  mInputTree.reset(); // here we reset the unique ptr, not the tree!
}
