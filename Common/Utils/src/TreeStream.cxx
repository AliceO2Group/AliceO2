// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//  For the functionality of TreeStream see the testTreeStream.cxx

#include "CommonUtils/TreeStream.h"
#include <TBranch.h>

using namespace o2::utils;

//_________________________________________________
TreeStream::TreeStream(const char* treename) : mTree(treename, treename)
{
  //
  // Standard ctor
}

//_________________________________________________
int TreeStream::CheckIn(Char_t type, void* pointer)
{
  // Insert object

  if (mCurrentIndex >= static_cast<int>(mElements.size())) {
    mElements.emplace_back();
    auto& element = mElements.back();
    element.type = type;
    TString name = mNextName;
    if (name.Length()) {
      if (mNextNameCounter > 0) {
        name += mNextNameCounter;
      }
    } else {
      name = TString::Format("B%d.", static_cast<int>(mElements.size()));
    }
    element.name = name.Data();
    element.ptr = pointer;
  } else {
    auto& element = mElements[mCurrentIndex];
    if (element.type != type) {
      mStatus++;
      return 1; // mismatched data element
    }
    element.ptr = pointer;
  }
  mCurrentIndex++;
  return 0;
}

//_________________________________________________
void TreeStream::BuildTree()
{
  // Build the Tree

  int entriesFilled = mTree.GetEntries();
  if (mBranches.size() < mElements.size())
    mBranches.resize(mElements.size());

  TString name;
  TBranch* br = nullptr;
  for (int i = 0; i < static_cast<int>(mElements.size()); i++) {
    //
    auto& element = mElements[i];
    if (mBranches[i])
      continue;
    name = element.name.data();
    if (name.IsNull()) {
      name = TString::Format("B%d", i);
    }
    if (element.cls) {
      br = mTree.Branch(name.Data(), element.cls->GetName(), &(element.ptr));
      mBranches[i] = br;
      if (entriesFilled) {
        br->SetAddress(nullptr);
        for (int ientry = 0; ientry < entriesFilled; ientry++) {
          br->Fill();
        }
        br->SetAddress(&(element.ptr));
      }
    }

    if (element.type > 0) {
      TString nameC = TString::Format("%s/%c", name.Data(), element.type);
      br = mTree.Branch(name.Data(), element.ptr, nameC.Data());
      if (entriesFilled) {
        br->SetAddress(nullptr);
        for (int ientry = 0; ientry < entriesFilled; ientry++) {
          br->Fill();
        }
        br->SetAddress(element.ptr);
      }
      mBranches[i] = br;
    }
  }
}

//_________________________________________________
void TreeStream::Fill()
{
  // Fill the tree

  int entries = mElements.size();
  if (entries > mTree.GetNbranches()) {
    BuildTree();
  }
  for (int i = 0; i < entries; i++) {
    auto& element = mElements[i];
    if (!element.type)
      continue;
    auto br = mBranches[i];
    if (br) {
      if (element.type)
        br->SetAddress(element.ptr);
    }
  }
  if (!mStatus)
    mTree.Fill(); // fill only in case of non conflicts
  mStatus = 0;
}

//_________________________________________________
TreeStream& TreeStream::Endl()
{
  // Perform pseudo endl operation

  if (mTree.GetNbranches() == 0)
    BuildTree();
  Fill();
  mStatus = 0;
  mCurrentIndex = 0;
  return *this;
}

//_________________________________________________
TreeStream& TreeStream::operator<<(const Char_t* name)
{
  // Stream the branch name
  //
  if (name[0] == '\n') {
    return Endl();
  }
  //
  // if tree was already defined ignore
  if (mTree.GetEntries() > 0)
    return *this;
  // check branch name if tree was not
  //
  Int_t last = 0;
  for (last = 0;; last++) {
    if (name[last] == 0)
      break;
  }

  if (last > 0 && name[last - 1] == '=') {
    mNextName = name;
    mNextName[last - 1] = 0;
    mNextNameCounter = 0;
  }
  return *this;
}
