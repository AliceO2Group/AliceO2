// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <sstream>

#include <TBranch.h>
#include <TFile.h>
#include <TRandom.h>

#include "QCProducer/TreeProducer.h"

using namespace std;

namespace o2
{
namespace qc
{
TreeProducer::TreeProducer(const char* treeName, const char* treeTitle, const int numberOfBranches,
                           const int numberOfEntriesInEachBranch)
  : mTreeName(treeName),
    mTreeTitle(treeTitle),
    mNumberOfBranches(numberOfBranches),
    mNumberOfEntriesInEachBranch(numberOfEntriesInEachBranch)
{
}

TObject* TreeProducer::produceData() const
{
  auto* tree = new TTree(mTreeName.c_str(), mTreeTitle.c_str());

  for (int i = 0; i < mNumberOfBranches; ++i) {
    createBranch(tree, i);
  }

  return tree;
}

void TreeProducer::createBranch(TTree* tree, int brunchNumber, const char* branchNamePrefix) const
{
  Float_t new_v;
  ostringstream branchName;

  branchName << branchNamePrefix << brunchNumber;
  tree->Branch(branchName.str().c_str(), &new_v, "new_values");

  for (int i = 0; i < mNumberOfEntriesInEachBranch; ++i) {
    new_v = gRandom->Gaus(0, 1);
    tree->Fill();
  }
}
}
}
