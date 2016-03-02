#include <sstream>
#include <TBranch.h>
#include <TRandom.h>
#include <TFile.h>

#include "TreeProducer.h"

using namespace std;

TreeProducer::TreeProducer(string treeNamePrefix,
													 string treeTitle,
													 double numberOfBranches,
													 double numberOfEntriesInEachBranch) : mProducedTreeNumber(0)
{
	mTreeNamePrefix = treeNamePrefix;
	mTreeTitle = treeTitle;
	mNumberOfBranches = numberOfBranches;
	mNumberOfEntriesInEachBranch = numberOfEntriesInEachBranch;
}

TObject* TreeProducer::produceData()
{
	ostringstream treeName;
	treeName << mTreeNamePrefix << mProducedTreeNumber++;
	TTree* tree = new TTree(treeName.str().c_str(), mTreeTitle.c_str());

	for (int i = 0; i < mNumberOfBranches; ++i) {
		createBranch(tree, i);
	}

  return tree;
}

void TreeProducer::createBranch(TTree* tree, int brunchNumber) const
{
	Float_t new_v;
	string branchNamePrefix = "default_branch_name_";
	ostringstream branchName;

	branchName << branchNamePrefix << brunchNumber;
	tree->Branch(branchName.str().c_str(), &new_v, "new_values");

  for (int i = 0; i < mNumberOfEntriesInEachBranch; ++i) {
		new_v = gRandom->Gaus(0, 1);
		tree->Fill();
  }
}
