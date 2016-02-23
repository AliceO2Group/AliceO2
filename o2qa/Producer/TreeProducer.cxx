#include <TBranch.h>
#include <TRandom.h>

#include "TreeProducer.h"

using namespace std;

TreeProducer::TreeProducer(string treeId)
{
	mTreeId = treeId;
}

TObject* TreeProducer::produceData()
{
  TTree* tree = new TTree(mTreeId.c_str(), "TestTree");
  return tree;
}

void TreeProducer::createBranch(TTree* tree) const
{
	Float_t new_v;
	TBranch *newBranch = tree->Branch("new_v", &new_v, "new_v/F");

	for (int i = 0; i < 10; ++i) {
		new_v = gRandom->Gaus(0, 1);
		newBranch->Fill();
	}

	tree->AddBranchToCache(newBranch);
}
