#pragma once

#include <TTree.h>
#include <string>

#include "Producer/Producer.h"

class TreeProducer : public Producer
{
public:
	TreeProducer(std::string treeNamePrefix,
               std::string treeTitle,
               double numberOfBranches,
               double numberOfEntriesInEachBranch);
	TObject* produceData() override;

private:
	std::string mTreeNamePrefix;
  std::string mTreeTitle;
  double mNumberOfBranches;
  double mNumberOfEntriesInEachBranch;
  int mProducedTreeNumber;

	void createBranch(TTree* tree, int brunchNumber) const;
};
