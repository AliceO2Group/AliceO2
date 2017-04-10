#pragma once

#include <TTree.h>

#include "QCProducer/Producer.h"

namespace o2
{
namespace qc
{
class TreeProducer : public Producer
{
 public:
  TreeProducer(const char* treeName, const char* treeTitle, const int numberOfBranches,
               const int numberOfEntriesInEachBranch);
  TObject* produceData() const override;

 private:
  const char* mTreeName;
  const char* mTreeTitle;
  const int mNumberOfBranches;
  const int mNumberOfEntriesInEachBranch;

  void createBranch(TTree* tree, int brunchNumber, const char* branchNamePrefix = "default_branch_name_") const;
};
}
}