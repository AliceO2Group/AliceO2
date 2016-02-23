#pragma once

#include <TTree.h>
#include <string>

#include "Producer.h"

class TreeProducer : public Producer
{
public:
	TreeProducer(std::string treeId);
	TObject* produceData() override;

private:
	std::string mTreeId;
	void createBranch(TTree* tree) const;
};
