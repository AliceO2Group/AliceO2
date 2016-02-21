#pragma once

#include "Producer.h"

#include <TTree.h>
#include <string>

class TreeProducer : public Producer
{
public:
	TreeProducer(std::string treeId);
	TObject* produceData() const override;

private:
	std::string mTreeId;
	void createBranch(TTree* tree) const;
};
