#pragma once

#include <TCollection.h>
#include <unordered_map>
#include <memory>
#include <TList.h>

class Merger
{
public:
	virtual ~Merger();
	TObject* mergeObject(TObject* object);
	TObject* mergeObjectWithGivenCollection(TObject* object, TCollection* mergeList);
	TCollection* addReceivedObjectToMapByName(TObject* receivedObject);

private:
	std::unordered_map<std::string, std::shared_ptr<TCollection>> mTitlesToDataObjectsMap;
};
