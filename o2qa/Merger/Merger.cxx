#include "Merger.h"
#include <FairMQLogger.h>
#include <TClass.h>

using namespace std;

TObject* Merger::mergeObject(TObject* object)
{
  TCollection* currentDataObjectsList = addReceivedObjectToMapByName(object);
  return mergeObjectWithGivenCollection(object, currentDataObjectsList);
}

TCollection* Merger::addReceivedObjectToMapByName(TObject* receivedObject)
{
  auto foundList = mTitlesToDataObjectsMap.find(receivedObject->GetTitle());

  if (foundList != mTitlesToDataObjectsMap.end()) {
    foundList->second->Add(receivedObject);
    return foundList->second.get();
  }
  else {
    auto newItemIterator = mTitlesToDataObjectsMap.insert(make_pair(receivedObject->GetTitle(),
                                                                           make_shared<TList>()));
    newItemIterator.first->second->SetOwner();
    return newItemIterator.first->second.get();
  }
}

TObject* Merger::mergeObjectWithGivenCollection(TObject* object, TCollection* mergeList)
{
  ostringstream newName;
	newName << object->GetName() << "clone";
  TObject* mergedObject = object->Clone(newName.str().c_str());

  if (!mergedObject->IsA()->GetMethodWithPrototype("Merge", "TCollection*")) {
    LOG(ERROR) << "Object does not implement a merge function!";
    return nullptr;
  }

  Int_t errorCode = 0;
  TString listHargs;
  listHargs.Form("((TCollection*)0x%lx)", (ULong_t) mergeList);

  mergedObject->Execute("Merge", listHargs.Data(), &errorCode);
  if (errorCode) {
    LOG(ERROR) << "Error " << errorCode << "running merge!";
    return nullptr;
  }

  return mergedObject;
}

Merger::~Merger()
{
  for (auto const& entry : mTitlesToDataObjectsMap) {
    entry.second->Delete();
  }
}
