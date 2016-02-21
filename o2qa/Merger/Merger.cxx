#include "Merger.h"
#include <FairMQLogger.h>
#include <TClass.h>

using namespace std;

TObject* Merger::mergeObject(TObject* object)
{
    TCollection* currentHistogramsList = addReceivedObjectToMapByName(object);
    TObject* mergedHistogram = mergeObjectWithGivenCollection(object, currentHistogramsList);
    return mergedHistogram;
}

TCollection* Merger::addReceivedObjectToMapByName(TObject* receivedObject)
{
    auto foundList = mHistogramIdTohistogramMap.find(receivedObject->GetName());

    if (foundList != mHistogramIdTohistogramMap.end()) {
        foundList->second->Add(receivedObject);
        return foundList->second.get();
    }
    else {   
        auto newItemIterator = mHistogramIdTohistogramMap.insert(make_pair(receivedObject->GetName(),
                                                                           make_shared<TList>()));
        newItemIterator.first->second->SetOwner();
        return newItemIterator.first->second.get();
    }
}

TObject* Merger::mergeObjectWithGivenCollection(TObject* object, TCollection* mergeList) 
{
    TObject* mergedObject = object->Clone(object->GetName());

    if (!mergedObject->IsA()->GetMethodWithPrototype("Merge", "TCollection*"))
    {
        LOG(ERROR) << "Object does not implement a merge function!";
        return nullptr;
    }
    Int_t errorCode = 0;
    TString listHargs;
    listHargs.Form("((TCollection*)0x%lx)", (ULong_t) mergeList);

    mergedObject->Execute("Merge", listHargs.Data(), &errorCode);
    if (errorCode)
    {
        LOG(ERROR) << "Error " << errorCode << "running merge!";
        return nullptr;
    }

    return mergedObject;
}

Merger::~Merger()
{
    for (auto const& entry : mHistogramIdTohistogramMap) { 
        entry.second->Delete();
    } 
}