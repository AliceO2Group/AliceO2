#include <FairMQLogger.h>

#include <TH1.h>
#include <TH2.h>
#include <TH3.h>
#include <THn.h>
#include <TTree.h>

#include "QCMerger/Merger.h"

using namespace std;

namespace o2
{
namespace qc
{
Merger::Merger(const int numberOfQCOgbjectForCompleteData)
  : NUMBER_OF_QC_OBJECTS_FOR_COMPLETE_DATA(numberOfQCOgbjectForCompleteData)
{
}

TObject* Merger::mergeObject(TObject* object)
{
  auto foundEntry = mTitlesToDataObjectsMap.find(object->GetTitle());

  if (foundEntry != mTitlesToDataObjectsMap.end() &&
      foundEntry->second->GetSize() >= NUMBER_OF_QC_OBJECTS_FOR_COMPLETE_DATA - 1) {
    TObject* output = mergeObjectWithGivenCollection(object);
    eraseCollection(object->GetTitle());
    return output;
  } else {
    addReceivedObjectToMapByName(object);
    return nullptr;
  }
}

void Merger::addReceivedObjectToMapByName(TObject* receivedObject)
{
  auto foundList = mTitlesToDataObjectsMap.find(receivedObject->GetTitle());

  if (foundList != mTitlesToDataObjectsMap.end()) {
    foundList->second->Add(receivedObject);
  } else {
    auto newItemIterator = mTitlesToDataObjectsMap.insert(make_pair(receivedObject->GetTitle(), new TList()));
    newItemIterator.first->second->Add(receivedObject);
  }
}

void Merger::eraseCollection(const char* title)
{
  auto foundCollection = mTitlesToDataObjectsMap.find(title);
  foundCollection->second->Delete();
  delete foundCollection->second;
  mTitlesToDataObjectsMap.erase(foundCollection);
}

void Merger::dumpObjectsCollectionToFile(const char* title)
{
  auto foundCollection = mTitlesToDataObjectsMap.find(title);

  ostringstream fileName;
  fileName << ++mNumberOfDumpedObjects << "_" << title << ".root";
  foundCollection->second->SaveAs(fileName.str().c_str());
  foundCollection->second->Delete();
  delete foundCollection->second;
  mTitlesToDataObjectsMap.erase(foundCollection);
}

TObject* Merger::mergeObjectWithGivenCollection(TObject* mergedObject)
{
  TCollection* mergeList = mTitlesToDataObjectsMap.find(mergedObject->GetTitle())->second;

  TObject* result = nullptr;
  TH1F* histogram1F = nullptr;
  TH2F* histogram2F = nullptr;
  TH3F* histogram3F = nullptr;
  THnF* histogramNF = nullptr;
  TTree* tree = nullptr;
  const char* className = mergedObject->ClassName();

  auto measureTime = chrono::high_resolution_clock::now();

  if (strcmp(className, "TH1F") == 0) {
    histogram1F = reinterpret_cast<TH1F*>(mergedObject);
    histogram1F->Merge(mergeList);
    result = histogram1F;
  } else if (strcmp(className, "TH2F") == 0) {
    histogram2F = reinterpret_cast<TH2F*>(mergedObject);
    histogram2F->Merge(mergeList);
    result = histogram2F;
  } else if (strcmp(className, "TH3F") == 0) {
    histogram3F = reinterpret_cast<TH3F*>(mergedObject);
    histogram3F->Merge(mergeList);
    result = histogram3F;
  } else if (strcmp(className, "THnT<float>") == 0) {
    histogramNF = reinterpret_cast<THnF*>(mergedObject);
    histogramNF->Merge(mergeList);
    result = histogramNF;
  } else if (strcmp(className, "TTree") == 0) {
    tree = reinterpret_cast<TTree*>(mergedObject);
    tree->Merge(mergeList);
    result = tree;
  } else {
    LOG(ERROR) << "Object with type " << className << " is not one of mergable type.";
  }

  mMergeTime = chrono::duration_cast<std::chrono::microseconds>(chrono::high_resolution_clock::now() - measureTime);

  return mergedObject;
}

double Merger::getMergeTime()
{
  return mMergeTime.count() / 1000.0; // in miliseconds
}

Merger::~Merger()
{
  for (auto const& entry : mTitlesToDataObjectsMap) {
    entry.second->Delete();
  }
}
}
}