#pragma once

#include <chrono>
#include <unordered_map>

#include <TList.h>

namespace o2
{
namespace qc
{
typedef std::unordered_map<std::string, TList*> TCollectionMap;

class Merger
{
 public:
  Merger(const int numberOfQCOgbjectForCompleteData);
  virtual ~Merger();

  TObject* mergeObject(TObject* object);
  TObject* mergeObjectWithGivenCollection(TObject* object);
  void addReceivedObjectToMapByName(TObject* receivedObject);
  double getMergeTime();
  void dumpObjectsCollectionToFile(const char* title);
  void eraseCollection(const char* title);

 private:
  TCollectionMap mTitlesToDataObjectsMap;
  std::chrono::microseconds mMergeTime{ 0 };
  unsigned int mNumberOfDumpedObjects{ 0 };
  const int NUMBER_OF_QC_OBJECTS_FOR_COMPLETE_DATA;
};
}
}