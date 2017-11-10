// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

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