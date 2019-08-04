// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_CDB_LOCAL_H_
#define ALICEO2_CDB_LOCAL_H_

//  class  LocalStorage						   //
//  access class to a DataBase in a local storage                  //
#include "CCDB/Manager.h" // for StorageFactory, StorageParameters
#include "Rtypes.h"       // for Bool_t, Int_t, ClassDef, LocalStorage::Class, etc
#include "CCDB/Storage.h" // for Storage
#include "TString.h"      // for TString

class TList;

class TObject;

namespace o2
{
namespace ccdb
{

class Condition;

class ConditionId;

class IdRunRange;

class LocalStorage : public Storage
{
  friend class LocalStorageFactory;

 public:
  Bool_t isReadOnly() const override
  {
    return kFALSE;
  };

  Bool_t hasSubVersion() const override
  {
    return kTRUE;
  };

  Bool_t hasConditionType(const char* path) const override;

  Bool_t idToFilename(const ConditionId& id, TString& filename) const override;

  void setRetry(Int_t /* nretry */, Int_t /* initsec */) override;

 protected:
  Condition* getCondition(const ConditionId& queryId) override;

  ConditionId* getConditionId(const ConditionId& queryId) override;

  TList* getAllEntries(const ConditionId& queryId) override;

  Bool_t putCondition(Condition* entry, const char* mirrors = "") override;

  TList* getIdListFromFile(const char* fileName) override;

 private:
  LocalStorage(const LocalStorage& source);

  LocalStorage& operator=(const LocalStorage& source);

  LocalStorage(const char* baseDir);

  ~LocalStorage() override;

  Bool_t filenameToId(const char* filename, IdRunRange& runRange, Int_t& version, Int_t& subVersion);

  Bool_t prepareId(ConditionId& id);

  //	Bool_t getId(const  ConditionId& query,  ConditionId& result);
  ConditionId* getId(const ConditionId& query);

  void queryValidFiles() override;

  void queryValidCVMFSFiles(TString& cvmfsOcdbTag);

  void getEntriesForLevel0(const char* level0, const ConditionId& query, TList* result);

  void getEntriesForLevel1(const char* level0, const char* Level1, const ConditionId& query, TList* result);

  TString mBaseDirectory; // path of the DB folder

  ClassDefOverride(LocalStorage, 0); // access class to a DataBase in a local storage
};

//  class  LocalStorageFactory
class LocalStorageFactory : public StorageFactory
{
 public:
  Bool_t validateStorageUri(const char* dbString) override;

  StorageParameters* createStorageParameter(const char* dbString) override;

 protected:
  Storage* createStorage(const StorageParameters* param) override;

  ClassDefOverride(LocalStorageFactory, 0);
};

//  class  LocalStorageParameters
class LocalStorageParameters : public StorageParameters
{
 public:
  LocalStorageParameters();

  LocalStorageParameters(const char* dbPath);

  LocalStorageParameters(const char* dbPath, const char* uri);

  ~LocalStorageParameters() override;

  const TString& getPathString() const
  {
    return mDBPath;
  };

  StorageParameters* cloneParam() const override;

  virtual ULong_t getHash() const;

  virtual Bool_t isEqual(const TObject* obj) const;

 private:
  TString mDBPath; // path of the DB folder
  ClassDefOverride(LocalStorageParameters, 0);
};
} // namespace ccdb
} // namespace o2
#endif
