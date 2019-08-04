// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_CDB_FILEDUMP_H_
#define ALICEO2_CDB_FILEDUMP_H_

#include "CCDB/Manager.h" // for StorageFactory, StorageParameters
#include "Rtypes.h"       // for Bool_t, Int_t, ClassDef, kFALSE, etc
#include "CCDB/Storage.h" // for Storage
#include "TString.h"      // for TString

class TFile; // lines 8-8
class TList;

class TObject;

namespace o2
{
namespace ccdb
{

class Condition;

class ConditionId;

class IdRunRange;

class FileStorage : public Storage
{
  friend class FileStorageFactory;

 public:
  Bool_t isReadOnly() const override
  {
    return mReadOnly;
  };

  Bool_t hasSubVersion() const override
  {
    return kFALSE;
  };

  Bool_t hasConditionType(const char* path) const override;

  Bool_t idToFilename(const ConditionId& id, TString& filename) const override;

  void setRetry(Int_t /* nretry */, Int_t /* initsec */) override;

 protected:
  Condition* getCondition(const ConditionId& query) override;

  ConditionId* getConditionId(const ConditionId& query) override;

  TList* getAllEntries(const ConditionId& query) override;

  Bool_t putCondition(Condition* entry, const char* mirrors = "") override;

  TList* getIdListFromFile(const char* fileName) override;

 private:
  FileStorage(const FileStorage& source);

  FileStorage& operator=(const FileStorage& source);

  FileStorage(const char* dbFile, Bool_t readOnly);

  ~FileStorage() override;

  Bool_t keyNameToId(const char* keyname, IdRunRange& runRange, Int_t& version, Int_t& subVersion);

  Bool_t idToKeyName(const IdRunRange& runRange, Int_t version, Int_t subVersion, TString& keyname);

  Bool_t makeDir(const TString& dir);

  Bool_t prepareId(ConditionId& id);

  //	Bool_t getId(const  ConditionId& query,  ConditionId& result);
  ConditionId* getId(const ConditionId& query);

  void queryValidFiles() override;

  void getEntriesForLevel0(const ConditionId& query, TList* result);

  void getEntriesForLevel1(const ConditionId& query, TList* result);

  TFile* mFile;     // FileStorage file
  Bool_t mReadOnly; // ReadOnly flag

  ClassDefOverride(FileStorage, 0);
};

/////////////////////////////////////////////////////////////////////
//                                                                 //
//  class  FileStorageFactory					   //
//                                                                 //
/////////////////////////////////////////////////////////////////////

class FileStorageFactory : public StorageFactory
{

 public:
  Bool_t validateStorageUri(const char* dbString) override;

  StorageParameters* createStorageParameter(const char* dbString) override;

 protected:
  Storage* createStorage(const StorageParameters* param) override;

  ClassDefOverride(FileStorageFactory, 0);
};

/////////////////////////////////////////////////////////////////////
//                                                                 //
//  class  FileStorageParameters					   //
//                                                                 //
/////////////////////////////////////////////////////////////////////

class FileStorageParameters : public StorageParameters
{

 public:
  FileStorageParameters();

  FileStorageParameters(const char* dbPath, Bool_t readOnly = kFALSE);

  ~FileStorageParameters() override;

  const TString& getPathString() const
  {
    return mDBPath;
  };

  Bool_t isReadOnly() const
  {
    return mReadOnly;
  };

  StorageParameters* cloneParam() const override;

  virtual ULong_t getHash() const;

  virtual Bool_t isEqual(const TObject* obj) const;

 private:
  TString mDBPath;  // FileStorage file path name
  Bool_t mReadOnly; // ReadOnly flag

  ClassDefOverride(FileStorageParameters, 0);
};
} // namespace ccdb
} // namespace o2
#endif
