// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_CDB_MANAGER_H_
#define ALICEO2_CDB_MANAGER_H_

#include <TList.h>   // for TList
#include <TMap.h>    // for TMap
#include <TObject.h> // for TObject
#include <cstddef>   // for NULL
#include "Rtypes.h"  // for Int_t, Bool_t, kFALSE, kTRUE, ClassDef, etc
#include "TString.h" // for TString
#include <CCDB/TObjectWrapper.h>

class TFile;

///  @file   Manager.h
///  @author Raffaele Grosso
///  @since  2014-12-02
///  @brief  Adapted to o2 from the original AliCDBManager.h in AliRoot
namespace o2
{
namespace ccdb
{

class Condition;

class ConditionId;

class ConditionMetaData;

class IdPath;

class IdRunRange;

class Storage;

class StorageFactory;

class StorageParameters;

/// @class Manager
/// Steer retrieval and upload of condition objects from/to
/// different storages (local, alien, file)

class Manager : public TObject
{

 public:
  void registerFactory(StorageFactory* factory);

  Bool_t hasStorage(const char* dbString) const;

  StorageParameters* createStorageParameter(const char* dbString) const;

  Storage* getStorage(const char* dbString);

  TList* getActiveStorages();

  const TMap* getStorageMap() const
  {
    return mStorageMap;
  }

  const TList* getRetrievedIds() const
  {
    return mIds;
  }

  void setDefaultStorage(const char* dbString);

  void setDefaultStorage(const StorageParameters* param);

  void setDefaultStorage(Storage* storage);

  void setDefaultStorageFromRun(Int_t run);

  Bool_t isDefaultStorageSet() const
  {
    return mDefaultStorage != nullptr;
  }

  Storage* getDefaultStorage() const
  {
    return mDefaultStorage;
  }

  void unsetDefaultStorage();

  void setSpecificStorage(const char* calibType, const char* dbString, Int_t version = -1, Int_t subVersion = -1);

  Storage* getSpecificStorage(const char* calibType);

  void setdrainMode(const char* dbString);

  void setdrainMode(const StorageParameters* param);

  void setdrainMode(Storage* storage);

  void unsetdrainMode()
  {
    mdrainStorage = nullptr;
  }

  Bool_t isdrainSet() const
  {
    return mdrainStorage != nullptr;
  }

  Bool_t drain(Condition* entry);

  Bool_t setOcdbUploadMode();

  void unsetOcdbUploadMode()
  {
    mOcdbUploadMode = kFALSE;
  }

  Bool_t isOcdbUploadMode() const
  {
    return mOcdbUploadMode;
  }

  Condition* getCondition(const ConditionId& query, Bool_t forceCaching = kFALSE);

  Condition* getCondition(const IdPath& path, Int_t runNumber = -1, Int_t version = -1, Int_t subVersion = -1);

  Condition* getCondition(const IdPath& path, const IdRunRange& runRange, Int_t version = -1, Int_t subVersion = -1);

  Condition* getConditionFromSnapshot(const char* path);

  const char* getUri(const char* path);

  TList* getAllObjects(const ConditionId& query);

  TList* getAllObjects(const IdPath& path, Int_t runNumber = -1, Int_t version = -1, Int_t subVersion = -1);

  TList* getAllObjects(const IdPath& path, const IdRunRange& runRange, Int_t version = -1, Int_t subVersion = -1);

  Bool_t putObject(TObject* object, const ConditionId& id, ConditionMetaData* metaData, const char* mirrors = "");

  template <typename T>
  Bool_t putObjectAny(T* ptr, const ConditionId& id, ConditionMetaData* metaData, const char* mirrors = "")
  {
    TObjectWrapper<T> local(ptr);
    return putObject(&local, id, metaData, mirrors);
  }

  Bool_t putCondition(Condition* entry, const char* mirrors = "");

  void setCacheFlag(Bool_t cacheFlag)
  {
    mCache = cacheFlag;
  }

  Bool_t getCacheFlag() const
  {
    return mCache;
  }

  ULong64_t setLock(Bool_t lockFlag = kTRUE, ULong64_t key = 0);

  Bool_t getLock() const
  {
    return mLock;
  }

  void setRawFlag(Bool_t rawFlag)
  {
    mRaw = rawFlag;
  }

  Bool_t getRawFlag() const
  {
    return mRaw;
  }

  void setRun(Int_t run);

  Int_t getRun() const
  {
    return mRun;
  }

  void setMirrorSEs(const char* mirrors);

  const char* getMirrorSEs() const;

  void destroyActiveStorages();

  void destroyActiveStorage(Storage* storage);

  void queryStorages();

  void print(Option_t* option = "") const;

  static void destroy();

  ~Manager() override;

  void clearCache();

  void unloadFromCache(const char* path);

  const TMap* getConditionCache() const
  {
    return &mConditionCache;
  }

  static Manager* Instance(TMap* entryCache = nullptr, Int_t run = -1);

  void init();

  void initFromCache(TMap* entryCache, Int_t run);

  Bool_t initFromSnapshot(const char* snapshotFileName, Bool_t overwrite = kTRUE);

  Bool_t setSnapshotMode(const char* snapshotFileName = "OCDB.root");

  void unsetSnapshotMode()
  {
    mSnapshotMode = kFALSE;
  }

  void dumpToSnapshotFile(const char* snapshotFileName, Bool_t singleKeys) const;

  void dumpToLightSnapshotFile(const char* lightSnapshotFileName) const;

  Int_t getStartRunLHCPeriod();

  Int_t getEndRunLHCPeriod();

  TString getLHCPeriod();

  TString getCvmfsOcdbTag() const
  {
    return mCvmfsOcdb;
  }

  Bool_t diffObjects(const char* cdbFile1, const char* cdbFile2) const;

  void extractBaseFolder(TString& url); // remove everything but the url from OCDB path

 protected:
  static TString sOcdbFolderXmlFile; // alien path of the XML file for OCDB folder <--> Run range correspondance

  Manager();

  Manager(const Manager& source);

  Manager& operator=(const Manager& source);

  static Manager* sInstance; // Manager instance

  Storage* getStorage(const StorageParameters* param);

  Storage* getActiveStorage(const StorageParameters* param);

  void putActiveStorage(StorageParameters* param, Storage* storage);

  void setSpecificStorage(const char* calibType, const StorageParameters* param, Int_t version = -1,
                          Int_t subVersion = -1);

  void alienToCvmfsUri(TString& uriString) const;

  void validateCvmfsCase() const;

  void getLHCPeriodAgainstAlienFile(Int_t run, TString& lhcPeriod, Int_t& startRun, Int_t& endRun);

  void getLHCPeriodAgainstCvmfsFile(Int_t run, TString& lhcPeriod, Int_t& startRun, Int_t& endRun);

  void cacheCondition(const char* path, Condition* entry);

  StorageParameters* selectSpecificStorage(const TString& path);

  ConditionId* getId(const ConditionId& query);

  ConditionId* getId(const IdPath& path, Int_t runNumber = -1, Int_t version = -1, Int_t subVersion = -1);

  ConditionId* getId(const IdPath& path, const IdRunRange& runRange, Int_t version = -1, Int_t subVersion = -1);

  TList mFactories;       //! list of registered storage factories
  TMap mActiveStorages;   //! list of active storages
  TMap mSpecificStorages; //! list of detector-specific storages
  TMap mConditionCache;   //! cache of the retrieved objects

  TList* mIds;       //! List of the retrieved object ConditionId's (to be streamed to file)
  TMap* mStorageMap; //! list of storages (to be streamed to file)

  Storage* mDefaultStorage; //! pointer to default storage
  Storage* mdrainStorage;   //! pointer to drain storage

  StorageParameters* mOfficialStorageParameters;  // Conditions data storage parameters
  StorageParameters* mReferenceStorageParameters; // Reference data storage parameters

  Int_t mRun;    //! The run number
  Bool_t mCache; //! The cache flag
  Bool_t mLock;  //! Lock flag, if ON default storage and run number cannot be reset

  Bool_t mSnapshotMode; //! flag saying if we are in snapshot mode
  TFile* mSnapshotFile;
  Bool_t mOcdbUploadMode; //! flag for uploads to Official CDBs (upload to cvmfs must follow upload
  // to AliEn)

  Bool_t mRaw;              // flag to say whether we are in the raw case
  TString mCvmfsOcdb;       // set from $OCDB_PATH, points to a cvmfs AliRoot package
  Int_t mStartRunLhcPeriod; // 1st run of the LHC period set
  Int_t mEndRunLhcPeriod;   // last run of the LHC period set
  TString mLhcPeriod;       // LHC period alien folder

 private:
  ULong64_t mKey; //! Key for locking/unlocking

  ClassDefOverride(Manager, 0);
};

/////////////////////////////////////////////////////////////////////
//                                                                 //
//  class StorageFactory                                     //
//                                                                 //
/////////////////////////////////////////////////////////////////////

class StorageParameters;

class StorageFactory : public TObject
{
  friend class Manager;

 public:
  ~StorageFactory()
    override = default;

  virtual Bool_t validateStorageUri(const char* dbString) = 0;

  virtual StorageParameters* createStorageParameter(const char* dbString) = 0;

 protected:
  virtual Storage* createStorage(const StorageParameters* param) = 0;

  ClassDefOverride(StorageFactory, 0);
};

/////////////////////////////////////////////////////////////////////
//                                                                 //
// class StorageParameters //
//                                                                 //
/////////////////////////////////////////////////////////////////////

class StorageParameters : public TObject
{

 public:
  StorageParameters();

  ~StorageParameters() override;

  const TString& getStorageType() const
  {
    return mType;
  };

  const TString& getUri() const
  {
    return mURI;
  };

  virtual StorageParameters* cloneParam() const = 0;

 protected:
  void setType(const char* type)
  {
    mType = type;
  };

  void setUri(const char* uri)
  {
    mURI = uri;
  };

 private:
  TString mType; //! CDB type
  TString mURI;  //! CDB URI

  ClassDefOverride(StorageParameters, 0);
};

} // namespace ccdb
} // namespace o2
#endif
