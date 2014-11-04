#ifndef ALICEO2_CDB_GRID_H_
#define ALICEO2_CDB_GRID_H_

#include "Storage.h"
#include "Manager.h"
#include "ConditionMetaData.h"

namespace AliceO2 {
namespace CDB {

class GridStorage : public Storage {
  friend class GridStorageFactory;

public:
  virtual Bool_t isReadOnly() const
  {
    return kFALSE;
  }
  virtual Bool_t hasSubVersion() const
  {
    return kFALSE;
  }
  virtual Bool_t hasConditionType(const char* path) const;
  virtual Bool_t idToFilename(const ConditionId& id, TString& filename) const;
  virtual void setRetry(Int_t nretry, Int_t initsec);
  virtual void setMirrorSEs(const char* mirrors)
  {
    mMirrorSEs = mirrors;
  }
  virtual const char* getMirrorSEs() const
  {
    return mMirrorSEs;
  }

protected:
  virtual Condition* getCondition(const ConditionId& queryId);
  virtual ConditionId* getConditionId(const ConditionId& queryId);
  virtual TList* getAllEntries(const ConditionId& queryId);
  virtual Bool_t putCondition(Condition* entry, const char* mirrors = "");
  virtual TList* getIdListFromFile(const char* fileName);

private:
  GridStorage(const char* gridUrl, const char* user, const char* dbFolder, const char* se, const char* cacheFolder,
       Bool_t operateDisconnected, Long64_t cacheSize, Long_t cleanupInterval);

  virtual ~GridStorage();

  GridStorage(const GridStorage& db);
  GridStorage& operator=(const GridStorage& db);

  Bool_t filenameToId(TString& filename, ConditionId& id);

  Bool_t prepareId(ConditionId& id);
  ConditionId* getId(const TObjArray& validFileIds, const ConditionId& query);
  Condition* getConditionFromFile(TString& filename, ConditionId* dataId);
  Bool_t putInCvmfs(TString& fullFilename, TFile* cdbFile) const;

  // TODO  use AliEnTag classes!
  Bool_t addTag(TString& foldername, const char* tagname);
  Bool_t tagFileId(TString& filename, const ConditionId* id);
  Bool_t tagFileConditionMetaData(TString& filename, const ConditionMetaData* md);

  void makeQueryFilter(Int_t firstRun, Int_t lastRun, const ConditionMetaData* md, TString& result) const;

  virtual void queryValidFiles();

  TString mGridUrl;            // GridStorage Url ("alien://aliendb4.cern.ch:9000")
  TString mUser;               // User
  TString mDBFolder;           // path of the DB folder
  TString mSE;                 // Storage Element
  TString mMirrorSEs;          // Mirror Storage Elements
  TString mCacheFolder;        // local cache folder
  Bool_t mOperateDisconnected; // Operate disconnected flag
  Long64_t mCacheSize;         // local cache size (in bytes)
  Long_t mCleanupInterval;     // local cache cleanup interval

  ClassDef(GridStorage, 0) // access class to a DataBase in an AliEn storage
};

/////////////////////////////////////////////////////////////////////
//                                                                 //
//  class  GridStorageFactory					   //
//                                                                 //
/////////////////////////////////////////////////////////////////////

class GridStorageFactory : public StorageFactory {

public:
  virtual Bool_t validateStorageUri(const char* gridString);
  virtual StorageParameters* createStorageParameter(const char* gridString);
  virtual ~GridStorageFactory()
  {
  }

protected:
  virtual Storage* createStorage(const StorageParameters* param);

  ClassDef(GridStorageFactory, 0)
};

/////////////////////////////////////////////////////////////////////
//                                                                 //
//  class  GridStorageParameters					   //
//                                                                 //
/////////////////////////////////////////////////////////////////////

class GridStorageParameters : public StorageParameters {

public:
  GridStorageParameters();
  GridStorageParameters(const char* gridUrl, const char* user, const char* dbFolder, const char* se, const char* cacheFolder,
            Bool_t operateDisconnected, Long64_t cacheSize, Long_t cleanupInterval);

  virtual ~GridStorageParameters();

  const TString& GridUrl() const
  {
    return mGridUrl;
  }
  const TString& getUser() const
  {
    return mUser;
  }
  const TString& getDBFolder() const
  {
    return mDBFolder;
  }
  const TString& getSE() const
  {
    return mSE;
  }
  const TString& getCacheFolder() const
  {
    return mCacheFolder;
  }
  Bool_t getOperateDisconnected() const
  {
    return mOperateDisconnected;
  }
  Long64_t getCacheSize() const
  {
    return mCacheSize;
  }
  Long_t getCleanupInterval() const
  {
    return mCleanupInterval;
  }

  virtual StorageParameters* cloneParam() const;

  virtual ULong_t getHash() const;
  virtual Bool_t isEqual(const TObject* obj) const;

private:
  TString mGridUrl;            // GridStorage url "Host:port"
  TString mUser;               // User
  TString mDBFolder;           // path of the DB folder
  TString mSE;                 // Storage Element
  TString mCacheFolder;        // Cache folder
  Bool_t mOperateDisconnected; // Operate disconnected flag
  Long64_t mCacheSize;         // local cache size (in bytes)
  Long_t mCleanupInterval;     // local cache cleanup interval

  ClassDef(GridStorageParameters, 0)
};
}
}
#endif
