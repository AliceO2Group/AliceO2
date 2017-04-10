#ifndef ALICEO2_CDB_GRID_H_
#define ALICEO2_CDB_GRID_H_

#include "CCDB/Manager.h"  // for StorageFactory, StorageParameters
#include "Rtypes.h"   // for Bool_t, Long64_t, Long_t, Int_t, ClassDef, etc
#include "CCDB/Storage.h"  // for Storage
#include "TString.h"  // for TString

class TFile;

class TList;

class TObjArray;

class TObject;
namespace o2 { namespace CDB { class Condition; }}
namespace o2 { namespace CDB { class ConditionId; }}
namespace o2 { namespace CDB { class ConditionMetaData; }}

namespace o2 {
namespace CDB {

class GridStorage : public Storage
{
    friend class GridStorageFactory;

  public:
    Bool_t isReadOnly() const override
    {
      return kFALSE;
    }

    Bool_t hasSubVersion() const override
    {
      return kFALSE;
    }

    Bool_t hasConditionType(const char *path) const override;

    Bool_t idToFilename(const ConditionId &id, TString &filename) const override;

    void setRetry(Int_t nretry, Int_t initsec) override;

    void setMirrorSEs(const char *mirrors) override
    {
      mMirrorSEs = mirrors;
    }

    const char *getMirrorSEs() const override
    {
      return mMirrorSEs;
    }

  protected:
    Condition *getCondition(const ConditionId &queryId) override;

    ConditionId *getConditionId(const ConditionId &queryId) override;

    TList *getAllEntries(const ConditionId &queryId) override;

    Bool_t putCondition(Condition *entry, const char *mirrors = "") override;

    TList *getIdListFromFile(const char *fileName) override;

  private:
    GridStorage(const char *gridUrl, const char *user, const char *dbFolder, const char *se, const char *cacheFolder,
                Bool_t operateDisconnected, Long64_t cacheSize, Long_t cleanupInterval);

    ~GridStorage() override;

    GridStorage(const GridStorage &db);

    GridStorage &operator=(const GridStorage &db);

    Bool_t filenameToId(TString &filename, ConditionId &id);

    Bool_t prepareId(ConditionId &id);

    ConditionId *getId(const TObjArray &validFileIds, const ConditionId &query);

    Condition *getConditionFromFile(TString &filename, ConditionId *dataId);

    Bool_t putInCvmfs(TString &fullFilename, TFile *cdbFile) const;

    // TODO  use AliEnTag classes!
    Bool_t addTag(TString &foldername, const char *tagname);

    Bool_t tagFileId(TString &filename, const ConditionId *id);

    Bool_t tagFileConditionMetaData(TString &filename, const ConditionMetaData *md);

    void makeQueryFilter(Int_t firstRun, Int_t lastRun, const ConditionMetaData *md, TString &result) const;

    void queryValidFiles() override;

    TString mGridUrl;            // GridStorage Url ("alien://aliendb4.cern.ch:9000")
    TString mUser;               // User
    TString mDBFolder;           // path of the DB folder
    TString mSE;                 // Storage Element
    TString mMirrorSEs;          // Mirror Storage Elements
    TString mCacheFolder;        // local cache folder
    Bool_t mOperateDisconnected; // Operate disconnected flag
    Long64_t mCacheSize;         // local cache size (in bytes)
    Long_t mCleanupInterval;     // local cache cleanup interval

  ClassDefOverride(GridStorage, 0) // access class to a DataBase in an AliEn storage
};

/////////////////////////////////////////////////////////////////////
//                                                                 //
//  class  GridStorageFactory					   //
//                                                                 //
/////////////////////////////////////////////////////////////////////

class GridStorageFactory : public StorageFactory
{

  public:
    Bool_t validateStorageUri(const char *gridString) override;

    StorageParameters *createStorageParameter(const char *gridString) override;

    ~GridStorageFactory()
    override = default;

  protected:
    Storage *createStorage(const StorageParameters *param) override;

  ClassDefOverride(GridStorageFactory, 0)
};

/////////////////////////////////////////////////////////////////////
//                                                                 //
//  class  GridStorageParameters					   //
//                                                                 //
/////////////////////////////////////////////////////////////////////

class GridStorageParameters : public StorageParameters
{

  public:
    GridStorageParameters();

    GridStorageParameters(const char *gridUrl, const char *user, const char *dbFolder, const char *se,
                          const char *cacheFolder,
                          Bool_t operateDisconnected, Long64_t cacheSize, Long_t cleanupInterval);

    ~GridStorageParameters() override;

    const TString &GridUrl() const
    {
      return mGridUrl;
    }

    const TString &getUser() const
    {
      return mUser;
    }

    const TString &getDBFolder() const
    {
      return mDBFolder;
    }

    const TString &getSE() const
    {
      return mSE;
    }

    const TString &getCacheFolder() const
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

    StorageParameters *cloneParam() const override;

    virtual ULong_t getHash() const;

    virtual Bool_t isEqual(const TObject *obj) const;

  private:
    TString mGridUrl;            // GridStorage url "Host:port"
    TString mUser;               // User
    TString mDBFolder;           // path of the DB folder
    TString mSE;                 // Storage Element
    TString mCacheFolder;        // Cache folder
    Bool_t mOperateDisconnected; // Operate disconnected flag
    Long64_t mCacheSize;         // local cache size (in bytes)
    Long_t mCleanupInterval;     // local cache cleanup interval

  ClassDefOverride(GridStorageParameters, 0)
};
}
}
#endif
