#ifndef ALICEO2_CDB_LOCAL_H_
#define ALICEO2_CDB_LOCAL_H_

//  class  LocalStorage						   //
//  access class to a DataBase in a local storage                  //
#include "CCDB/Manager.h"  // for StorageFactory, StorageParameters
#include "Rtypes.h"   // for Bool_t, Int_t, ClassDef, LocalStorage::Class, etc
#include "CCDB/Storage.h"  // for Storage
#include "TString.h"  // for TString

class TList;

class TObject;
namespace AliceO2 { namespace CDB { class Condition; }}
namespace AliceO2 { namespace CDB { class ConditionId; }}
namespace AliceO2 { namespace CDB { class IdRunRange; }}

namespace AliceO2 {
namespace CDB {

class LocalStorage : public Storage
{
    friend class LocalStorageFactory;

  public:
    virtual Bool_t isReadOnly() const
    {
      return kFALSE;
    };

    virtual Bool_t hasSubVersion() const
    {
      return kTRUE;
    };

    virtual Bool_t hasConditionType(const char *path) const;

    virtual Bool_t idToFilename(const ConditionId &id, TString &filename) const;

    virtual void setRetry(Int_t /* nretry */, Int_t /* initsec */);

  protected:
    virtual Condition *getCondition(const ConditionId &queryId);

    virtual ConditionId *getConditionId(const ConditionId &queryId);

    virtual TList *getAllEntries(const ConditionId &queryId);

    virtual Bool_t putCondition(Condition *entry, const char *mirrors = "");

    virtual TList *getIdListFromFile(const char *fileName);

  private:
    LocalStorage(const LocalStorage &source);

    LocalStorage &operator=(const LocalStorage &source);

    LocalStorage(const char *baseDir);

    virtual ~LocalStorage();

    Bool_t filenameToId(const char *filename, IdRunRange &runRange, Int_t &version, Int_t &subVersion);

    Bool_t prepareId(ConditionId &id);

    //	Bool_t getId(const  ConditionId& query,  ConditionId& result);
    ConditionId *getId(const ConditionId &query);

    virtual void queryValidFiles();

    void queryValidCVMFSFiles(TString &cvmfsOcdbTag);

    void getEntriesForLevel0(const char *level0, const ConditionId &query, TList *result);

    void getEntriesForLevel1(const char *level0, const char *Level1, const ConditionId &query, TList *result);

    TString mBaseDirectory; // path of the DB folder

  ClassDef(LocalStorage, 0) // access class to a DataBase in a local storage
};

//  class  LocalStorageFactory
class LocalStorageFactory : public StorageFactory
{
  public:
    virtual Bool_t validateStorageUri(const char *dbString);

    virtual StorageParameters *createStorageParameter(const char *dbString);

  protected:
    virtual Storage *createStorage(const StorageParameters *param);

  ClassDef(LocalStorageFactory, 0)
};

//  class  LocalStorageParameters
class LocalStorageParameters : public StorageParameters
{
  public:
    LocalStorageParameters();

    LocalStorageParameters(const char *dbPath);

    LocalStorageParameters(const char *dbPath, const char *uri);

    virtual ~LocalStorageParameters();

    const TString &getPathString() const
    {
      return mDBPath;
    };

    virtual StorageParameters *cloneParam() const;

    virtual ULong_t getHash() const;

    virtual Bool_t isEqual(const TObject *obj) const;

  private:
    TString mDBPath; // path of the DB folder
  ClassDef(LocalStorageParameters, 0)
};
}
}
#endif
