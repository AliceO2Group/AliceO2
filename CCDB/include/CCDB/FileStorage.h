#ifndef ALICEO2_CDB_FILEDUMP_H_
#define ALICEO2_CDB_FILEDUMP_H_

#include "CCDB/Manager.h"  // for StorageFactory, StorageParameters
#include "Rtypes.h"   // for Bool_t, Int_t, ClassDef, kFALSE, etc
#include "CCDB/Storage.h"  // for Storage
#include "TString.h"  // for TString

class TFile;  // lines 8-8
class TList;

class TObject;
namespace AliceO2 { namespace CDB { class Condition; }}
namespace AliceO2 { namespace CDB { class ConditionId; }}
namespace AliceO2 { namespace CDB { class IdRunRange; }}

namespace AliceO2 {
namespace CDB {

class FileStorage : public Storage
{
    friend class FileStorageFactory;

  public:
    virtual Bool_t isReadOnly() const
    {
      return mReadOnly;
    };

    virtual Bool_t hasSubVersion() const
    {
      return kFALSE;
    };

    virtual Bool_t hasConditionType(const char *path) const;

    virtual Bool_t idToFilename(const ConditionId &id, TString &filename) const;

    virtual void setRetry(Int_t /* nretry */, Int_t /* initsec */);

  protected:
    virtual Condition *getCondition(const ConditionId &query);

    virtual ConditionId *getConditionId(const ConditionId &query);

    virtual TList *getAllEntries(const ConditionId &query);

    virtual Bool_t putCondition(Condition *entry, const char *mirrors = "");

    virtual TList *getIdListFromFile(const char *fileName);

  private:
    FileStorage(const FileStorage &source);

    FileStorage &operator=(const FileStorage &source);

    FileStorage(const char *dbFile, Bool_t readOnly);

    virtual ~FileStorage();

    Bool_t keyNameToId(const char *keyname, IdRunRange &runRange, Int_t &version, Int_t &subVersion);

    Bool_t idToKeyName(const IdRunRange &runRange, Int_t version, Int_t subVersion, TString &keyname);

    Bool_t makeDir(const TString &dir);

    Bool_t prepareId(ConditionId &id);

    //	Bool_t getId(const  ConditionId& query,  ConditionId& result);
    ConditionId *getId(const ConditionId &query);

    virtual void queryValidFiles();

    void getEntriesForLevel0(const ConditionId &query, TList *result);

    void getEntriesForLevel1(const ConditionId &query, TList *result);

    TFile *mFile;     // FileStorage file
    Bool_t mReadOnly; // ReadOnly flag

  ClassDef(FileStorage, 0)
};

/////////////////////////////////////////////////////////////////////
//                                                                 //
//  class  FileStorageFactory					   //
//                                                                 //
/////////////////////////////////////////////////////////////////////

class FileStorageFactory : public StorageFactory
{

  public:
    virtual Bool_t validateStorageUri(const char *dbString);

    virtual StorageParameters *createStorageParameter(const char *dbString);

  protected:
    virtual Storage *createStorage(const StorageParameters *param);

  ClassDef(FileStorageFactory, 0)
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

    FileStorageParameters(const char *dbPath, Bool_t readOnly = kFALSE);

    virtual ~FileStorageParameters();

    const TString &getPathString() const
    {
      return mDBPath;
    };

    Bool_t isReadOnly() const
    {
      return mReadOnly;
    };

    virtual StorageParameters *cloneParam() const;

    virtual ULong_t getHash() const;

    virtual Bool_t isEqual(const TObject *obj) const;

  private:
    TString mDBPath;  // FileStorage file path name
    Bool_t mReadOnly; // ReadOnly flag

  ClassDef(FileStorageParameters, 0)
};
}
}
#endif
