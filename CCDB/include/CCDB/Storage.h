#ifndef ALICEO2_CDB_STORAGE_H_
#define ALICEO2_CDB_STORAGE_H_

//  interface to specific storage classes                          //
//  ( GridStorage,  LocalStorage, FileStorage)			   //
#include <TList.h>      // for TList
#include <TObjArray.h>  // for TObjArray
#include "CCDB/IdPath.h"     // for IdPath
#include "Rtypes.h"     // for Int_t, Bool_t, Short_t, Storage::Class, etc
#include "TObject.h"    // for TObject
#include "TString.h"    // for TString

namespace o2 { namespace CDB { class Condition; }}  // lines 18-18
namespace o2 { namespace CDB { class ConditionId; }}
namespace o2 { namespace CDB { class ConditionMetaData; }}
namespace o2 { namespace CDB { class IdRunRange; }}

class TFile;

namespace o2 {
namespace CDB {

class Condition;

class IdPath;

class Param;

class Storage : public TObject
{

  public:
    Storage();

    void setUri(const TString &uri)
    {
      mURI = uri;
    }

    const TString &getUri() const
    {
      return mURI;
    }

    const TString &getStorageType() const
    {
      return mType;
    }

    const TString &getBaseFolder() const
    {
      return mBaseFolder;
    }

    void readSelectionFromFile(const char *fileName);

    void addSelection(const ConditionId &selection);

    void addSelection(const IdPath &path, const IdRunRange &runRange, Int_t version, Int_t subVersion = -1);

    void addSelection(const IdPath &path, Int_t firstRun, Int_t lastRun, Int_t version, Int_t subVersion = -1);

    void removeSelection(const ConditionId &selection);

    void removeSelection(const IdPath &path, const IdRunRange &runRange);

    void removeSelection(const IdPath &path, Int_t firstRun = -1, Int_t lastRun = -1);

    void removeSelection(int position);

    void removeAllSelections();

    void printSelectionList();

    Condition *getObject(const ConditionId &query);

    Condition *getObject(const IdPath &path, Int_t runNumber, Int_t version = -1, Int_t subVersion = -1);

    Condition *getObject(const IdPath &path, const IdRunRange &runRange, Int_t version = -1, Int_t subVersion = -1);

    TList *getAllObjects(const ConditionId &query);

    TList *getAllObjects(const IdPath &path, Int_t runNumber, Int_t version = -1, Int_t subVersion = -1);

    TList *getAllObjects(const IdPath &path, const IdRunRange &runRange, Int_t version = -1, Int_t subVersion = -1);

    ConditionId *getId(const ConditionId &query);

    ConditionId *getId(const IdPath &path, Int_t runNumber, Int_t version = -1, Int_t subVersion = -1);

    ConditionId *getId(const IdPath &path, const IdRunRange &runRange, Int_t version = -1, Int_t subVersion = -1);

    Bool_t putObject(TObject *object, ConditionId &id, ConditionMetaData *metaData, const char *mirrors = "");

    Bool_t putObject(Condition *entry, const char *mirrors = "");

    virtual void setMirrorSEs(const char *mirrors);

    virtual const char *getMirrorSEs() const;

    virtual Bool_t isReadOnly() const = 0;

    virtual Bool_t hasSubVersion() const = 0;

    virtual Bool_t hasConditionType(const char *path) const = 0;

    virtual Bool_t idToFilename(const ConditionId &id, TString &filename) const = 0;

    void queryStorages(Int_t run, const char *pathFilter = "*", Int_t version = -1, ConditionMetaData *mdFilter = nullptr);

    void printrQueryStorages();

    TObjArray *getQueryStoragesList()
    {
      return &mValidFileIds;
    }

    virtual void setRetry(Int_t nretry, Int_t initsec) = 0;

  protected:
    ~Storage() override;

    void getSelection(/*const*/ ConditionId *id);

    virtual Condition *getCondition(const ConditionId &query) = 0;

    virtual ConditionId *getConditionId(const ConditionId &query) = 0;

    virtual TList *getAllEntries(const ConditionId &query) = 0;

    virtual Bool_t putCondition(Condition *entry, const char *mirrors = "") = 0;

    virtual TList *getIdListFromFile(const char *fileName) = 0;

    virtual void queryValidFiles() = 0;

    void loadTreeFromFile(Condition *entry) const;
    // void 	setTreeToFile( Condition* entry, TFile* file) const;

    TObjArray mValidFileIds;   // list of ConditionId's of the files valid for a given run (cached as mRun)
    Int_t mRun;                // run number, used to manage list of valid files
    IdPath mPathFilter;          // path filter, used to manage list of valid files
    Int_t mVersion;            // version, used to manage list of valid files
    ConditionMetaData *mConditionMetaDataFilter; // metadata, used to manage list of valid files

    TList mSelections;         // list of selection criteria
    TString mURI;              // storage URI;
    TString mType;             //! LocalStorage, GridStorage: base folder name - Dump: file name
    TString mBaseFolder;       //! LocalStorage, GridStorage: base folder name - Dump: file name
    Short_t mNretry;           // Number of retries in opening the file
    Short_t mInitRetrySeconds; // Seconds for first retry

  private:
    Storage(const Storage &source);

    Storage &operator=(const Storage &source);

  ClassDefOverride(Storage, 0)
};
}
}
#endif
