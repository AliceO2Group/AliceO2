//  access class to a DataBase in a dump storage (single file)     //
#include "CCDB/FileStorage.h"
#include <FairLogger.h>   // for LOG
#include <TFile.h>        // for TFile
#include <TKey.h>         // for TKey
#include <TObjString.h>   // for TObjString
#include <TRegexp.h>      // for TRegexp
#include <TSystem.h>      // for TSystem, gSystem
#include "CCDB/Condition.h"    // for Condition

using namespace o2::CDB;

ClassImp(FileStorage)

FileStorage::FileStorage(const char *dbFile, Bool_t readOnly) : mFile(nullptr), mReadOnly(readOnly)
{
  // constructor

  // opening file
  mFile = TFile::Open(dbFile, mReadOnly ? "READ" : "UPDATE");
  if (!mFile) {
    LOG(ERROR) << R"(Can't open file ")" << dbFile << R"(")" << FairLogger::endl;
  } else {
    LOG(ERROR) << R"(File ")" << dbFile << R"(" opened)" << FairLogger::endl;
    if (mReadOnly)
      LOG(DEBUG) << "in read-only mode" << FairLogger::endl;
  }

  mType = "dump";
  mBaseFolder = dbFile;
}

FileStorage::~FileStorage()
{
  // destructor

  if (mFile) {
    mFile->Close();
    delete mFile;
  }
}

Bool_t FileStorage::keyNameToId(const char *keyname, IdRunRange &runRange, Int_t &version, Int_t &subVersion)
{
  // build  ConditionId from keyname numbers

  Ssiz_t mSize;

  // valid keyname: Run#firstRun_#lastRun_v#version_s#subVersion.root
  TRegexp keyPattern("^Run[0-9]+_[0-9]+_v[0-9]+_s[0-9]+$");
  keyPattern.Index(keyname, &mSize);
  if (!mSize) {
    LOG(DEBUG) << R"(Bad keyname ")" << keyname << R"(".)" << FairLogger::endl;
    return kFALSE;
  }

  TObjArray *strArray = (TObjArray *) TString(keyname).Tokenize("_");

  TString firstRunString(((TObjString *) strArray->At(0))->GetString());
  runRange.setFirstRun(atoi(firstRunString.Data() + 3));
  runRange.setLastRun(atoi(((TObjString *) strArray->At(1))->GetString()));

  TString verString(((TObjString *) strArray->At(2))->GetString());
  version = atoi(verString.Data() + 1);

  TString subVerString(((TObjString *) strArray->At(3))->GetString());
  subVersion = atoi(subVerString.Data() + 1);

  delete strArray;

  return kTRUE;
}

Bool_t FileStorage::idToKeyName(const IdRunRange &runRange, Int_t version, Int_t subVersion, TString &keyname)
{
  // build key name from  ConditionId data (run range, version, subVersion)

  if (!runRange.isValid()) {
    LOG(DEBUG) << "Invalid run range [" << runRange.getFirstRun() << "," << runRange.getLastRun() << "]."
               << FairLogger::endl;
    return kFALSE;
  }

  if (version < 0) {
    LOG(DEBUG) << R"(Invalid version ')" << version << R"('.)" << FairLogger::endl;
    return kFALSE;
  }

  if (subVersion < 0) {
    LOG(DEBUG) << R"(Invalid subversion ')" << subVersion << R"('.)" << FairLogger::endl;
    return kFALSE;
  }

  keyname += "Run";
  keyname += runRange.getFirstRun();
  keyname += "_";
  keyname += runRange.getLastRun();
  keyname += "_v";
  keyname += version;
  keyname += "_s";
  keyname += subVersion;

  return kTRUE;
}

Bool_t FileStorage::makeDir(const TString &path)
{
  // descend into TDirectory, making TDirectories if they don't exist
  TObjArray *strArray = (TObjArray *) path.Tokenize("/");

  TIter iter(strArray);
  TObjString *str;

  while ((str = (TObjString *) iter.Next())) {

    TString dirName(str->GetString());
    if (!dirName.Length()) {
      continue;
    }

    if (gDirectory->cd(dirName)) {
      continue;
    }

    TDirectory *aDir = gDirectory->mkdir(dirName, "");
    if (!aDir) {
      LOG(ERROR) << R"(Can't create directory ")" << dirName.Data() << R"("!)" << FairLogger::endl;
      delete strArray;

      return kFALSE;
    }

    aDir->cd();
  }

  delete strArray;

  return kTRUE;
}

Bool_t FileStorage::prepareId(ConditionId &id)
{
  // prepare id (version, subVersion) of the object that will be stored (called by putCondition)

  IdRunRange aIdRunRange;                         // the runRange got from filename
  IdRunRange lastIdRunRange(-1, -1);              // highest runRange found
  Int_t aVersion, aSubVersion;                // the version subVersion got from filename
  Int_t lastVersion = 0, lastSubVersion = -1; // highest version and subVersion found

  TIter iter(gDirectory->GetListOfKeys());
  TKey *key;

  if (!id.hasVersion()) { // version not specified: look for highest version & subVersion

    while ((key = (TKey *) iter.Next())) { // loop on keys

      const char *keyName = key->GetName();

      if (!keyNameToId(keyName, aIdRunRange, aVersion, aSubVersion)) {
        LOG(DEBUG) << "Bad keyname <" << keyName << ">!I'll skip it." << FairLogger::endl;
        continue;
      }

      if (!aIdRunRange.isOverlappingWith(id.getIdRunRange())) {
        continue;
      }
      if (aVersion < lastVersion) {
        continue;
      }
      if (aVersion > lastVersion) {
        lastSubVersion = -1;
      }
      if (aSubVersion < lastSubVersion) {
        continue;
      }
      lastVersion = aVersion;
      lastSubVersion = aSubVersion;
      lastIdRunRange = aIdRunRange;
    }

    id.setVersion(lastVersion);
    id.setSubVersion(lastSubVersion + 1);

  } else { // version specified, look for highest subVersion only

    while ((key = (TKey *) iter.Next())) { // loop on the keys

      const char *keyName = key->GetName();

      if (!keyNameToId(keyName, aIdRunRange, aVersion, aSubVersion)) {
        LOG(DEBUG) << "Bad keyname <" << keyName << ">!I'll skip it." << FairLogger::endl;
        continue;
      }

      if (aIdRunRange.isOverlappingWith(id.getIdRunRange()) && aVersion == id.getVersion() &&
          aSubVersion > lastSubVersion) {
        lastSubVersion = aSubVersion;
        lastIdRunRange = aIdRunRange;
      }
    }

    id.setSubVersion(lastSubVersion + 1);
  }

  TString lastStorage = id.getLastStorage();
  if (lastStorage.Contains(TString("grid"), TString::kIgnoreCase) && id.getSubVersion() > 0) {
    LOG(ERROR) << "GridStorage to FileStorage Storage error! local object with version v" << id.getVersion() << "_s"
               << id.getSubVersion() - 1 << " found:" << FairLogger::endl;
    LOG(ERROR) << "this object has been already transferred from GridStorage (check v" << id.getVersion() << "_s0)!"
               << FairLogger::endl;
    return kFALSE;
  }

  if (lastStorage.Contains(TString("new"), TString::kIgnoreCase) && id.getSubVersion() > 0) {
    LOG(DEBUG) << "A NEW object is being stored with version v" << id.getVersion() << "_s" << id.getSubVersion()
               << FairLogger::endl;
    LOG(DEBUG) << "and it will hide previously stored object with v" << id.getVersion() << "_s"
               << id.getSubVersion() - 1 << "!" << FairLogger::endl;
  }

  if (!lastIdRunRange.isAnyRange() && !(lastIdRunRange.isEqual(&id.getIdRunRange())))
    LOG(WARNING) << "Run range modified w.r.t. previous version (Run" << lastIdRunRange.getFirstRun() << "_"
                 << lastIdRunRange.getLastRun() << "_v" << id.getVersion() << "_s" << id.getSubVersion() - 1 << ")"
                 << FairLogger::endl;

  return kTRUE;
}

ConditionId *FileStorage::getId(const ConditionId &query)
{
  // look for filename matching query (called by getCondition)

  IdRunRange aIdRunRange;          // the runRange got from filename
  Int_t aVersion, aSubVersion; // the version and subVersion got from filename

  TIter iter(gDirectory->GetListOfKeys());
  TKey *key;

  ConditionId *result = new ConditionId();
  result->setPath(query.getPathString());

  if (!query.hasVersion()) { // neither version and subversion specified -> look for highest version
    // and subVersion

    while ((key = (TKey *) iter.Next())) { // loop on the keys

      if (!keyNameToId(key->GetName(), aIdRunRange, aVersion, aSubVersion)) {
        continue;
      }
      // aIdRunRange, aVersion, aSubVersion filled from filename

      if (!aIdRunRange.isSupersetOf(query.getIdRunRange())) {
        continue;
      }
      // aIdRunRange contains requested run!

      if (result->getVersion() < aVersion) {
        result->setVersion(aVersion);
        result->setSubVersion(aSubVersion);

        result->setFirstRun(aIdRunRange.getFirstRun());
        result->setLastRun(aIdRunRange.getLastRun());

      } else if (result->getVersion() == aVersion && result->getSubVersion() < aSubVersion) {

        result->setSubVersion(aSubVersion);

        result->setFirstRun(aIdRunRange.getFirstRun());
        result->setLastRun(aIdRunRange.getLastRun());
      } else if (result->getVersion() == aVersion && result->getSubVersion() == aSubVersion) {
        LOG(ERROR) << "More than one object valid for run " << query.getFirstRun() << ", version " << aVersion << "_"
                   << aSubVersion << "!" << FairLogger::endl;

        delete result;
        return nullptr;
      }
    }

  } else if (!query.hasSubVersion()) { // version specified but not subversion -> look for highest
    // subVersion

    result->setVersion(query.getVersion());

    while ((key = (TKey *) iter.Next())) { // loop on the keys

      if (!keyNameToId(key->GetName(), aIdRunRange, aVersion, aSubVersion)) {
        continue;
      }
      // aIdRunRange, aVersion, aSubVersion filled from filename

      if (!aIdRunRange.isSupersetOf(query.getIdRunRange())) {
        continue;
      }
      // aIdRunRange contains requested run!

      if (query.getVersion() != aVersion) {
        continue;
      }
      // aVersion is requested version!

      if (result->getSubVersion() == aSubVersion) {
        LOG(ERROR) << "More than one object valid for run " << query.getFirstRun() << " version " << aVersion << "_"
                   << aSubVersion << "!" << FairLogger::endl;
        delete result;
        return nullptr;
      }
      if (result->getSubVersion() < aSubVersion) {

        result->setSubVersion(aSubVersion);

        result->setFirstRun(aIdRunRange.getFirstRun());
        result->setLastRun(aIdRunRange.getLastRun());
      }
    }

  } else { // both version and subversion specified

    while ((key = (TKey *) iter.Next())) { // loop on the keys

      if (!keyNameToId(key->GetName(), aIdRunRange, aVersion, aSubVersion)) {
        continue;
      }
      // aIdRunRange, aVersion, aSubVersion filled from filename

      if (!aIdRunRange.isSupersetOf(query.getIdRunRange())) {
        continue;
      }
      // aIdRunRange contains requested run!

      if (query.getVersion() != aVersion || query.getSubVersion() != aSubVersion) {
        continue;
      }
      // aVersion and aSubVersion are requested version and subVersion!

      if (result->getVersion() == aVersion && result->getSubVersion() == aSubVersion) {
        LOG(ERROR) << "More than one object valid for run " << query.getFirstRun() << " version " << aVersion << "_"
                   << aSubVersion << "!" << FairLogger::endl;
        delete result;
        return nullptr;
      }
      result->setVersion(aVersion);
      result->setSubVersion(aSubVersion);
      result->setFirstRun(aIdRunRange.getFirstRun());
      result->setLastRun(aIdRunRange.getLastRun());
    }
  }

  return result;
}

Condition *FileStorage::getCondition(const ConditionId &queryId)
{
  // get  Condition from the database

  TDirectory::TContext context(gDirectory, mFile);

  if (!(mFile && mFile->IsOpen())) {
    LOG(ERROR) << "FileStorage is not initialized properly" << FairLogger::endl;
    return nullptr;
  }

  if (!gDirectory->cd(queryId.getPathString())) {
    return nullptr;
  }

  ConditionId *dataId = getConditionId(queryId);

  if (!dataId || !dataId->isSpecified()) {
    if (dataId) {
      delete dataId;
    }
    return nullptr;
  }

  TString keyname;
  if (!idToKeyName(dataId->getIdRunRange(), dataId->getVersion(), dataId->getSubVersion(), keyname)) {
    LOG(DEBUG) << "Bad ID encountered! Subnormal error!" << FairLogger::endl;
    delete dataId;
    return nullptr;
  }

  // get the only  Condition object from the file
  // the object in the file is an  Condition entry named keyname
  // keyName = Run#firstRun_#lastRun_v#version_s#subVersion

  TObject *anObject = gDirectory->Get(keyname);
  if (!anObject) {
    LOG(DEBUG) << "Bad storage data: NULL entry object!" << FairLogger::endl;
    delete dataId;
    return nullptr;
  }

  if (Condition::Class() != anObject->IsA()) {
    LOG(DEBUG) << "Bad storage data: Invalid entry object!" << FairLogger::endl;
    delete dataId;
    return nullptr;
  }

  ((Condition *) anObject)->setLastStorage("dump");

  delete dataId;
  return (Condition *) anObject;
}

ConditionId *FileStorage::getConditionId(const ConditionId &queryId)
{
  // get  Condition from the database

  TDirectory::TContext context(gDirectory, mFile);

  if (!(mFile && mFile->IsOpen())) {
    LOG(ERROR) << "FileStorage is not initialized properly" << FairLogger::endl;
    return nullptr;
  }

  if (!gDirectory->cd(queryId.getPathString())) {
    return nullptr;
  }

  ConditionId *dataId = nullptr;

  // look for a filename matching query requests (path, runRange, version, subVersion)
  if (!queryId.hasVersion()) {
    // if version is not specified, first check the selection criteria list
    ConditionId selectedId(queryId);
    getSelection(&selectedId);
    dataId = getId(queryId);
  } else {
    dataId = getId(queryId);
  }

  if (dataId && !dataId->isSpecified()) {
    delete dataId;
    return nullptr;
  }

  return dataId;
}

void FileStorage::getEntriesForLevel0(const ConditionId &queryId, TList *result)
{
  // multiple request ( Storage::GetAllObjects)

  TDirectory *saveDir = gDirectory;

  TIter iter(gDirectory->GetListOfKeys());
  TKey *key;

  while ((key = (TKey *) iter.Next())) {

    TString keyNameStr(key->GetName());
    if (queryId.getPath().doesLevel1Contain(keyNameStr)) {
      gDirectory->cd(keyNameStr);
      getEntriesForLevel1(queryId, result);

      saveDir->cd();
    }
  }
}

void FileStorage::getEntriesForLevel1(const ConditionId &queryId, TList *result)
{
  // multiple request ( Storage::GetAllObjects)

  TIter iter(gDirectory->GetListOfKeys());
  TKey *key;

  TDirectory *level0Dir = (TDirectory *) gDirectory->GetMother();

  while ((key = (TKey *) iter.Next())) {

    TString keyNameStr(key->GetName());
    if (queryId.getPath().doesLevel2Contain(keyNameStr)) {

      IdPath aPath(level0Dir->GetName(), gDirectory->GetName(), keyNameStr);
      ConditionId anId(aPath, queryId.getIdRunRange(), queryId.getVersion(), -1);

      Condition *anCondition = getCondition(anId);
      if (anCondition) {
        result->Add(anCondition);
      }
    }
  }
}

TList *FileStorage::getAllEntries(const ConditionId &queryId)
{
  // return list of CDB entries matching a generic request (Storage::GetAllObjects)

  TDirectory::TContext context(gDirectory, mFile);

  if (!(mFile && mFile->IsOpen())) {
    LOG(ERROR) << "FileStorage is not initialized properly" << FairLogger::endl;
    return nullptr;
  }

  TList *result = new TList();
  result->SetOwner();

  TIter iter(gDirectory->GetListOfKeys());
  TKey *key;

  while ((key = (TKey *) iter.Next())) {

    TString keyNameStr(key->GetName());
    if (queryId.getPath().doesLevel0Contain(keyNameStr)) {
      gDirectory->cd(keyNameStr);
      getEntriesForLevel0(queryId, result);

      mFile->cd();
    }
  }

  return result;
}

Bool_t FileStorage::putCondition(Condition *entry, const char *mirrors)
{
  // put an  Condition object into the database

  TDirectory::TContext context(gDirectory, mFile);

  if (!(mFile && mFile->IsOpen())) {
    LOG(ERROR) << "FileStorage is not initialized properly" << FairLogger::endl;
    return kFALSE;
  }

  if (mReadOnly) {
    LOG(ERROR) << "FileStorage is read only!" << FairLogger::endl;
    return kFALSE;
  }

  TString mirrorsString(mirrors);
  if (!mirrorsString.IsNull())
    LOG(WARNING) << "LocalStorage storage cannot take mirror SEs into account. They will be ignored." <<
                 FairLogger::endl;

  ConditionId &id = entry->getId();

  if (!gDirectory->cd(id.getPathString())) {
    if (!makeDir(id.getPathString())) {
      LOG(ERROR) << R"(Can't open directory ")" << id.getPathString().Data() << R"("!)" << FairLogger::endl;
      return kFALSE;
    }
  }

  // set version and subVersion for the entry to be stored
  if (!prepareId(id)) {
    return kFALSE;
  }

  // build keyname from entry's id
  TString keyname;
  if (!idToKeyName(id.getIdRunRange(), id.getVersion(), id.getSubVersion(), keyname)) {
    LOG(ERROR) << "Invalid ID encountered! Subnormal error!" << FairLogger::endl;
    return kFALSE;
  }

  // write object (key name: Run#firstRun_#lastRun_v#version_s#subVersion)
  Bool_t result = gDirectory->WriteTObject(entry, keyname);
  if (!result) {
    LOG(ERROR) << R"(Can't write entry to file ")" << mFile->GetName() << R"(")" << FairLogger::endl;
  }

  if (result) {
    LOG(INFO) << "CDB object stored into file " << mFile->GetName() << FairLogger::endl;
    LOG(INFO) << "TDirectory/key name: " << id.getPathString().Data() << "/" << keyname.Data() << FairLogger::endl;
  }

  return result;
}

TList *FileStorage::getIdListFromFile(const char *fileName)
{

  TString turl(fileName);
  if (turl[0] != '/') {
    turl.Prepend(TString(gSystem->WorkingDirectory()) + '/');
  }
  TFile *file = TFile::Open(turl);
  if (!file) {
    LOG(ERROR) << "Can't open selection file <" << turl.Data() << ">!" << FairLogger::endl;
    return nullptr;
  }
  file->cd();

  TList *list = new TList();
  list->SetOwner();
  int i = 0;
  TString keycycle;

  ConditionId *id;
  while (1) {
    i++;
    keycycle = " ConditionId;";
    keycycle += i;

    id = (ConditionId *) file->Get(keycycle);
    if (!id) {
      break;
    }
    list->AddFirst(id);
  }
  file->Close();
  delete file;
  file = nullptr;
  return list;
}

Bool_t FileStorage::hasConditionType(const char *path) const
{
  // check for path in storage

  TDirectory::TContext context(gDirectory, mFile);
  if (!(mFile && mFile->IsOpen())) {
    LOG(ERROR) << "FileStorage is not initialized properly" << FairLogger::endl;
    return kFALSE;
  }

  return gDirectory->cd(path);
}

void FileStorage::queryValidFiles()
{
  // Query the CDB for files valid for  Storage::mRun
  // fills list mValidFileIds with  ConditionId objects created from file name

  LOG(ERROR) << "Not yet (and maybe never) implemented" << FairLogger::endl;
}

Bool_t FileStorage::idToFilename(const ConditionId & /*id*/, TString & /*filename*/) const
{
  // build file name from  ConditionId (path, run range, version) and mDBFolder

  LOG(ERROR) << "Not implemented" << FairLogger::endl;
  return kFALSE;
}

void FileStorage::setRetry(Int_t /* nretry */, Int_t /* initsec */)
{
  // Function to set the exponential retry for putting entries in the OCDB
  LOG(INFO) << "This function sets the exponential retry for putting entries in the OCDB - to be "
    "used ONLY for  GridStorage --> returning without doing anything" << FairLogger::endl;
  return;
}

// FileStorage factory
ClassImp(FileStorageFactory)

Bool_t FileStorageFactory::validateStorageUri(const char *dbString)
{
  // check if the string is valid dump URI
  TRegexp dbPattern("^dump://.+$");
  return TString(dbString).Contains(dbPattern);
}

StorageParameters *FileStorageFactory::createStorageParameter(const char *dbString)
{
  // create  FileStorageParameters class from the URI string

  if (!validateStorageUri(dbString)) {
    return nullptr;
  }

  TString pathname(dbString + sizeof("dump://") - 1);

  Bool_t readOnly;

  if (pathname.Contains(TRegexp(";ReadOnly$"))) {
    pathname.Resize(pathname.Length() - sizeof(";ReadOnly") + 1);
    readOnly = kTRUE;
  } else {
    readOnly = kFALSE;
  }

  gSystem->ExpandPathName(pathname);

  if (pathname[0] != '/') {
    pathname.Prepend(TString(gSystem->WorkingDirectory()) + '/');
  }

  return new FileStorageParameters(pathname, readOnly);
}

Storage *FileStorageFactory::createStorage(const StorageParameters *param)
{
  // create FileStorage instance from parameters
  if (FileStorageParameters::Class() == param->IsA()) {
    const FileStorageParameters *dumpParam = (const FileStorageParameters *) param;
    FileStorage *dumpStorage = new FileStorage(dumpParam->getPathString(), dumpParam->isReadOnly());
    return dumpStorage;
  }
  return nullptr;
}

// FileStorage parameter class
ClassImp(FileStorageParameters)

FileStorageParameters::FileStorageParameters() : StorageParameters(), mDBPath(), mReadOnly(kFALSE)
{
  // default constructor
}

FileStorageParameters::FileStorageParameters(const char *dbPath, Bool_t readOnly) : mDBPath(dbPath), mReadOnly(readOnly)
{
  // constructor
  TString uri;
  uri += "dump://";
  uri += dbPath;

  if (mReadOnly) {
    uri += ";ReadOnly";
  }

  setUri(uri);
  setType("dump");
}

FileStorageParameters::~FileStorageParameters()
{
  // destructor
}

StorageParameters *FileStorageParameters::cloneParam() const
{
  // clone parameter

  return new FileStorageParameters(mDBPath, mReadOnly);
}

ULong_t FileStorageParameters::getHash() const
{
  // return getHash function

  return mDBPath.Hash();
}

Bool_t FileStorageParameters::isEqual(const TObject *obj) const
{
  // check if this object is equal to  StorageParameters obj

  if (this == obj) {
    return kTRUE;
  }

  if (FileStorageParameters::Class() != obj->IsA()) {
    return kFALSE;
  }

  FileStorageParameters *other = (FileStorageParameters *) obj;

  return mDBPath == other->mDBPath;
}
