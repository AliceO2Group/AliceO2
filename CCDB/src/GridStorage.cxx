// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// access class to a DataBase in an AliEn storage                                              //

#include "CCDB/GridStorage.h"
#include <fairlogger/Logger.h> // for LOG
#include <TFile.h>             // for TFile
#include <TGrid.h>             // for gGrid, TGrid
#include <TGridResult.h>       // for TGridResult
#include <TObjString.h>        // for TObjString
#include <TROOT.h>             // for TROOT, gROOT
#include <TRegexp.h>           // for TRegexp
#include "CCDB/Condition.h"    // for Condition
#include "TSystem.h"           // for TSystem, gSystem

using namespace o2::ccdb;

ClassImp(GridStorage);

GridStorage::GridStorage(const char* gridUrl, const char* user, const char* dbFolder, const char* se,
                         const char* cacheFolder,
                         Bool_t operateDisconnected, Long64_t cacheSize, Long_t cleanupInterval)
  : Storage(),
    mGridUrl(gridUrl),
    mUser(user),
    mDBFolder(dbFolder),
    mSE(se),
    mMirrorSEs(""),
    mCacheFolder(cacheFolder),
    mOperateDisconnected(operateDisconnected),
    mCacheSize(cacheSize),
    mCleanupInterval(cleanupInterval)
{
  // constructor //
  // alalal

  // if the same GridStorage is alreay active, skip connection
  if (!gGrid || mGridUrl != gGrid->GridUrl() || ((mUser != "") && (mUser != gGrid->GetUser()))) {
    // connection to the GridStorage
    LOG(INFO) << "Connection to the GridStorage...";
    if (gGrid) {
      LOG(INFO) << "gGrid = " << gGrid << ";  mGridUrl = " << mGridUrl.Data()
                << ";  gGrid->GridUrl() = " << gGrid->GridUrl();
      LOG(INFO) << "mUser = " << mUser.Data() << ";  gGrid->getUser() = " << gGrid->GetUser();
    }
    TGrid::Connect(mGridUrl.Data(), mUser.Data());
  }

  if (!gGrid) {
    LOG(ERROR) << "Connection failed!";
    return;
  }

  TString initDir(gGrid->Pwd(0));
  if (mDBFolder[0] != '/') {
    mDBFolder.Prepend(initDir);
  }

  // check DBFolder: trying to cd to DBFolder; if it does not exist, create it
  if (!gGrid->Cd(mDBFolder.Data(), 0)) {
    LOG(DEBUG) << "Creating new folder <" << mDBFolder.Data() << "> ...";
    TGridResult* res = gGrid->Command(Form("mkdir -p %s", mDBFolder.Data()));
    TString result = res->GetKey(0, "__result__");
    if (result == "0") {
      LOG(FATAL) << R"(Cannot create folder ")" << mDBFolder.Data() << R"("!)";
      return;
    }
  } else {
    LOG(DEBUG) << "Folder <" << mDBFolder.Data() << "> found";
  }

  // removes any '/' at the end of path, then append one '/'
  while (mDBFolder.EndsWith("/")) {
    mDBFolder.Remove(mDBFolder.Last('/'));
  }
  mDBFolder += "/";

  mType = "alien";
  mBaseFolder = mDBFolder;

  // Setting the cache

  // Check if local cache folder is already defined
  TString origCache(TFile::GetCacheFileDir());
  if (mCacheFolder.Length() > 0) {
    if (origCache.Length() == 0) {
      LOG(INFO) << "Setting local cache to: " << mCacheFolder.Data();
    } else if (mCacheFolder != origCache) {
      LOG(WARNING) << "LocalStorage cache folder was already defined, changing it to: " << mCacheFolder.Data();
    }

    // default settings are: operateDisconnected=kTRUE, forceCacheread = kFALSE
    if (!TFile::SetCacheFileDir(mCacheFolder.Data(), mOperateDisconnected)) {
      LOG(ERROR) << "Could not set cache folder " << mCacheFolder.Data() << " !";
      mCacheFolder = "";
    } else {
      // reset mCacheFolder because the function may have
      // slightly changed the folder name (e.g. '/' added)
      mCacheFolder = TFile::GetCacheFileDir();
    }

    // default settings are: cacheSize=1GB, cleanupInterval = 0
    if (!TFile::ShrinkCacheFileDir(mCacheSize, mCleanupInterval)) {
      LOG(ERROR) << "Could not set following values to ShrinkCacheFileDir: cacheSize = " << mCacheSize
                 << " cleanupInterval = " << mCleanupInterval << " !";
    }
  }

  // return to the initial directory
  gGrid->Cd(initDir.Data(), 0);

  mNretry = 3;           // default
  mInitRetrySeconds = 5; // default
}

GridStorage::~GridStorage()
{
  // destructor
  delete gGrid;
  gGrid = nullptr;
}

Bool_t GridStorage::filenameToId(TString& filename, ConditionId& id)
{
  // build  ConditionId from full path filename (mDBFolder/path/Run#x_#y_v#z_s0.root)

  if (filename.Contains(mDBFolder)) {
    filename = filename(mDBFolder.Length(), filename.Length() - mDBFolder.Length());
  }

  TString idPath = filename(0, filename.Last('/'));
  id.setPath(idPath);
  if (!id.isValid()) {
    return kFALSE;
  }

  filename = filename(idPath.Length() + 1, filename.Length() - idPath.Length());

  Ssiz_t mSize;
  // valid filename: Run#firstRun_#lastRun_v#version_s0.root
  TRegexp keyPattern("^Run[0-9]+_[0-9]+_v[0-9]+_s0.root$");
  keyPattern.Index(filename, &mSize);
  if (!mSize) {

    // TODO backward compatibility ... maybe remove later!
    Ssiz_t oldmSize;
    TRegexp oldKeyPattern("^Run[0-9]+_[0-9]+_v[0-9]+.root$");
    oldKeyPattern.Index(filename, &oldmSize);
    if (!oldmSize) {
      LOG(DEBUG) << "Bad filename <" << filename.Data() << ">.";
      return kFALSE;
    } else {
      LOG(DEBUG) << "Old filename format <" << filename.Data() << ">.";
      id.setSubVersion(-11); // TODO trick to ensure backward compatibility
    }

  } else {
    id.setSubVersion(-1); // TODO trick to ensure backward compatibility
  }

  filename.Resize(filename.Length() - sizeof(".root") + 1);

  TObjArray* strArray = (TObjArray*)filename.Tokenize("_");

  TString firstRunString(((TObjString*)strArray->At(0))->GetString());
  id.setFirstRun(atoi(firstRunString.Data() + 3));
  id.setLastRun(atoi(((TObjString*)strArray->At(1))->GetString()));

  TString verString(((TObjString*)strArray->At(2))->GetString());
  id.setVersion(atoi(verString.Data() + 1));

  delete strArray;

  return kTRUE;
}

Bool_t GridStorage::idToFilename(const ConditionId& id, TString& filename) const
{
  // build file name from  ConditionId (path, run range, version) and mDBFolder

  if (!id.getIdRunRange().isValid()) {
    LOG(DEBUG) << "Invalid run range [" << id.getFirstRun() << "," << id.getLastRun() << "].";
    return kFALSE;
  }

  if (id.getVersion() < 0) {
    LOG(DEBUG) << "Invalid version <" << id.getVersion() << ">.";
    return kFALSE;
  }

  filename = Form("Run%d_%d_v%d", id.getFirstRun(), id.getLastRun(), id.getVersion());

  if (id.getSubVersion() != -11) {
    filename += "_s0";
  } // TODO to ensure backward compatibility
  filename += ".root";

  filename.Prepend(mDBFolder + id.getPathString() + '/');

  return kTRUE;
}

void GridStorage::setRetry(Int_t nretry, Int_t initsec)
{

  // Function to set the exponential retry for putting entries in the OCDB

  LOG(WARNING) << "WARNING!!! You are changing the exponential retry times and delay: this "
                  "function should be used by experts!";
  mNretry = nretry;
  mInitRetrySeconds = initsec;
  LOG(DEBUG) << "mNretry = " << mNretry << ", mInitRetrySeconds = " << mInitRetrySeconds;
}

Bool_t GridStorage::prepareId(ConditionId& id)
{
  // prepare id (version) of the object that will be stored (called by putCondition)

  TString initDir(gGrid->Pwd(0));

  TString dirName(mDBFolder);

  Bool_t dirExist = kFALSE;

  // go to the path; if directory does not exist, create it
  for (int i = 0; i < 3; i++) {
    // TString cmd("find -d ");
    // cmd += Form("%s ",dirName);
    // cmd +=
    // gGrid->Command(cmd.Data());
    dirName += Form("%s/", id.getPathLevel(i).Data());
    dirExist = gGrid->Cd(dirName, 0);
    if (!dirExist) {
      LOG(DEBUG) << "Creating new folder <" << dirName.Data() << "> ...";
      if (!gGrid->Mkdir(dirName, "", 0)) {
        LOG(ERROR) << "Cannot create directory <" << dirName.Data() << "> !";
        gGrid->Cd(initDir.Data());
        return kFALSE;
      }

      // if folders are new add tags to them
      if (i == 1) {

      } else if (i == 2) {
        LOG(DEBUG) << R"(Tagging level 2 folder with "CDB" and "CDB_MD" tag)";
        if (!addTag(dirName, "CDB")) {
          LOG(ERROR) << "Could not tag folder " << dirName.Data() << " !";
          if (!gGrid->Rmdir(dirName.Data())) {
            LOG(ERROR) << "Unexpected: could not remove " << dirName.Data() << " directory!";
          }
          return 0;
        }
        if (!addTag(dirName, "CDB_MD")) {
          LOG(ERROR) << "Could not tag folder " << dirName.Data() << " !";
          if (!gGrid->Rmdir(dirName.Data())) {
            LOG(ERROR) << "Unexpected: could not remove " << dirName.Data() << " directory!";
          }
          return 0;
        }
      }
    }
  }
  gGrid->Cd(initDir, 0);

  TString filename;
  ConditionId anId;                  // the id got from filename
  IdRunRange lastIdRunRange(-1, -1); // highest runRange found
  Int_t lastVersion = 0;             // highest version found

  TGridResult* res = gGrid->Ls(dirName);

  // loop on the files in the directory, look for highest version
  for (int i = 0; i < res->GetEntries(); i++) {
    filename = res->GetFileNamePath(i);
    if (!filenameToId(filename, anId)) {
      continue;
    }
    if (anId.getIdRunRange().isOverlappingWith(id.getIdRunRange()) && anId.getVersion() > lastVersion) {
      lastVersion = anId.getVersion();
      lastIdRunRange = anId.getIdRunRange();
    }
  }
  delete res;

  // GRP entries with explicitly set version escape default incremental versioning
  if (id.getPathString().Contains("GRP") && id.hasVersion() && lastVersion != 0) {
    LOG(DEBUG) << "Condition " << id.ToString().Data() << " won't be put in the destination OCDB";
    return kFALSE;
  }

  id.setVersion(lastVersion + 1);
  id.setSubVersion(0);

  TString lastStorage = id.getLastStorage();
  if (lastStorage.Contains(TString("new"), TString::kIgnoreCase) && id.getVersion() > 1) {
    LOG(DEBUG) << "A NEW object is being stored with version " << id.getVersion();
    LOG(DEBUG) << "and it will hide previously stored object with version " << id.getVersion() - 1 << "!";
  }

  if (!lastIdRunRange.isAnyRange() && !(lastIdRunRange.isEqual(&id.getIdRunRange())))
    LOG(WARNING) << "Run range modified w.r.t. previous version (Run" << lastIdRunRange.getFirstRun() << "_"
                 << lastIdRunRange.getLastRun() << "_v" << id.getVersion() << ")";

  return kTRUE;
}

ConditionId* GridStorage::getId(const TObjArray& validFileIds, const ConditionId& query)
{
  // look for the ConditionId that matches query's requests (highest or exact version)

  if (validFileIds.GetEntriesFast() < 1) {
    return nullptr;
  }

  TIter iter(&validFileIds);

  ConditionId* anIdPtr = nullptr;
  ConditionId* result = nullptr;

  while ((anIdPtr = dynamic_cast<ConditionId*>(iter.Next()))) {
    if (anIdPtr->getPathString() != query.getPathString()) {
      continue;
    }

    // if(!CheckVersion(query, anIdPtr, result)) return NULL;

    if (!query.hasVersion()) { // look for highest version
      if (result && result->getVersion() > anIdPtr->getVersion()) {
        continue;
      }
      if (result && result->getVersion() == anIdPtr->getVersion()) {
        LOG(ERROR) << "More than one object valid for run " << query.getFirstRun() << " version "
                   << anIdPtr->getVersion() << "!";
        return nullptr;
      }
      result = new ConditionId(*anIdPtr);
    } else { // look for specified version
      if (query.getVersion() != anIdPtr->getVersion()) {
        continue;
      }
      if (result && result->getVersion() == anIdPtr->getVersion()) {
        LOG(ERROR) << "More than one object valid for run " << query.getFirstRun() << " version "
                   << anIdPtr->getVersion() << "!";
        return nullptr;
      }
      result = new ConditionId(*anIdPtr);
    }
  }

  return result;
}

ConditionId* GridStorage::getConditionId(const ConditionId& queryId)
{
  // get  ConditionId from the database
  // User must delete returned object

  ConditionId* dataId = nullptr;

  ConditionId selectedId(queryId);
  if (!selectedId.hasVersion()) {
    // if version is not specified, first check the selection criteria list
    getSelection(&selectedId);
  }

  TObjArray validFileIds;
  validFileIds.SetOwner(1);

  // look for file matching query requests (path, runRange, version)
  if (selectedId.getFirstRun() == mRun && mPathFilter.isSupersetOf(selectedId.getPathString()) &&
      mVersion == selectedId.getVersion() && !mConditionMetaDataFilter) {
    // look into list of valid files previously loaded with  Storage::FillValidFileIds()
    LOG(DEBUG) << "List of files valid for run " << selectedId.getFirstRun()
               << " was loaded. Looking there for fileids valid for path " << selectedId.getPathString().Data() << "!";
    dataId = getId(mValidFileIds, selectedId);

  } else {
    // List of files valid for reqested run was not loaded. Looking directly into CDB
    LOG(DEBUG) << "List of files valid for run " << selectedId.getFirstRun() << " and version "
               << selectedId.getVersion() << " was not loaded. Looking directly into CDB for fileids valid for path "
               << selectedId.getPathString().Data() << "!";

    TString filter;
    makeQueryFilter(selectedId.getFirstRun(), selectedId.getLastRun(), nullptr, filter);

    TString pattern = ".root";
    TString optionQuery = "-y -m";
    if (selectedId.getVersion() >= 0) {
      pattern.Prepend(Form("_v%d_s0", selectedId.getVersion()));
      optionQuery = "";
    }

    TString folderCopy(Form("%s%s/Run", mDBFolder.Data(), selectedId.getPathString().Data()));

    if (optionQuery.Contains("-y")) {
      LOG(INFO) << "Only latest version will be returned";
    }

    LOG(DEBUG) << "** mDBFolder = " << folderCopy.Data() << ", pattern = " << pattern.Data()
               << ", filter = " << filter.Data();
    TGridResult* res = gGrid->Query(folderCopy, pattern, filter, optionQuery.Data());
    if (res) {
      for (int i = 0; i < res->GetEntries(); i++) {
        ConditionId* validFileId = new ConditionId();
        TString filename = res->GetKey(i, "lfn");
        if (filename == "") {
          continue;
        }
        if (filenameToId(filename, *validFileId)) {
          validFileIds.AddLast(validFileId);
        }
      }
      delete res;
    } else {
      return nullptr; // this should be only in case of file catalogue glitch
    }

    dataId = getId(validFileIds, selectedId);
  }

  return dataId;
}

Condition* GridStorage::getCondition(const ConditionId& queryId)
{
  // get  Condition from the database

  ConditionId* dataId = getConditionId(queryId);

  if (!dataId) {
    LOG(FATAL) << "No valid CDB object found! request was: " << queryId.ToString().Data();
    return nullptr;
  }

  TString filename;
  if (!idToFilename(*dataId, filename)) {
    LOG(DEBUG) << "Bad data ID encountered! Subnormal error!";
    delete dataId;
    LOG(FATAL) << "No valid CDB object found! request was: " << queryId.ToString().Data();
  }

  Condition* anCondition = getConditionFromFile(filename, dataId);

  delete dataId;
  if (!anCondition)
    LOG(FATAL) << "No valid CDB object found! request was: " << queryId.ToString().Data();

  return anCondition;
}

Condition* GridStorage::getConditionFromFile(TString& filename, ConditionId* dataId)
{
  // Get AliCBCondition object from file "filename"

  LOG(DEBUG) << "Opening file: " << filename.Data();

  filename.Prepend("/alien");

  // if option="CACHEREAD" TFile will use the local caching facility!
  TString option = "READ";
  if (mCacheFolder != "") {

    // Check if local cache folder was changed in the meanwhile
    TString origCache(TFile::GetCacheFileDir());
    if (mCacheFolder != origCache) {
      LOG(WARNING) << "LocalStorage cache folder has been overwritten!! mCacheFolder = " << mCacheFolder.Data()
                   << " origCache = " << origCache.Data();
      TFile::SetCacheFileDir(mCacheFolder.Data(), mOperateDisconnected);
      TFile::ShrinkCacheFileDir(mCacheSize, mCleanupInterval);
    }

    option.Prepend("CACHE");
  }

  LOG(DEBUG) << "Option: " << option.Data();

  TFile* file = TFile::Open(filename, option);
  if (!file) {
    LOG(DEBUG) << "Can't open file <" << filename.Data() << ">!";
    return nullptr;
  }

  // get the only  Condition object from the file
  // the object in the file is an  Condition entry named " Condition"

  Condition* anCondition = dynamic_cast<Condition*>(file->Get(" Condition"));

  if (!anCondition) {
    LOG(DEBUG) << "Bad storage data: file does not contain an  Condition object!";
    file->Close();
    return nullptr;
  }

  // The object's ConditionId is not reset during storage
  // If object's ConditionId runRange or version do not match with filename,
  // it means that someone renamed file by hand. In this case a warning msg is issued.

  if (anCondition) {
    ConditionId entryId = anCondition->getId();
    Int_t tmpSubVersion = dataId->getSubVersion();
    dataId->setSubVersion(entryId.getSubVersion()); // otherwise filename and id may mismatch
    if (!entryId.isEqual(dataId)) {
      LOG(WARNING) << "Mismatch between file name and object's ConditionId!";
      LOG(WARNING) << "File name: " << dataId->ToString().Data();
      LOG(WARNING) << "Object's ConditionId: " << entryId.ToString().Data();
    }
    dataId->setSubVersion(tmpSubVersion);
  }

  anCondition->setLastStorage("grid");

  // Check whether entry contains a TTree. In case load the tree in memory!
  loadTreeFromFile(anCondition);

  // close file, return retieved entry
  file->Close();
  delete file;
  file = nullptr;

  return anCondition;
}

TList* GridStorage::getAllEntries(const ConditionId& queryId)
{
  // return list of CDB entries matching a generic request (Storage::GetAllObjects)

  TList* result = new TList();
  result->SetOwner();

  TObjArray validFileIds;
  validFileIds.SetOwner(1);

  Bool_t alreadyLoaded = kFALSE;

  // look for file matching query requests (path, runRange)
  if (queryId.getFirstRun() == mRun && mPathFilter.isSupersetOf(queryId.getPathString()) && mVersion < 0 &&
      !mConditionMetaDataFilter) {
    // look into list of valid files previously loaded with  Storage::FillValidFileIds()
    LOG(DEBUG) << "List of files valid for run " << queryId.getFirstRun() << R"( and for path ")"
               << queryId.getPathString().Data() << R"(" was loaded. Looking there!)";

    alreadyLoaded = kTRUE;

  } else {
    // List of files valid for reqested run was not loaded. Looking directly into CDB
    LOG(DEBUG) << "List of files valid for run " << queryId.getFirstRun() << R"( and for path ")"
               << queryId.getPathString().Data() << " was not loaded. Looking directly into CDB!";

    TString filter;
    makeQueryFilter(queryId.getFirstRun(), queryId.getLastRun(), nullptr, filter);

    TString path = queryId.getPathString();

    TString pattern = "Run*.root";
    TString optionQuery = "-y";

    TString addFolder = "";
    if (!path.Contains("*")) {
      if (!path.BeginsWith("/")) {
        addFolder += "/";
      }
      addFolder += path;
    } else {
      if (path.BeginsWith("/")) {
        path.Remove(0, 1);
      }
      if (path.EndsWith("/")) {
        path.Remove(path.Length() - 1, 1);
      }
      TObjArray* tokenArr = path.Tokenize("/");
      if (tokenArr->GetEntries() != 3) {
        LOG(ERROR) << "Not a 3 level path! Keeping old query...";
        pattern.Prepend(path + "/");
      } else {
        TString str0 = ((TObjString*)tokenArr->At(0))->String();
        TString str1 = ((TObjString*)tokenArr->At(1))->String();
        TString str2 = ((TObjString*)tokenArr->At(2))->String();
        if (str0 != "*" && str1 != "*" && str2 == "*") {
          // e.g. "ITS/Calib/*"
          addFolder = "/" + str0 + "/" + str1;
        } else if (str0 != "*" && str1 == "*" && str2 == "*") {
          // e.g. "ITS/*/*"
          addFolder = "/" + str0;
        } else if (str0 == "*" && str1 == "*" && str2 == "*") {
          // e.g. "*/*/*"
          // do nothing: addFolder is already an empty string;
        } else {
          // e.g. "ITS/*/RecoParam"
          pattern.Prepend(path + "/");
        }
      }
      delete tokenArr;
      tokenArr = nullptr;
    }

    TString folderCopy(Form("%s%s", mDBFolder.Data(), addFolder.Data()));

    LOG(DEBUG) << "mDBFolder = " << folderCopy.Data() << ", pattern = " << pattern.Data()
               << ", filter = " << filter.Data();

    TGridResult* res = gGrid->Query(folderCopy, pattern, filter, optionQuery.Data());

    if (!res) {
      LOG(ERROR) << "GridStorage query failed";
      return nullptr;
    }

    for (int i = 0; i < res->GetEntries(); i++) {
      ConditionId* validFileId = new ConditionId();
      TString filename = res->GetKey(i, "lfn");
      if (filename == "") {
        continue;
      }
      if (filenameToId(filename, *validFileId)) {
        validFileIds.AddLast(validFileId);
      }
    }
    delete res;
  }

  TIter* iter = nullptr;
  if (alreadyLoaded) {
    iter = new TIter(&mValidFileIds);
  } else {
    iter = new TIter(&validFileIds);
  }

  TObjArray selectedIds;
  selectedIds.SetOwner(1);

  // loop on list of valid Ids to select the right version to get.
  // According to query and to the selection criteria list, version can be the highest or exact
  IdPath pathCopy;
  ConditionId* anIdPtr = nullptr;
  ConditionId* dataId = nullptr;
  IdPath queryPath = queryId.getPath();
  while ((anIdPtr = dynamic_cast<ConditionId*>(iter->Next()))) {
    IdPath thisCDBPath = anIdPtr->getPath();
    if (!(queryPath.isSupersetOf(thisCDBPath)) || pathCopy.getPathString() == thisCDBPath.getPathString()) {
      continue;
    }
    pathCopy = thisCDBPath;

    // check the selection criteria list for this query
    ConditionId thisId(*anIdPtr);
    thisId.setVersion(queryId.getVersion());
    if (!thisId.hasVersion()) {
      getSelection(&thisId);
    }

    if (alreadyLoaded) {
      dataId = getId(mValidFileIds, thisId);
    } else {
      dataId = getId(validFileIds, thisId);
    }
    if (dataId) {
      selectedIds.Add(dataId);
    }
  }

  delete iter;
  iter = nullptr;

  // selectedIds contains the Ids of the files matching all requests of query!
  // All the objects are now ready to be retrieved
  iter = new TIter(&selectedIds);
  while ((anIdPtr = dynamic_cast<ConditionId*>(iter->Next()))) {
    TString filename;
    if (!idToFilename(*anIdPtr, filename)) {
      LOG(DEBUG) << "Bad data ID encountered! Subnormal error!";
      continue;
    }

    Condition* anCondition = getConditionFromFile(filename, anIdPtr);

    if (anCondition) {
      result->Add(anCondition);
    }
  }
  delete iter;
  iter = nullptr;

  return result;
}

Bool_t GridStorage::putCondition(Condition* entry, const char* mirrors)
{
  // put an  Condition object into the database

  ConditionId& id = entry->getId();

  // set version for the entry to be stored
  if (!prepareId(id)) {
    return kFALSE;
  }

  // build filename from entry's id
  TString filename;
  if (!idToFilename(id, filename)) {
    LOG(ERROR) << "Bad ID encountered, cannot make a file name out of it!";
    return kFALSE;
  }

  TString folderToTag = Form("%s%s", mDBFolder.Data(), id.getPathString().Data());

  TDirectory* saveDir = gDirectory;

  TString fullFilename = Form("/alien%s", filename.Data());
  TString seMirrors(mirrors);
  if (seMirrors.IsNull() || seMirrors.IsWhitespace()) {
    seMirrors = getMirrorSEs();
  }
  // specify SE to filename
  // if a list of SEs was passed to this method or set via setMirrorSEs, set the first as SE for
  // opening the file.
  // The other SEs will be used in cascade in case of failure in opening the file.
  // The remaining SEs will be used to create replicas.
  TObjArray* arraySEs = seMirrors.Tokenize(',');
  Int_t nSEs = arraySEs->GetEntries();
  Int_t remainingSEs = 1;
  if (nSEs == 0) {
    if (mSE != "default") {
      fullFilename += Form("?se=%s", mSE.Data());
    }
  } else {
    remainingSEs = nSEs;
  }

  // open file
  TFile* file = nullptr;
  TFile* reopenedFile = nullptr;
  LOG(DEBUG) << "mNretry = " << mNretry << ", mInitRetrySeconds = " << mInitRetrySeconds;
  TString targetSE("");

  Bool_t result = kFALSE;
  Bool_t reOpenResult = kFALSE;
  Int_t reOpenAttempts = 0;
  while (!reOpenResult && reOpenAttempts < 2) { // loop to check the file after closing it, to catch
    // the unlikely but possible case when the file
    // is cleaned up by alien just before closing as a consequence of a network disconnection while
    // writing

    while (!file && remainingSEs > 0) {
      if (nSEs != 0) {
        TObjString* target = (TObjString*)arraySEs->At(nSEs - remainingSEs);
        targetSE = target->String();
        if (!(targetSE.BeginsWith("ALICE::") && targetSE.CountChar(':') == 4)) {
          LOG(ERROR) << R"(")" << targetSE.Data() << R"(" is an invalid storage element identifier.)";
          continue;
        }
        if (fullFilename.Contains('?')) {
          fullFilename.Remove(fullFilename.Last('?'));
        }
        fullFilename += Form("?se=%s", targetSE.Data());
      }
      Int_t remainingAttempts = mNretry;
      Int_t nsleep = mInitRetrySeconds; // number of seconds between attempts. We let it increase exponentially
      LOG(DEBUG) << "Uploading file into SE #" << nSEs - remainingSEs + 1 << ": " << targetSE.Data();
      while (remainingAttempts > 0) {
        LOG(DEBUG) << "Uploading file into OCDB at " << targetSE.Data() << " - Attempt #"
                   << mNretry - remainingAttempts + 1;
        remainingAttempts--;
        file = TFile::Open(fullFilename, "CREATE");
        if (!file || !file->IsWritable()) {
          if (file) { // file is not writable
            file->Close();
            delete file;
            file = nullptr;
          }
          TString message(TString::Format("Attempt %d failed.", mNretry - remainingAttempts));
          if (remainingAttempts > 0) {
            message += " Sleeping for ";
            message += nsleep;
            message += " seconds";
          } else {
            if (remainingSEs > 0) {
              message += " Trying to upload at next SE";
            }
          }
          LOG(DEBUG) << message.Data();
          if (remainingAttempts > 0) {
            sleep(nsleep);
          }
        } else {
          remainingAttempts = 0;
        }
        nsleep *= mInitRetrySeconds;
      }
      remainingSEs--;
    }
    if (!file) {
      LOG(ERROR) << "All " << mNretry << " attempts have failed on all " << nSEs << " SEs. Returning...";
      return kFALSE;
    }

    file->cd();

    // setTreeToFile(entry, file);
    entry->setVersion(id.getVersion());

    // write object (key name: " Condition")
    result = (file->WriteTObject(entry, " Condition") != 0);
    file->Close();
    if (!result) {
      LOG(ERROR) << "Can't write entry to file <" << filename.Data() << ">!";
    } else {
      LOG(DEBUG) << "Reopening file " << fullFilename.Data() << " for checking its correctness";
      reopenedFile = TFile::Open(fullFilename.Data(), "READ");
      if (!reopenedFile) {
        reOpenResult = kFALSE;
        LOG(INFO) << R"(The file ")" << fullFilename.Data()
                  << R"(" was closed successfully but cannot be reopened. Trying now to regenerate it (regeneration attempt number )"
                  << ++reOpenAttempts;
        delete file;
        file = nullptr;
        LOG(DEBUG) << "Removing file " << filename.Data();
        if (!gGrid->Rm(filename.Data()))
          LOG(ERROR) << "Can't delete file!";
        remainingSEs++;
      } else {
        reOpenResult = kTRUE;
        if (!Manager::Instance()->isOcdbUploadMode()) {
          reopenedFile->Close();
          delete reopenedFile;
          reopenedFile = nullptr;
        }
      }
    }
  }

  if (saveDir) {
    saveDir->cd();
  } else
    gROOT->cd();
  delete file;
  file = nullptr;

  if (result && reOpenResult) {

    if (!tagFileId(filename, &id)) {
      LOG(INFO) << R"(CDB tagging failed. Deleting file ")" << filename.Data() << R"("!)";
      if (!gGrid->Rm(filename.Data()))
        LOG(ERROR) << "Can't delete file!";
      return kFALSE;
    }

    tagFileConditionMetaData(filename, entry->getConditionMetaData());
  } else {
    LOG(ERROR) << "The file could not be opened or the object could not be written.";
    if (!gGrid->Rm(filename.Data()))
      LOG(ERROR) << "Can't delete file!";
    return kFALSE;
  }

  LOG(INFO) << R"(CDB object stored into file ")" << filename.Data() << R"(" )";
  if (nSEs == 0)
    LOG(INFO) << "Storage Element: " << mSE.Data();
  else
    LOG(INFO) << "Storage Element: " << targetSE.Data();

  // In case of other SEs specified by the user, mirror the file to the remaining SEs
  for (Int_t i = 0; i < nSEs; i++) {
    if (i == nSEs - remainingSEs - 1) {
      continue;
    } // skip mirroring to the SE where the file was saved
    TString mirrorCmd("mirror ");
    mirrorCmd += filename;
    mirrorCmd += " ";
    TObjString* target = (TObjString*)arraySEs->At(i);
    TString mirrorSE(target->String());
    mirrorCmd += mirrorSE;
    LOG(DEBUG) << R"(mirror command: ")" << mirrorCmd.Data() << R"(")";
    LOG(INFO) << "Mirroring to storage element: " << mirrorSE.Data();
    gGrid->Command(mirrorCmd.Data());
  }
  arraySEs->Delete();
  arraySEs = nullptr;

  if (Manager::Instance()->isOcdbUploadMode()) { // if uploading to OCDBs, add to cvmfs too
    if (!filename.BeginsWith("/alice/data") && !filename.BeginsWith("/alice/simulation/2008/v4-15-Release")) {
      LOG(ERROR) << R"(Cannot upload to CVMFS OCDBs a non official CDB object: ")" << filename.Data() << R"("!)";
    } else {
      if (!putInCvmfs(filename, reopenedFile))
        LOG(ERROR) << R"(Could not upload AliEn file ")" << filename.Data() << R"(" to CVMFS OCDB!)";
    }
    reopenedFile->Close();
    delete reopenedFile;
    reopenedFile = nullptr;
  }

  return kTRUE;
}

Bool_t GridStorage::putInCvmfs(TString& filename, TFile* cdbFile) const
{
  // Add the CDB object to cvmfs OCDB

  TString cvmfsFilename(filename);
  // cvmfsFilename.Remove(TString::kTrailing, '/');
  TString basename = (cvmfsFilename(cvmfsFilename.Last('/') + 1, cvmfsFilename.Length()));
  TString cvmfsDirname = cvmfsFilename.Remove(cvmfsFilename.Last('/'), cvmfsFilename.Length());
  TRegexp threeLevelsRE("[^/]+/[^/]+/[^/]+$");
  TString threeLevels = cvmfsDirname(threeLevelsRE);

  TRegexp re_RawFolder("^/alice/data/20[0-9]+/OCDB");
  TRegexp re_MCFolder("^/alice/simulation/2008/v4-15-Release");
  TString rawFolder = cvmfsDirname(re_RawFolder);
  TString mcFolder = cvmfsDirname(re_MCFolder);
  if (!rawFolder.IsNull()) {
    cvmfsDirname.Replace(0, 6, "/cvmfs/alice-ocdb.cern.ch/calibration");
  } else if (!mcFolder.IsNull()) {
    cvmfsDirname.Replace(0, 36, "/cvmfs/alice-ocdb.cern.ch/calibration/MC");
  } else {
    LOG(ERROR) << "OCDB folder set for an invalid OCDB storage:\n   " << cvmfsDirname.Data();
    return kFALSE;
  }
  // now cvmfsDirname is the full dirname in cvmfs
  LOG(DEBUG) << R"(Publishing ")" << basename.Data() << R"(" in ")" << cvmfsDirname.Data() << R"(")";

  // Tar the file with the right prefix path. Include the directory structure in the tarball
  // to cover the case of a containing directory being new in cvmfs, plus a container directory
  // to avoid clashing with stuff present in the local directory
  TString firstLevel(threeLevels(0, threeLevels.First('/')));
  TString tempDir("tmpToCvmfsOcdbs");
  gSystem->Exec(Form("rm -r %s > /dev/null 2>&1", tempDir.Data())); // to be sure not to publish other stuff in cvmfs
  Int_t result = gSystem->Exec(Form("mkdir -p %s/%s", tempDir.Data(), threeLevels.Data()));
  if (result != 0) {
    LOG(ERROR) << R"(Could not create the directory ")" << tempDir.Data() << "/" << threeLevels.Data() << R"(")";
    return kFALSE;
  }
  cdbFile->Cp(Form("%s/%s/%s", tempDir.Data(), threeLevels.Data(), basename.Data()));
  TString tarFileName("cdbObjectToAdd.tar.gz");
  TString cvmfsBaseFolder(cvmfsDirname(0, cvmfsDirname.Last('/')));
  cvmfsBaseFolder = cvmfsBaseFolder(0, cvmfsBaseFolder.Last('/'));
  cvmfsBaseFolder = cvmfsBaseFolder(0, cvmfsBaseFolder.Last('/'));
  // tarCommand should be e.g.: tar --transform
  // 's,^,/cvmfs/alice-ocdb.cern.ch/calibration/data/2010/OCDB/,S' -cvzf objecttoadd.tar.gz basename
  result = gSystem->Exec(Form("tar --transform 's,^%s,%s,S' -cvzf %s %s", tempDir.Data(), cvmfsBaseFolder.Data(),
                              tarFileName.Data(), tempDir.Data()));
  if (result != 0) {
    LOG(ERROR) << R"(Could not create the tarball for the object ")" << filename.Data() << R"(")";
    return kFALSE;
  }

  // Copy the file to cvmfs (requires to have the executable in the path and access to the server)
  result = gSystem->Exec(Form("ocdb-cvmfs %s", tarFileName.Data()));
  if (result != 0) {
    LOG(ERROR) << R"(Could not execute "ocdb-cvmfs )" << filename.Data() << R"(")";
    return kFALSE;
  }

  // Remove the local file and the tar-file
  gSystem->Exec(Form("rm -r %s", tempDir.Data()));
  gSystem->Exec(Form("rm %s", tarFileName.Data()));

  return kTRUE;
}

Bool_t GridStorage::addTag(TString& folderToTag, const char* tagname)
{
  // add "tagname" tag (CDB or CDB_MD) to folder where object will be stored

  Bool_t result = kTRUE;
  LOG(DEBUG) << "adding " << tagname << R"( tag to folder ")" << folderToTag.Data() << R"(")";
  TString addTagCommand = Form("addTag %s %s", folderToTag.Data(), tagname);
  TGridResult* gridres = gGrid->Command(addTagCommand.Data());
  const char* resCode = gridres->GetKey(0, "__result__"); // '1' if success
  if (resCode[0] != '1') {
    LOG(ERROR) << R"(Couldn't add ")" << tagname << R"(" tags to folder )" << folderToTag.Data() << "!";
    result = kFALSE;
  }
  delete gridres;
  return result;
}

Bool_t GridStorage::tagFileId(TString& filename, const ConditionId* id)
{
  // tag stored object in CDB table using object ConditionId's parameters

  TString dirname(filename);
  Int_t dirNumber = gGrid->Mkdir(dirname.Remove(dirname.Last('/')), "-d");

  TString addTagValue1 = Form("addTagValue %s CDB ", filename.Data());
  TString addTagValue2 =
    Form("first_run=%d last_run=%d version=%d ", id->getFirstRun(), id->getLastRun(), id->getVersion());
  TString addTagValue3 = Form(R"(path_level_0="%s" path_level_1="%s" path_level_2="%s" )",
                              id->getPathLevel(0).Data(), id->getPathLevel(1).Data(), id->getPathLevel(2).Data());
  // TString addTagValue4 = Form("version_path=\"%s\"
  // dir_number=%d",Form("%d_%s",id->getVersion(),filename.Data()),dirNumber);
  TString addTagValue4 = Form(R"(version_path="%09d%s" dir_number=%d)", id->getVersion(), filename.Data(), dirNumber);
  TString addTagValue =
    Form("%s%s%s%s", addTagValue1.Data(), addTagValue2.Data(), addTagValue3.Data(), addTagValue4.Data());

  Bool_t result = kFALSE;
  LOG(DEBUG) << "Tagging file. Tag command: " << addTagValue.Data();
  TGridResult* res = gGrid->Command(addTagValue.Data());
  const char* resCode = res->GetKey(0, "__result__"); // '1' if success
  if (resCode[0] != '1') {
    LOG(ERROR) << "Couldn't add CDB tag value to file " << filename.Data() << " !";
    result = kFALSE;
  } else {
    LOG(DEBUG) << "Object successfully tagged.";
    result = kTRUE;
  }
  delete res;
  return result;
}

Bool_t GridStorage::tagFileConditionMetaData(TString& filename, const ConditionMetaData* md)
{
  // tag stored object in CDB table using object ConditionId's parameters

  TString addTagValue1 = Form("addTagValue %s CDB_MD ", filename.Data());
  TString addTagValue2 = Form(R"(object_classname="%s" responsible="%s" beam_period=%d )", md->getObjectClassName(),
                              md->getResponsible(), md->getBeamPeriod());
  TString addTagValue3 = Form(R"(aliroot_version="%s" comment="%s")", md->getAliRootVersion(), md->getComment());
  TString addTagValue = Form("%s%s%s", addTagValue1.Data(), addTagValue2.Data(), addTagValue3.Data());

  Bool_t result = kFALSE;
  LOG(DEBUG) << "Tagging file. Tag command: " << addTagValue.Data();
  TGridResult* res = gGrid->Command(addTagValue.Data());
  const char* resCode = res->GetKey(0, "__result__"); // '1' if success
  if (resCode[0] != '1') {
    LOG(WARNING) << "Couldn't add CDB_MD tag value to file " << filename.Data() << " !";
    result = kFALSE;
  } else {
    LOG(DEBUG) << "Object successfully tagged.";
    result = kTRUE;
  }
  return result;
}

TList* GridStorage::getIdListFromFile(const char* fileName)
{

  TString turl(fileName);
  turl.Prepend("/alien" + mDBFolder);
  turl += "?se=";
  turl += mSE.Data();
  TFile* file = TFile::Open(turl);
  if (!file) {
    LOG(ERROR) << "Can't open selection file <" << turl.Data() << ">!";
    return nullptr;
  }

  TList* list = new TList();
  list->SetOwner();
  int i = 0;
  TString keycycle;

  ConditionId* id;
  while (1) {
    i++;
    keycycle = " ConditionId;";
    keycycle += i;

    id = (ConditionId*)file->Get(keycycle);
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

Bool_t GridStorage::hasConditionType(const char* path) const
{
  // check for path in storage's DBFolder

  TString initDir(gGrid->Pwd(0));
  TString dirName(mDBFolder);
  dirName += path; // dirName = mDBFolder/path
  Bool_t result = kFALSE;
  if (gGrid->Cd(dirName, 0)) {
    result = kTRUE;
  }
  gGrid->Cd(initDir.Data(), 0);
  return result;
}

void GridStorage::queryValidFiles()
{
  // Query the CDB for files valid for  Storage::mRun
  // Fills list mValidFileIds with  ConditionId objects extracted from CDB files
  // selected from AliEn metadata.
  // If mVersion was not set, mValidFileIds is filled with highest versions.

  TString filter;
  makeQueryFilter(mRun, mRun, mConditionMetaDataFilter, filter);

  TString path = mPathFilter.getPathString();

  TString pattern = "Run*";
  TString optionQuery = "-y";
  if (mVersion >= 0) {
    pattern += Form("_v%d_s0", mVersion);
    optionQuery = "";
  }
  pattern += ".root";
  LOG(DEBUG) << "pattern: " << pattern.Data();

  TString addFolder = "";
  if (!path.Contains("*")) {
    if (!path.BeginsWith("/")) {
      addFolder += "/";
    }
    addFolder += path;
  } else {
    if (path.BeginsWith("/")) {
      path.Remove(0, 1);
    }
    if (path.EndsWith("/")) {
      path.Remove(path.Length() - 1, 1);
    }
    TObjArray* tokenArr = path.Tokenize("/");
    if (tokenArr->GetEntries() != 3) {
      LOG(ERROR) << "Not a 3 level path! Keeping old query...";
      pattern.Prepend(path + "/");
    } else {
      TString str0 = ((TObjString*)tokenArr->At(0))->String();
      TString str1 = ((TObjString*)tokenArr->At(1))->String();
      TString str2 = ((TObjString*)tokenArr->At(2))->String();
      if (str0 != "*" && str1 != "*" && str2 == "*") {
        // e.g. "ITS/Calib/*"
        addFolder = "/" + str0 + "/" + str1;
      } else if (str0 != "*" && str1 == "*" && str2 == "*") {
        // e.g. "ITS/*/*"
        addFolder = "/" + str0;
      } else if (str0 == "*" && str1 == "*" && str2 == "*") {
        // e.g. "*/*/*"
        // do nothing: addFolder is already an empty string;
      } else {
        // e.g. "ITS/*/RecoParam"
        pattern.Prepend(path + "/");
      }
    }
    delete tokenArr;
    tokenArr = nullptr;
  }

  TString folderCopy(Form("%s%s", mDBFolder.Data(), addFolder.Data()));

  LOG(DEBUG) << "mDBFolder = " << folderCopy.Data() << ", pattern = " << pattern.Data()
             << ", filter = " << filter.Data();

  if (optionQuery == "-y") {
    LOG(INFO) << "Only latest version will be returned";
  }

  TGridResult* res = gGrid->Query(folderCopy, pattern, filter, optionQuery.Data());

  if (!res) {
    LOG(ERROR) << "GridStorage query failed";
    return;
  }

  TIter next(res);
  TMap* map;
  while ((map = (TMap*)next())) {
    TObjString* entry;
    if ((entry = (TObjString*)((TMap*)map)->GetValue("lfn"))) {
      TString& filename = entry->String();
      if (filename.IsNull()) {
        continue;
      }
      LOG(DEBUG) << "Found valid file: " << filename.Data();
      ConditionId* validFileId = new ConditionId();
      Bool_t result = filenameToId(filename, *validFileId);
      if (result) {
        mValidFileIds.AddLast(validFileId);
      } else {
        delete validFileId;
      }
    }
  }
  delete res;
}

void GridStorage::makeQueryFilter(Int_t firstRun, Int_t lastRun, const ConditionMetaData* md, TString& result) const
{
  // create filter for file query

  result = Form("CDB:first_run<=%d and CDB:last_run>=%d", firstRun, lastRun);

  //    if(version >= 0) {
  //            result += Form(" and CDB:version=%d", version);
  //    }
  //    if(pathFilter.getLevel0() != "*") {
  //            result += Form(" and CDB:path_level_0=\"%s\"", pathFilter.getLevel0().Data());
  //    }
  //    if(pathFilter.getLevel1() != "*") {
  //            result += Form(" and CDB:path_level_1=\"%s\"", pathFilter.getLevel1().Data());
  //    }
  //    if(pathFilter.getLevel2() != "*") {
  //            result += Form(" and CDB:path_level_2=\"%s\"", pathFilter.getLevel2().Data());
  //    }

  if (md) {
    if (md->getObjectClassName()[0] != '\0') {
      result += Form(R"( and CDB_MD:object_classname="%s")", md->getObjectClassName());
    }
    if (md->getResponsible()[0] != '\0') {
      result += Form(R"( and CDB_MD:responsible="%s")", md->getResponsible());
    }
    if (md->getBeamPeriod() != 0) {
      result += Form(" and CDB_MD:beam_period=%d", md->getBeamPeriod());
    }
    if (md->getAliRootVersion()[0] != '\0') {
      result += Form(R"( and CDB_MD:aliroot_version="%s")", md->getAliRootVersion());
    }
    if (md->getComment()[0] != '\0') {
      result += Form(R"( and CDB_MD:comment="%s")", md->getComment());
    }
  }
  LOG(DEBUG) << "filter: " << result.Data();
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                             //
//  GridStorage factory                                                                                //
//                                                                                             //
/////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(GridStorageFactory);

Bool_t GridStorageFactory::validateStorageUri(const char* gridString)
{
  // check if the string is valid GridStorage URI

  TRegexp gridPattern("^alien://.+$");

  return TString(gridString).Contains(gridPattern);
}

StorageParameters* GridStorageFactory::createStorageParameter(const char* gridString)
{
  // create  GridStorageParameters class from the URI string

  if (!validateStorageUri(gridString)) {
    return nullptr;
  }

  TString buffer(gridString);

  TString gridUrl = "alien://";
  TString user = "";
  TString dbFolder = "";
  TString se = "default";
  TString cacheFolder = "";
  Bool_t operateDisconnected = kTRUE;
  Long64_t cacheSize = (UInt_t)1024 * 1024 * 1024; // 1GB
  Long_t cleanupInterval = 0;

  TObjArray* arr = buffer.Tokenize('?');
  TIter iter(arr);
  TObjString* str = nullptr;

  while ((str = (TObjString*)iter.Next())) {
    TString entry(str->String());
    Int_t indeq = entry.Index('=');
    if (indeq == -1) {
      if (entry.BeginsWith("alien://")) { // maybe it's a gridUrl!
        gridUrl = entry;
        continue;
      } else {
        LOG(ERROR) << "Invalid entry! " << entry.Data();
        continue;
      }
    }

    TString key = entry(0, indeq);
    TString value = entry(indeq + 1, entry.Length() - indeq);

    if (key.Contains("grid", TString::kIgnoreCase)) {
      gridUrl += value;
    } else if (key.Contains("user", TString::kIgnoreCase)) {
      user = value;
    } else if (key.Contains("se", TString::kIgnoreCase)) {
      se = value;
    } else if (key.Contains("cacheF", TString::kIgnoreCase)) {
      cacheFolder = value;
      if (!cacheFolder.IsNull() && !cacheFolder.EndsWith("/")) {
        cacheFolder += "/";
      }
    } else if (key.Contains("folder", TString::kIgnoreCase)) {
      dbFolder = value;
    } else if (key.Contains("operateDisc", TString::kIgnoreCase)) {
      if (value == "kTRUE") {
        operateDisconnected = kTRUE;
      } else if (value == "kFALSE") {
        operateDisconnected = kFALSE;
      } else if (value == "0" || value == "1") {
        operateDisconnected = (Bool_t)value.Atoi();
      } else {
        LOG(ERROR) << "Invalid entry! " << entry.Data();
        return nullptr;
      }
    } else if (key.Contains("cacheS", TString::kIgnoreCase)) {
      if (value.IsDigit()) {
        cacheSize = value.Atoi();
      } else {
        LOG(ERROR) << "Invalid entry! " << entry.Data();
        return nullptr;
      }
    } else if (key.Contains("cleanupInt", TString::kIgnoreCase)) {
      if (value.IsDigit()) {
        cleanupInterval = value.Atoi();
      } else {
        LOG(ERROR) << "Invalid entry! " << entry.Data();
        return nullptr;
      }
    } else {
      LOG(ERROR) << "Invalid entry! " << entry.Data();
      return nullptr;
    }
  }
  delete arr;
  arr = nullptr;

  LOG(DEBUG) << "gridUrl:       " << gridUrl.Data();
  LOG(DEBUG) << "user:  " << user.Data();
  LOG(DEBUG) << "dbFolder:      " << dbFolder.Data();
  LOG(DEBUG) << "s.e.:  " << se.Data();
  LOG(DEBUG) << "local cache folder: " << cacheFolder.Data();
  LOG(DEBUG) << "local cache operate disconnected: " << operateDisconnected;
  LOG(DEBUG) << "local cache size: " << cacheSize << "";
  LOG(DEBUG) << "local cache cleanup interval: " << cleanupInterval << "";

  if (dbFolder == "") {
    LOG(ERROR) << "Base folder must be specified!";
    return nullptr;
  }

  return new GridStorageParameters(gridUrl.Data(), user.Data(), dbFolder.Data(), se.Data(), cacheFolder.Data(),
                                   operateDisconnected,
                                   cacheSize, cleanupInterval);
}

Storage* GridStorageFactory::createStorage(const StorageParameters* param)
{
  // create  GridStorage storage instance from parameters
  GridStorage* grid = nullptr;
  if (GridStorageParameters::Class() == param->IsA()) {
    const GridStorageParameters* gridParam = (const GridStorageParameters*)param;
    grid = new GridStorage(gridParam->GridUrl().Data(), gridParam->getUser().Data(), gridParam->getDBFolder().Data(),
                           gridParam->getSE().Data(), gridParam->getCacheFolder().Data(),
                           gridParam->getOperateDisconnected(),
                           gridParam->getCacheSize(), gridParam->getCleanupInterval());
  }
  if (!gGrid && grid) {
    delete grid;
    grid = nullptr;
  }
  return grid;
}

//  GridStorage Parameter class
ClassImp(GridStorageParameters);

GridStorageParameters::GridStorageParameters()
  : StorageParameters(),
    mGridUrl(),
    mUser(),
    mDBFolder(),
    mSE(),
    mCacheFolder(),
    mOperateDisconnected(),
    mCacheSize(),
    mCleanupInterval()
{
  // default constructor
}

GridStorageParameters::GridStorageParameters(const char* gridUrl, const char* user, const char* dbFolder,
                                             const char* se,
                                             const char* cacheFolder, Bool_t operateDisconnected, Long64_t cacheSize,
                                             Long_t cleanupInterval)
  : StorageParameters(),
    mGridUrl(gridUrl),
    mUser(user),
    mDBFolder(dbFolder),
    mSE(se),
    mCacheFolder(cacheFolder),
    mOperateDisconnected(operateDisconnected),
    mCacheSize(cacheSize),
    mCleanupInterval(cleanupInterval)
{
  // constructor
  setType("alien");
  TString uri = Form(
    "%s?User=%s?DBFolder=%s?SE=%s?CacheFolder=%s"
    "?OperateDisconnected=%d?CacheSize=%lld?CleanupInterval=%ld",
    mGridUrl.Data(), mUser.Data(), mDBFolder.Data(), mSE.Data(), mCacheFolder.Data(),
    mOperateDisconnected, mCacheSize, mCleanupInterval);
  setUri(uri.Data());
}

GridStorageParameters::~GridStorageParameters() = default;

StorageParameters* GridStorageParameters::cloneParam() const
{
  // clone parameter
  return new GridStorageParameters(mGridUrl.Data(), mUser.Data(), mDBFolder.Data(), mSE.Data(), mCacheFolder.Data(),
                                   mOperateDisconnected, mCacheSize, mCleanupInterval);
}

ULong_t GridStorageParameters::getHash() const
{
  // return getHash function
  return mGridUrl.Hash() + mUser.Hash() + mDBFolder.Hash() + mSE.Hash() + mCacheFolder.Hash();
}

Bool_t GridStorageParameters::isEqual(const TObject* obj) const
{
  // check if this object is equal to  StorageParameters obj
  if (this == obj) {
    return kTRUE;
  }
  if (GridStorageParameters::Class() != obj->IsA()) {
    return kFALSE;
  }
  GridStorageParameters* other = (GridStorageParameters*)obj;
  if (mGridUrl != other->mGridUrl) {
    return kFALSE;
  }
  if (mUser != other->mUser) {
    return kFALSE;
  }
  if (mDBFolder != other->mDBFolder) {
    return kFALSE;
  }
  if (mSE != other->mSE) {
    return kFALSE;
  }
  if (mCacheFolder != other->mCacheFolder) {
    return kFALSE;
  }
  if (mOperateDisconnected != other->mOperateDisconnected) {
    return kFALSE;
  }
  if (mCacheSize != other->mCacheSize) {
    return kFALSE;
  }
  if (mCleanupInterval != other->mCleanupInterval) {
    return kFALSE;
  }
  return kTRUE;
}
