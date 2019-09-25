// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// access class to a DataBase in a local storage                                               //

#include "CCDB/LocalStorage.h"
#include <fairlogger/Logger.h> // for LOG
#include <TFile.h>             // for TFile
#include <TObjString.h>        // for TObjString
#include <TRegexp.h>           // for TRegexp
#include <TSystem.h>           // for TSystem, gSystem
#include "CCDB/Condition.h"    // for Condition

using namespace o2::ccdb;

ClassImp(LocalStorage);

LocalStorage::LocalStorage(const char* baseDir) : mBaseDirectory(baseDir)
{
  // constructor

  LOG(DEBUG) << "mBaseDirectory = " << mBaseDirectory.Data();

  // check baseDire: trying to cd to baseDir; if it does not exist, create it
  void* dir = gSystem->OpenDirectory(baseDir);
  if (dir == nullptr) {
    if (gSystem->mkdir(baseDir, kTRUE)) {
      LOG(ERROR) << R"(Can't open directory ")" << baseDir << R"("!)"; //!!!!!!!! to be commented out for testing
    }

  } else {
    LOG(DEBUG) << R"(Folder ")" << mBaseDirectory.Data() << R"(" found)";
    gSystem->FreeDirectory(dir);
  }
  mType = "local";
  mBaseFolder = mBaseDirectory;
}

LocalStorage::~LocalStorage() = default;

Bool_t LocalStorage::filenameToId(const char* filename, IdRunRange& runRange, Int_t& version, Int_t& subVersion)
{
  // build  ConditionId from filename numbers

  Ssiz_t mSize;

  // valid filename: Run#firstRun_#lastRun_v#version_s#subVersion.root
  TRegexp keyPattern("^Run[0-9]+_[0-9]+_v[0-9]+_s[0-9]+.root$");
  keyPattern.Index(filename, &mSize);
  if (!mSize) {
    LOG(DEBUG) << "Bad filename <" << filename << ">.";
    return kFALSE;
  }

  TString idString(filename);
  idString.Resize(idString.Length() - sizeof(".root") + 1);

  TObjArray* strArray = (TObjArray*)idString.Tokenize("_");

  TString firstRunString(((TObjString*)strArray->At(0))->GetString());
  runRange.setFirstRun(atoi(firstRunString.Data() + 3));
  runRange.setLastRun(atoi(((TObjString*)strArray->At(1))->GetString()));

  TString verString(((TObjString*)strArray->At(2))->GetString());
  version = atoi(verString.Data() + 1);

  TString subVerString(((TObjString*)strArray->At(3))->GetString());
  subVersion = atoi(subVerString.Data() + 1);

  delete strArray;

  return kTRUE;
}

Bool_t LocalStorage::idToFilename(const ConditionId& id, TString& filename) const
{
  // build file name from  ConditionId data (run range, version, subVersion)

  LOG(DEBUG) << "mBaseDirectory = " << mBaseDirectory.Data();

  if (!id.getIdRunRange().isValid()) {
    LOG(DEBUG) << R"(Invalid run range ")" << id.getFirstRun() << "," << id.getLastRun() << R"(".)";
    return kFALSE;
  }

  if (id.getVersion() < 0) {
    LOG(DEBUG) << "Invalid version <" << id.getVersion() << ">.";
    return kFALSE;
  }

  if (id.getSubVersion() < 0) {
    LOG(DEBUG) << "Invalid subversion <" << id.getSubVersion() << ">.";
    return kFALSE;
  }

  filename = Form("Run%d_%d_v%d_s%d.root", id.getFirstRun(), id.getLastRun(), id.getVersion(), id.getSubVersion());

  filename.Prepend(mBaseDirectory + '/' + id.getPathString() + '/');

  return kTRUE;
}

Bool_t LocalStorage::prepareId(ConditionId& id)
{
  // prepare id (version, subVersion) of the object that will be stored (called by putCondition)

  TString dirName = Form("%s/%s", mBaseDirectory.Data(), id.getPathString().Data());

  // go to the path; if directory does not exist, create it
  void* dirPtr = gSystem->OpenDirectory(dirName);
  if (!dirPtr) {
    gSystem->mkdir(dirName, kTRUE);
    dirPtr = gSystem->OpenDirectory(dirName);

    if (!dirPtr) {
      LOG(ERROR) << R"(Can't create directory ")" << dirName.Data() << R"("!)";
      return kFALSE;
    }
  }

  const char* filename;
  IdRunRange aIdRunRange;                     // the runRange got from filename
  IdRunRange lastIdRunRange(-1, -1);          // highest runRange found
  Int_t aVersion, aSubVersion;                // the version subVersion got from filename
  Int_t lastVersion = 0, lastSubVersion = -1; // highest version and subVersion found

  if (!id.hasVersion()) { // version not specified: look for highest version & subVersion

    while ((filename = gSystem->GetDirEntry(dirPtr))) { // loop on the files

      TString aString(filename);
      if (aString == "." || aString == "..") {
        continue;
      }

      if (!filenameToId(filename, aIdRunRange, aVersion, aSubVersion)) {
        LOG(DEBUG) << "Bad filename <" << filename << ">! I'll skip it.";
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

    while ((filename = gSystem->GetDirEntry(dirPtr))) { // loop on the files

      TString aString(filename);
      if (aString == "." || aString == "..") {
        continue;
      }

      if (!filenameToId(filename, aIdRunRange, aVersion, aSubVersion)) {
        LOG(DEBUG) << "Bad filename <" << filename << ">!I'll skip it.";
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

  gSystem->FreeDirectory(dirPtr);

  TString lastStorage = id.getLastStorage();
  if (lastStorage.Contains(TString("grid"), TString::kIgnoreCase) && id.getSubVersion() > 0) {
    LOG(ERROR) << "GridStorage to LocalStorage Storage error! local object with version v" << id.getVersion() << "_s"
               << id.getSubVersion() - 1 << " found:";
    LOG(ERROR) << "This object has been already transferred from GridStorage (check v" << id.getVersion() << "_s0)!";
    return kFALSE;
  }

  if (lastStorage.Contains(TString("new"), TString::kIgnoreCase) && id.getSubVersion() > 0) {
    LOG(DEBUG) << "A NEW object is being stored with version v" << id.getVersion() << "_s" << id.getSubVersion();
    LOG(DEBUG) << "and it will hide previously stored object with v" << id.getVersion() << "_s"
               << id.getSubVersion() - 1 << "!";
  }

  if (!lastIdRunRange.isAnyRange() && !(lastIdRunRange.isEqual(&id.getIdRunRange())))
    LOG(WARNING) << "Run range modified w.r.t. previous version (Run" << lastIdRunRange.getFirstRun() << "_"
                 << lastIdRunRange.getLastRun() << "_v" << id.getVersion() << "_s" << id.getSubVersion() - 1;

  return kTRUE;
}

ConditionId* LocalStorage::getId(const ConditionId& query)
{
  // look for filename matching query (called by getConditionId)

  // if querying for mRun and not specifying a version, look in the mValidFileIds list
  if (!Manager::Instance()->getCvmfsOcdbTag().IsNull() && query.getFirstRun() == mRun && !query.hasVersion()) {
    // if(query.getFirstRun() == mRun && !query.hasVersion()) {
    // get id from mValidFileIds
    TIter iter(&mValidFileIds);

    ConditionId* anIdPtr = nullptr;
    ConditionId* result = nullptr;

    while ((anIdPtr = dynamic_cast<ConditionId*>(iter.Next()))) {
      if (anIdPtr->getPathString() == query.getPathString()) {
        result = new ConditionId(*anIdPtr);
        break;
      }
    }
    return result;
  }

  // otherwise browse in the local filesystem CDB storage
  TString dirName = Form("%s/%s", mBaseDirectory.Data(), query.getPathString().Data());

  void* dirPtr = gSystem->OpenDirectory(dirName);
  if (!dirPtr) {
    LOG(DEBUG) << "Directory <" << (query.getPathString()).Data() << "> not found";
    LOG(DEBUG) << "in DB folder " << mBaseDirectory.Data();
    return nullptr;
  }

  const char* filename;
  ConditionId* result = new ConditionId();
  result->setPath(query.getPathString());

  IdRunRange aIdRunRange;      // the runRange got from filename
  Int_t aVersion, aSubVersion; // the version and subVersion got from filename

  if (!query.hasVersion()) { // neither version and subversion specified -> look for highest version
    // and subVersion

    while ((filename = gSystem->GetDirEntry(dirPtr))) { // loop on files

      TString aString(filename);
      if (aString.BeginsWith('.')) {
        continue;
      }

      if (!filenameToId(filename, aIdRunRange, aVersion, aSubVersion)) {
        continue;
      }
      // aIdRunRange, aVersion, aSubVersion filled from filename

      if (!aIdRunRange.isSupersetOf(query.getIdRunRange())) {
        continue;
      }
      // aIdRunRange contains requested run!

      LOG(DEBUG) << "Filename " << filename << " matches\n";

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
        LOG(ERROR) << "More than one object valid for run " << query.getFirstRun() << " version " << aVersion << "_"
                   << aSubVersion << "!";
        gSystem->FreeDirectory(dirPtr);
        delete result;
        return nullptr;
      }
    }

  } else if (!query.hasSubVersion()) { // version specified but not subversion -> look for highest
    // subVersion
    result->setVersion(query.getVersion());

    while ((filename = gSystem->GetDirEntry(dirPtr))) { // loop on files

      TString aString(filename);
      if (aString.BeginsWith('.')) {
        continue;
      }

      if (!filenameToId(filename, aIdRunRange, aVersion, aSubVersion)) {
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
                   << aSubVersion << "!";
        gSystem->FreeDirectory(dirPtr);
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

    // ConditionId dataId(queryId.getPathString(), -1, -1, -1, -1);
    // Bool_t result;
    while ((filename = gSystem->GetDirEntry(dirPtr))) { // loop on files

      TString aString(filename);
      if (aString.BeginsWith('.')) {
        continue;
      }

      if (!filenameToId(filename, aIdRunRange, aVersion, aSubVersion)) {
        LOG(DEBUG) << "Could not make id from file: " << filename;
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

      result->setVersion(aVersion);
      result->setSubVersion(aSubVersion);
      result->setFirstRun(aIdRunRange.getFirstRun());
      result->setLastRun(aIdRunRange.getLastRun());
      break;
    }
  }

  gSystem->FreeDirectory(dirPtr);

  return result;
}

Condition* LocalStorage::getCondition(const ConditionId& queryId)
{
  // get  Condition from the storage (the CDB file matching the query is
  // selected by getConditionId and the contained  id is passed here)

  ConditionId* dataId = getConditionId(queryId);

  TString errMessage(TString::Format("No valid CDB object found! request was: %s", queryId.ToString().Data()));
  if (!dataId || !dataId->isSpecified()) {
    LOG(ERROR) << "No file found matching this id!";
    throw std::runtime_error(errMessage.Data());
    return nullptr;
  }

  TString filename;
  if (!idToFilename(*dataId, filename)) {
    LOG(ERROR) << "Bad data ID encountered!";
    delete dataId;
    throw std::runtime_error(errMessage.Data());
    return nullptr;
  }

  TFile file(filename, "READ"); // open file
  if (!file.IsOpen()) {
    LOG(ERROR) << "Can't open file <" << filename.Data() << ">!";
    delete dataId;
    throw std::runtime_error(errMessage.Data());
    return nullptr;
  }

  // get the only  Condition object from the file
  // the object in the file is an  Condition entry named " Condition"

  Condition* anCondition = dynamic_cast<Condition*>(file.Get(" Condition"));
  if (!anCondition) {
    LOG(ERROR) << "Bad storage data: No  Condition in file!";
    file.Close();
    delete dataId;
    throw std::runtime_error(errMessage.Data());
    return nullptr;
  }

  ConditionId& entryId = anCondition->getId();

  // The object's ConditionId are not reset during storage
  // If object's ConditionId runRange or version do not match with filename,
  // it means that someone renamed file by hand. In this case a warning msg is issued.

  anCondition->setLastStorage("local");

  if (!entryId.isEqual(dataId)) {
    LOG(WARNING) << "Mismatch between file name and object's ConditionId!";
    LOG(WARNING) << "File name: " << dataId->ToString().Data();
    LOG(WARNING) << "Object's ConditionId: " << entryId.ToString().Data();
  }

  // Check whether entry contains a TTree. In case load the tree in memory!
  loadTreeFromFile(anCondition);

  // close file, return retrieved entry
  file.Close();
  delete dataId;

  return anCondition;
}

ConditionId* LocalStorage::getConditionId(const ConditionId& queryId)
{
  // get  ConditionId from the storage
  // Via getId, select the CDB file matching the query and return
  // the contained  ConditionId

  ConditionId* dataId = nullptr;

  // look for a filename matching query requests (path, runRange, version, subVersion)
  if (!queryId.hasVersion()) {
    // if version is not specified, first check the selection criteria list
    ConditionId selectedId(queryId);
    getSelection(&selectedId);
    dataId = getId(selectedId);
  } else {
    dataId = getId(queryId);
  }

  if (dataId && !dataId->isSpecified()) {
    delete dataId;
    return nullptr;
  }

  return dataId;
}

void LocalStorage::getEntriesForLevel0(const char* level0, const ConditionId& queryId, TList* result)
{
  // multiple request ( Storage::GetAllObjects)

  TString level0Dir = Form("%s/%s", mBaseDirectory.Data(), level0);

  void* level0DirPtr = gSystem->OpenDirectory(level0Dir);
  if (!level0DirPtr) {
    LOG(DEBUG) << "Can't open level0 directory <" << level0Dir.Data() << ">!";
    return;
  }

  const char* level1;
  Long_t flag = 0;
  while ((level1 = gSystem->GetDirEntry(level0DirPtr))) {

    TString level1Str(level1);
    // skip directories starting with a dot (".svn" and similar in old svn working copies)
    if (level1Str.BeginsWith('.')) {
      continue;
    }

    TString fullPath = Form("%s/%s", level0Dir.Data(), level1);

    Int_t res = gSystem->GetPathInfo(fullPath.Data(), nullptr, (Long64_t*)nullptr, &flag, nullptr);

    if (res) {
      LOG(DEBUG) << "Error reading entry " << level1Str.Data() << " !";
      continue;
    }
    if (!(flag & 2)) {
      continue;
    } // bit 1 of flag = directory!

    if (queryId.getPath().doesLevel1Contain(level1)) {
      getEntriesForLevel1(level0, level1, queryId, result);
    }
  }

  gSystem->FreeDirectory(level0DirPtr);
}

void LocalStorage::getEntriesForLevel1(const char* level0, const char* level1, const ConditionId& queryId,
                                       TList* result)
{
  // multiple request ( Storage::GetAllObjects)

  TString level1Dir = Form("%s/%s/%s", mBaseDirectory.Data(), level0, level1);

  void* level1DirPtr = gSystem->OpenDirectory(level1Dir);
  if (!level1DirPtr) {
    LOG(DEBUG) << "Can't open level1 directory <" << level1Dir.Data() << ">!";
    return;
  }

  const char* level2;
  Long_t flag = 0;
  while ((level2 = gSystem->GetDirEntry(level1DirPtr))) {

    TString level2Str(level2);
    // skip directories starting with a dot (".svn" and similar in old svn working copies)
    if (level2Str.BeginsWith('.')) {
      continue;
    }

    TString fullPath = Form("%s/%s", level1Dir.Data(), level2);

    Int_t res = gSystem->GetPathInfo(fullPath.Data(), nullptr, (Long64_t*)nullptr, &flag, nullptr);

    if (res) {
      LOG(DEBUG) << "Error reading entry " << level2Str.Data() << " !";
      continue;
    }
    if (!(flag & 2)) {
      continue;
    } // skip if not a directory

    if (queryId.getPath().doesLevel2Contain(level2)) {

      IdPath entryPath(level0, level1, level2);

      ConditionId entryId(entryPath, queryId.getIdRunRange(), queryId.getVersion(), queryId.getSubVersion());

      // check filenames to see if any includes queryId.getIdRunRange()
      void* level2DirPtr = gSystem->OpenDirectory(fullPath);
      if (!level2DirPtr) {
        LOG(DEBUG) << "Can't open level2 directory <" << fullPath.Data() << ">!";
        return;
      }
      const char* level3;
      Long_t file_flag = 0;
      while ((level3 = gSystem->GetDirEntry(level2DirPtr))) {
        TString fileName(level3);
        TString fullFileName = Form("%s/%s", fullPath.Data(), level3);

        Int_t file_res = gSystem->GetPathInfo(fullFileName.Data(), nullptr, (Long64_t*)nullptr, &file_flag, nullptr);

        if (file_res) {
          LOG(DEBUG) << "Error reading entry " << level2Str.Data() << " !";
          continue;
        }
        if (file_flag) {
          continue;
        } // it is not a regular file!

        // skip if result already contains an entry for this path
        Bool_t alreadyLoaded = kFALSE;
        Int_t nEntries = result->GetEntries();
        for (int i = 0; i < nEntries; i++) {
          Condition* lCondition = (Condition*)result->At(i);
          ConditionId lId = lCondition->getId();
          TString lIdPath = lId.getPathString();
          if (lIdPath.EqualTo(entryPath.getPathString())) {
            alreadyLoaded = kTRUE;
            break;
          }
        }
        if (alreadyLoaded) {
          continue;
        }

        // skip filenames not matching the regex below
        TRegexp re("^Run[0-9]+_[0-9]+_");
        if (!fileName.Contains(re)) {
          continue;
        }
        // Extract first- and last-run and version and subversion.
        // This allows to avoid quering for a calibration path if we did not find a filename with
        // run-range including the one specified in the query and
        // with version, subversion matching the query
        TString fn = fileName(3, fileName.Length() - 3);
        TString firstRunStr = fn(0, fn.First('_'));
        fn.Remove(0, firstRunStr.Length() + 1);
        TString lastRunStr = fn(0, fn.First('_'));
        fn.Remove(0, lastRunStr.Length() + 1);
        TString versionStr = fn(1, fn.First('_') - 1);
        fn.Remove(0, versionStr.Length() + 2);
        TString subvStr = fn(1, fn.First('.') - 1);
        Int_t firstRun = firstRunStr.Atoi();
        Int_t lastRun = lastRunStr.Atoi();
        IdRunRange rr(firstRun, lastRun);
        Int_t version = versionStr.Atoi();
        Int_t subVersion = subvStr.Atoi();

        Condition* anCondition = nullptr;
        Bool_t versionOK = kTRUE, subVersionOK = kTRUE;
        if (queryId.hasVersion() && version != queryId.getVersion()) {
          versionOK = kFALSE;
        }
        if (queryId.hasSubVersion() && subVersion != queryId.getSubVersion()) {
          subVersionOK = kFALSE;
        }
        if (rr.isSupersetOf(queryId.getIdRunRange()) && versionOK && subVersionOK) {
          anCondition = getCondition(entryId);
          result->Add(anCondition);
        }
      }
    }
  }

  gSystem->FreeDirectory(level1DirPtr);
}

TList* LocalStorage::getAllEntries(const ConditionId& queryId)
{
  // return list of CDB entries matching a generic request (Storage::GetAllObjects)

  TList* result = new TList();
  result->SetOwner();

  // if querying for mRun and not specifying a version, look in the mValidFileIds list
  if (queryId.getFirstRun() == mRun && !queryId.hasVersion()) {
    // get id from mValidFileIds
    TIter* iter = new TIter(&mValidFileIds);
    TObjArray selectedIds;
    selectedIds.SetOwner(1);

    // loop on list of valid Ids to select the right version to get.
    // According to query and to the selection criteria list, version can be the highest or exact
    ConditionId* anIdPtr = nullptr;
    ConditionId* dataId = nullptr;
    IdPath queryPath = queryId.getPathString();
    while ((anIdPtr = dynamic_cast<ConditionId*>(iter->Next()))) {
      IdPath thisCDBPath = anIdPtr->getPathString();
      if (!(queryPath.isSupersetOf(thisCDBPath))) {
        continue;
      }

      ConditionId thisId(*anIdPtr);
      dataId = getId(thisId);
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
      Condition* anCondition = getCondition(*anIdPtr);
      if (anCondition) {
        result->Add(anCondition);
      }
    }
    delete iter;
    iter = nullptr;
    return result;
  }

  void* storageDirPtr = gSystem->OpenDirectory(mBaseDirectory);
  if (!storageDirPtr) {
    LOG(DEBUG) << "Can't open storage directory <" << mBaseDirectory.Data() << ">";
    return nullptr;
  }

  const char* level0;
  Long_t flag = 0;
  while ((level0 = gSystem->GetDirEntry(storageDirPtr))) {

    TString level0Str(level0);
    // skip directories starting with a dot (".svn" and similar in old svn working copies)
    if (level0Str.BeginsWith('.')) {
      continue;
    }

    TString fullPath = Form("%s/%s", mBaseDirectory.Data(), level0);

    Int_t res = gSystem->GetPathInfo(fullPath.Data(), nullptr, (Long64_t*)nullptr, &flag, nullptr);

    if (res) {
      LOG(DEBUG) << "Error reading entry " << level0Str.Data() << " !";
      continue;
    }

    if (!(flag & 2)) {
      continue;
    } // bit 1 of flag = directory!

    if (queryId.getPath().doesLevel0Contain(level0)) {
      getEntriesForLevel0(level0, queryId, result);
    }
  }

  gSystem->FreeDirectory(storageDirPtr);

  return result;
}

Bool_t LocalStorage::putCondition(Condition* entry, const char* mirrors)
{
  // put an  Condition object into the database

  ConditionId& id = entry->getId();

  // set version and subVersion for the entry to be stored
  if (!prepareId(id)) {
    return kFALSE;
  }

  // build filename from entry's id
  TString filename = "";
  if (!idToFilename(id, filename)) {

    LOG(DEBUG) << "Bad ID encountered! Subnormal error!";
    return kFALSE;
  }

  TString mirrorsString(mirrors);
  if (!mirrorsString.IsNull())
    LOG(WARNING) << " LocalStorage storage cannot take mirror SEs into account. They will be ignored.";

  // open file
  TFile file(filename, "CREATE");
  if (!file.IsOpen()) {
    LOG(ERROR) << "Can't open file <" << filename.Data() << ">!";
    return kFALSE;
  }

  // setTreeToFile(entry, &file);

  entry->setVersion(id.getVersion());
  entry->setSubVersion(id.getSubVersion());

  // write object (key name: " Condition")
  Bool_t result = file.WriteTObject(entry, " Condition");
  if (!result)
    LOG(DEBUG) << "Can't write entry to file: " << filename.Data();

  file.Close();
  if (result) {
    if (!(id.getPathString().Contains("SHUTTLE/STATUS")))
      LOG(INFO) << R"(CDB object stored into file ")" << filename.Data() << R"(")";
  }

  return result;
}

TList* LocalStorage::getIdListFromFile(const char* fileName)
{

  TString fullFileName(fileName);
  fullFileName.Prepend(mBaseDirectory + '/');
  TFile* file = TFile::Open(fullFileName);
  if (!file) {
    LOG(ERROR) << "Can't open selection file <" << fullFileName.Data() << ">!";
    return nullptr;
  }
  file->cd();

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

Bool_t LocalStorage::hasConditionType(const char* path) const
{
  // check for path in storage's mBaseDirectory

  TString dirName = Form("%s/%s", mBaseDirectory.Data(), path);
  Bool_t result = kFALSE;

  void* dirPtr = gSystem->OpenDirectory(dirName);
  if (dirPtr) {
    result = kTRUE;
  }
  gSystem->FreeDirectory(dirPtr);

  return result;
}

void LocalStorage::queryValidFiles()
{
  // Query the CDB for files valid for  Storage::mRun.
  // Fills list mValidFileIds with  ConditionId objects extracted from CDB files
  // present in the local storage.
  // If mVersion was not set, mValidFileIds is filled with highest versions.
  // In the CVMFS case, the mValidFileIds is filled from the file containing
  // the filepaths corresponding to the highest versions for the give OCDB tag
  // by launching the script which extracts the last versions for the given run.
  //

  if (mVersion != -1)
    LOG(WARNING) << "Version parameter is not used by local storage query!";
  if (mConditionMetaDataFilter) {
    LOG(WARNING) << "CDB meta data parameters are not used by local storage query!";
    delete mConditionMetaDataFilter;
    mConditionMetaDataFilter = nullptr;
  }

  // Check if in CVMFS case
  TString cvmfsOcdbTag(gSystem->Getenv("OCDB_PATH"));
  if (!cvmfsOcdbTag.IsNull()) {
    queryValidCVMFSFiles(cvmfsOcdbTag);
    return;
  }

  void* storageDirPtr = gSystem->OpenDirectory(mBaseDirectory);

  const char* level0;
  while ((level0 = gSystem->GetDirEntry(storageDirPtr))) {

    TString level0Str(level0);
    if (level0Str.BeginsWith(".")) {
      continue;
    }

    if (mPathFilter.doesLevel0Contain(level0)) {
      TString level0Dir = Form("%s/%s", mBaseDirectory.Data(), level0);
      void* level0DirPtr = gSystem->OpenDirectory(level0Dir);
      const char* level1;
      while ((level1 = gSystem->GetDirEntry(level0DirPtr))) {

        TString level1Str(level1);
        if (level1Str.BeginsWith(".")) {
          continue;
        }

        if (mPathFilter.doesLevel1Contain(level1)) {
          TString level1Dir = Form("%s/%s/%s", mBaseDirectory.Data(), level0, level1);

          void* level1DirPtr = gSystem->OpenDirectory(level1Dir);
          const char* level2;
          while ((level2 = gSystem->GetDirEntry(level1DirPtr))) {

            TString level2Str(level2);
            if (level2Str.BeginsWith(".")) {
              continue;
            }

            if (mPathFilter.doesLevel2Contain(level2)) {
              TString dirName = Form("%s/%s/%s/%s", mBaseDirectory.Data(), level0, level1, level2);

              void* dirPtr = gSystem->OpenDirectory(dirName);

              const char* filename;

              IdRunRange aIdRunRange;                // the runRange got from filename
              IdRunRange hvIdRunRange;               // the runRange of the highest version valid file
              Int_t aVersion, aSubVersion;           // the version and subVersion got from filename
              Int_t highestV = -1, highestSubV = -1; // the highest version and subVersion for this calibration type

              while ((filename = gSystem->GetDirEntry(dirPtr))) { // loop on files

                TString aString(filename);
                if (aString.BeginsWith(".")) {
                  continue;
                }

                if (!filenameToId(filename, aIdRunRange, aVersion, aSubVersion)) {
                  continue;
                }

                IdRunRange runrg(mRun, mRun);
                if (!aIdRunRange.isSupersetOf(runrg)) {
                  continue;
                }

                // check to keep the highest version/subversion (in case of more than one)
                if (aVersion > highestV) {
                  highestV = aVersion;
                  highestSubV = aSubVersion;
                  hvIdRunRange = aIdRunRange;
                } else if (aVersion == highestV) {
                  if (aSubVersion > highestSubV) {
                    highestSubV = aSubVersion;
                    hvIdRunRange = aIdRunRange;
                  }
                }
              }
              if (highestV >= 0) {
                IdPath validPath(level0, level1, level2);
                ConditionId* validId = new ConditionId(validPath, hvIdRunRange, highestV, highestSubV);
                mValidFileIds.AddLast(validId);
              }

              gSystem->FreeDirectory(dirPtr);
            }
          }
          gSystem->FreeDirectory(level1DirPtr);
        }
      }
      gSystem->FreeDirectory(level0DirPtr);
    }
  }
  gSystem->FreeDirectory(storageDirPtr);
}

void LocalStorage::queryValidCVMFSFiles(TString& cvmfsOcdbTag)
{
  // Called in the CVMFS case to fill the mValidFileIds from the file containing
  // the filepaths corresponding to the highest versions for the given OCDB tag
  // by launching the script which extracts the last versions for the given run.
  //

  TString command = cvmfsOcdbTag;
  LOG(DEBUG) << R"(Getting valid files from CVMFS-OCDB tag ")" << cvmfsOcdbTag.Data() << R"(")";
  // CVMFS-OCDB tag. This is the file $OCDB_PATH/catalogue/20??.list.gz
  // containing all CDB file paths (for the given AR tag)
  cvmfsOcdbTag.Strip(TString::kTrailing, '/');
  cvmfsOcdbTag.Append("/");
  gSystem->ExpandPathName(cvmfsOcdbTag);
  if (gSystem->AccessPathName(cvmfsOcdbTag))
    LOG(FATAL) << "cvmfs OCDB set to an invalid path: " << cvmfsOcdbTag.Data();

  // The file containing the list of valid files for the current run has to be generated
  // by running the (shell+awk) script on the CVMFS OCDB tag file.

  // the script in cvmfs to extract CDB filepaths for the given run has the following fullpath
  // w.r.t. $OCDB_PATH: bin/OCDBperRun.sh
  command = command.Strip(TString::kTrailing, '/');
  command.Append("/bin/getOCDBFilesPerRun.sh ");
  command += cvmfsOcdbTag;
  // from URI define the last two levels of the path of the cvmfs ocdb tag (e.g. data/2012.list.gz)
  TString uri(getUri());
  uri.Remove(TString::kTrailing, '/');
  TObjArray* osArr = uri.Tokenize('/');
  TObjString* mcdata_os = dynamic_cast<TObjString*>(osArr->At(osArr->GetEntries() - 3));
  TObjString* yeartype_os = nullptr;
  TString mcdata = mcdata_os->GetString();
  if (mcdata == TString("data")) {
    yeartype_os = dynamic_cast<TObjString*>(osArr->At(osArr->GetEntries() - 2));
  } else {
    mcdata_os = dynamic_cast<TObjString*>(osArr->At(osArr->GetEntries() - 2));
    yeartype_os = dynamic_cast<TObjString*>(osArr->At(osArr->GetEntries() - 1));
  }
  mcdata = mcdata_os->GetString();
  TString yeartype = yeartype_os->GetString();
  command += mcdata;
  command += '/';
  command += yeartype;
  command += ".list.gz cvmfs ";
  command += TString::Itoa(mRun, 10);
  command += ' ';
  command += TString::Itoa(mRun, 10);
  command += " -y > ";
  TString runValidFile(gSystem->WorkingDirectory());
  runValidFile += '/';
  runValidFile += mcdata;
  runValidFile += '_';
  runValidFile += yeartype;
  runValidFile += '_';
  runValidFile += TString::Itoa(mRun, 10);
  command += runValidFile;
  LOG(DEBUG) << R"(Running command: ")" << command.Data() << R"(")";
  Int_t result = gSystem->Exec(command.Data());
  if (result != 0) {
    LOG(ERROR) << R"(Was not able to execute ")" << command.Data() << R"(")";
  }

  // We expect the file with valid paths for this run to be generated in the current directory
  // and to be named as the CVMFS OCDB tag, without .gz, with '_runnumber' appended
  // Fill mValidFileIds from file
  std::ifstream file(runValidFile.Data());
  if (!file.is_open()) {
    LOG(FATAL) << R"(Error opening file ")" << runValidFile.Data() << R"("!)";
  }
  TString filepath;
  while (filepath.ReadLine(file)) {
    // skip line in case it is not a root file path
    if (!filepath.EndsWith(".root")) {
      continue;
    }
    // extract three-level path and basename
    TObjArray* tokens = filepath.Tokenize('/');
    if (tokens->GetEntries() < 5) {
      LOG(ERROR) << R"(")" << filepath.Data() << R"(" is not a valid cvmfs path for an OCDB object)";
      continue;
    }
    TObjString* baseNameOstr = (TObjString*)tokens->At(tokens->GetEntries() - 1);
    TString baseName(baseNameOstr->String());
    TObjString* l0oStr = (TObjString*)tokens->At(tokens->GetEntries() - 4);
    TObjString* l1oStr = (TObjString*)tokens->At(tokens->GetEntries() - 3);
    TObjString* l2oStr = (TObjString*)tokens->At(tokens->GetEntries() - 2);
    TString l0(l0oStr->String());
    TString l1(l1oStr->String());
    TString l2(l2oStr->String());
    TString threeLevels = l0 + '/' + l1 + '/' + l2;

    IdPath validPath(threeLevels);
    // use basename and three-level path to create ConditionId
    IdRunRange aIdRunRange;      // the runRange got from filename
    Int_t aVersion, aSubVersion; // the version and subVersion got from filename
    if (!filenameToId(baseName, aIdRunRange, aVersion, aSubVersion))
      LOG(ERROR) << R"(Could not create a valid CDB id from path: ")" << filepath.Data() << R"(")";

    IdRunRange runrg(mRun, mRun);
    if (!aIdRunRange.isSupersetOf(runrg)) {
      continue;
    } // should never happen (would mean awk script wrong output)
    // aIdRunRange contains requested run!
    ConditionId* validId = new ConditionId(validPath, aIdRunRange, aVersion, aSubVersion);
    mValidFileIds.AddLast(validId);
  }

  file.close();
  return;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                             //
//  LocalStorage factory                                                                       //
//                                                                                             //
/////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(LocalStorageFactory);

Bool_t LocalStorageFactory::validateStorageUri(const char* dbString)
{
  // check if the string is valid local URI

  TRegexp dbPatternLocalStorage("^local://.+$");

  return (TString(dbString).Contains(dbPatternLocalStorage) || TString(dbString).BeginsWith("snapshot://folder="));
}

StorageParameters* LocalStorageFactory::createStorageParameter(const char* dbString)
{
  // create  LocalStorageParameters class from the URI string

  if (!validateStorageUri(dbString)) {
    return nullptr;
  }

  TString checkSS(dbString);
  if (checkSS.BeginsWith("snapshot://")) {
    TString snapshotPath("OCDB");
    snapshotPath.Prepend(TString(gSystem->WorkingDirectory()) + '/');
    checkSS.Remove(0, checkSS.First(':') + 3);
    return new LocalStorageParameters(snapshotPath, checkSS);
  }

  // if the string argument is not a snapshot URI, than it is a plain local URI
  TString pathname(dbString + sizeof("local://") - 1);

  if (gSystem->ExpandPathName(pathname)) {
    return nullptr;
  }

  if (pathname[0] != '/') {
    pathname.Prepend(TString(gSystem->WorkingDirectory()) + '/');
  }
  // pathname.Prepend("local://");

  return new LocalStorageParameters(pathname);
}

Storage* LocalStorageFactory::createStorage(const StorageParameters* param)
{
  // create LocalStorage storage instance from parameters

  if (LocalStorageParameters::Class() == param->IsA()) {

    const LocalStorageParameters* localParam = (const LocalStorageParameters*)param;

    return new LocalStorage(localParam->getPathString());
  }

  return nullptr;
}

void LocalStorage::setRetry(Int_t /* nretry */, Int_t /* initsec */)
{

  // Function to set the exponential retry for putting entries in the OCDB

  LOG(INFO) << "This function sets the exponential retry for putting entries in the OCDB - to be "
               "used ONLY for  GridStorage --> returning without doing anything";
  return;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                             //
//  LocalStorage Parameter class                                                               // //
//                                                                                             //
/////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(LocalStorageParameters);

LocalStorageParameters::LocalStorageParameters() : StorageParameters(), mDBPath()
{
  // default constructor
}

LocalStorageParameters::LocalStorageParameters(const char* dbPath) : StorageParameters(), mDBPath(dbPath)
{
  // constructor

  setType("local");
  setUri(TString("local://") + dbPath);
}

LocalStorageParameters::LocalStorageParameters(const char* dbPath, const char* uri)
  : StorageParameters(), mDBPath(dbPath)
{
  // constructor

  setType("local");
  setUri(TString("alien://") + uri);
}

LocalStorageParameters::~LocalStorageParameters() = default;

StorageParameters* LocalStorageParameters::cloneParam() const
{
  // clone parameter

  return new LocalStorageParameters(mDBPath);
}

ULong_t LocalStorageParameters::getHash() const
{
  // return getHash function

  return mDBPath.Hash();
}

Bool_t LocalStorageParameters::isEqual(const TObject* obj) const
{
  // check if this object is equal to  StorageParameters obj

  if (this == obj) {
    return kTRUE;
  }

  if (LocalStorageParameters::Class() != obj->IsA()) {
    return kFALSE;
  }

  LocalStorageParameters* other = (LocalStorageParameters*)obj;

  return mDBPath == other->mDBPath;
}
