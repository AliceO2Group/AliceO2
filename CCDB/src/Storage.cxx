// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "CCDB/Storage.h"
#include <fairlogger/Logger.h> // for LOG
#include <TH1.h>               // for TH1
#include <TKey.h>              // for TKey
#include <TNtuple.h>           // for TNtuple
#include "CCDB/Condition.h"    // for Condition
#include "CCDB/Manager.h"      // for Manager

using namespace o2::ccdb;

ClassImp(Storage);

Storage::Storage()
  : mValidFileIds(),
    mRun(-1),
    mPathFilter(),
    mVersion(-1),
    mConditionMetaDataFilter(nullptr),
    mSelections(),
    mURI(),
    mType(),
    mBaseFolder(),
    mNretry(0),
    mInitRetrySeconds(0)
{
  // constructor

  mValidFileIds.SetOwner(1);
  mSelections.SetOwner(1);
}

Storage::~Storage()
{
  // destructor

  removeAllSelections();
  mValidFileIds.Clear();
  delete mConditionMetaDataFilter;
}

void Storage::getSelection(/*const*/ ConditionId* id)
{
  // return required version and subversion from the list of selection criteria

  TIter iter(&mSelections);
  ConditionId* aSelection;

  // loop on the list of selection criteria
  while ((aSelection = (ConditionId*)iter.Next())) {
    // check if selection element contains id's path and run (range)
    if (aSelection->isSupersetOf(*id)) {
      LOG(DEBUG) << "Using selection criterion: " << aSelection->ToString().Data() << " ";
      // return required version and subversion

      id->setVersion(aSelection->getVersion());
      id->setSubVersion(aSelection->getSubVersion());
      return;
    }
  }

  // no valid element is found in the list of selection criteria -> return
  LOG(DEBUG) << "Looking for objects with most recent version";
  return;
}

void Storage::readSelectionFromFile(const char* fileName)
{
  // read selection criteria list from file

  removeAllSelections();

  TList* list = getIdListFromFile(fileName);
  if (!list) {
    return;
  }

  list->SetOwner();
  Int_t nId = list->GetEntries();
  ConditionId* id;
  TKey* key;

  for (int i = nId - 1; i >= 0; i--) {
    key = (TKey*)list->At(i);
    id = (ConditionId*)key->ReadObj();
    if (id) {
      addSelection(*id);
    }
  }
  delete list;
  LOG(INFO) << "Selection criteria list filled with " << mSelections.GetEntries() << " entries";
  printSelectionList();
}

void Storage::addSelection(const ConditionId& selection)
{
  // add a selection criterion

  IdPath path = selection.getPath();
  if (!path.isValid()) {
    return;
  }

  TIter iter(&mSelections);
  const ConditionId* anId;
  while ((anId = (ConditionId*)iter.Next())) {
    if (selection.isSupersetOf(*anId)) {
      LOG(WARNING) << "This selection is more general than a previous one and will hide it!";
      LOG(WARNING) << (anId->ToString()).Data();
      mSelections.AddBefore(anId, new ConditionId(selection));
      return;
    }
  }
  mSelections.AddFirst(new ConditionId(selection));
}

void Storage::addSelection(const IdPath& path, const IdRunRange& runRange, Int_t version, Int_t subVersion)
{
  // add a selection criterion

  addSelection(ConditionId(path, runRange, version, subVersion));
}

void Storage::addSelection(const IdPath& path, Int_t firstRun, Int_t lastRun, Int_t version, Int_t subVersion)
{
  // add a selection criterion

  addSelection(ConditionId(path, firstRun, lastRun, version, subVersion));
}

void Storage::removeSelection(const ConditionId& selection)
{
  // remove a selection criterion

  TIter iter(&mSelections);
  ConditionId* aSelection;

  while ((aSelection = (ConditionId*)iter.Next())) {
    if (selection.isSupersetOf(*aSelection)) {
      mSelections.Remove(aSelection);
    }
  }
}

void Storage::removeSelection(const IdPath& path, const IdRunRange& runRange)
{
  // remove a selection criterion

  removeSelection(ConditionId(path, runRange, -1, -1));
}

void Storage::removeSelection(const IdPath& path, Int_t firstRun, Int_t lastRun)
{
  // remove a selection criterion

  removeSelection(ConditionId(path, firstRun, lastRun, -1, -1));
}

void Storage::removeSelection(int position)
{
  // remove a selection criterion from its position in the list

  delete mSelections.RemoveAt(position);
}

void Storage::removeAllSelections()
{
  // remove all selection criteria

  mSelections.Clear();
}

void Storage::printSelectionList()
{
  // prints the list of selection criteria

  TIter iter(&mSelections);
  ConditionId* aSelection;

  // loop on the list of selection criteria
  int index = 0;
  while ((aSelection = (ConditionId*)iter.Next())) {
    LOG(INFO) << "index " << index++ << " -> selection: " << aSelection->ToString().Data();
  }
}

Condition* Storage::getObject(const ConditionId& query)
{
  // get an  Condition object from the database

  // check if query's path and runRange are valid
  // query is invalid also if version is not specified and subversion is!
  if (!query.isValid()) {
    LOG(ERROR) << "Invalid query: " << query.ToString().Data();
    return nullptr;
  }

  // query is not specified if path contains wildcard or runrange = [-1,-1]
  if (!query.isSpecified()) {
    LOG(ERROR) << "Unspecified query: " << query.ToString().Data();
    return nullptr;
  }

  // This is needed otherwise TH1  objects (histos, TTree's) are lost when file is closed!
  Bool_t oldStatus = TH1::AddDirectoryStatus();
  TH1::AddDirectory(kFALSE);

  Condition* entry = getCondition(query);

  if (oldStatus != kFALSE) {
    TH1::AddDirectory(kTRUE);
  }

  // if drain storage is set, drain entry into drain storage
  if (entry && (Manager::Instance())->isdrainSet()) {
    Manager::Instance()->drain(entry);
  }

  return entry;
}

Condition* Storage::getObject(const IdPath& path, Int_t runNumber, Int_t version, Int_t subVersion)
{
  // get an  Condition object from the database

  return getObject(ConditionId(path, runNumber, runNumber, version, subVersion));
}

Condition* Storage::getObject(const IdPath& path, const IdRunRange& runRange, Int_t version, Int_t subVersion)
{
  // get an  Condition object from the database

  return getObject(ConditionId(path, runRange, version, subVersion));
}

TList* Storage::getAllObjects(const ConditionId& query)
{
  // get multiple  Condition objects from the database

  if (!query.isValid()) {
    LOG(ERROR) << "Invalid query: " << query.ToString().Data();
    return nullptr;
  }

  if (query.isAnyRange()) {
    LOG(ERROR) << "Unspecified run or runrange: " << query.ToString().Data();
    return nullptr;
  }

  // This is needed otherwise TH1  objects (histos, TTree's) are lost when file is closed!
  Bool_t oldStatus = TH1::AddDirectoryStatus();
  TH1::AddDirectory(kFALSE);

  TList* result = getAllEntries(query);

  if (oldStatus != kFALSE) {
    TH1::AddDirectory(kTRUE);
  }

  Int_t nEntries = result->GetEntries();

  LOG(INFO) << nEntries << " objects retrieved. Request was: " << query.ToString().Data();
  for (int i = 0; i < nEntries; i++) {
    Condition* entry = (Condition*)result->At(i);
    LOG(INFO) << entry->getId().ToString().Data();
  }

  // if drain storage is set, drain entries into drain storage
  if ((Manager::Instance())->isdrainSet()) {
    for (int i = 0; i < result->GetEntries(); i++) {
      Condition* entry = (Condition*)result->At(i);
      Manager::Instance()->drain(entry);
    }
  }

  return result;
}

TList* Storage::getAllObjects(const IdPath& path, Int_t runNumber, Int_t version, Int_t subVersion)
{
  // get multiple  Condition objects from the database

  return getAllObjects(ConditionId(path, runNumber, runNumber, version, subVersion));
}

TList* Storage::getAllObjects(const IdPath& path, const IdRunRange& runRange, Int_t version, Int_t subVersion)
{
  // get multiple  Condition objects from the database

  return getAllObjects(ConditionId(path, runRange, version, subVersion));
}

ConditionId* Storage::getId(const ConditionId& query)
{
  // get the ConditionId of the valid object from the database (does not open the file)

  // check if query's path and runRange are valid
  // query is invalid also if version is not specified and subversion is!
  if (!query.isValid()) {
    LOG(ERROR) << "Invalid query: " << query.ToString().Data();
    return nullptr;
  }

  // query is not specified if path contains wildcard or runrange = [-1,-1]
  if (!query.isSpecified()) {
    LOG(ERROR) << "Unspecified query: " << query.ToString().Data();
    return nullptr;
  }

  ConditionId* id = getConditionId(query);

  return id;
}

ConditionId* Storage::getId(const IdPath& path, Int_t runNumber, Int_t version, Int_t subVersion)
{
  // get the ConditionId of the valid object from the database (does not open the file)

  return getId(ConditionId(path, runNumber, runNumber, version, subVersion));
}

ConditionId* Storage::getId(const IdPath& path, const IdRunRange& runRange, Int_t version, Int_t subVersion)
{
  // get the ConditionId of the valid object from the database (does not open the file)

  return getId(ConditionId(path, runRange, version, subVersion));
}

Bool_t Storage::putObject(TObject* object, ConditionId& id, ConditionMetaData* metaData, const char* mirrors)
{
  // store an  Condition object into the database

  if (object == nullptr) {
    LOG(ERROR) << "Null Condition! No storage will be done!";
    return kFALSE;
  }

  Condition anCondition(object, id, metaData);

  return putObject(&anCondition, mirrors);
}

Bool_t Storage::putObject(Condition* entry, const char* mirrors)
{
  // store an  Condition object into the database

  if (!entry) {
    LOG(ERROR) << "No entry!";
    return kFALSE;
  }

  if (entry->getObject() == nullptr) {
    LOG(ERROR) << "No valid object in CDB entry!";
    return kFALSE;
  }

  if (!entry->getId().isValid()) {
    LOG(ERROR) << "Invalid entry ID: " << entry->getId().ToString().Data();
    return kFALSE;
  }

  if (!entry->getId().isSpecified()) {
    LOG(ERROR) << "Unspecified entry ID: " << entry->getId().ToString().Data();
    return kFALSE;
  }

  TString strMirrors(mirrors);
  if (!strMirrors.IsNull() && !strMirrors.IsWhitespace()) {
    return putCondition(entry, mirrors);
  } else {
    return putCondition(entry);
  }
}

void Storage::queryStorages(Int_t run, const char* pathFilter, Int_t version, ConditionMetaData* md)
{
  // query CDB for files valid for given run, and fill list mValidFileIds
  // Actual query is done in virtual function queryValidFiles()
  // If version is not specified, the query will fill mValidFileIds
  // with highest versions

  mRun = run;

  mPathFilter = pathFilter;
  if (!mPathFilter.isValid()) {
    LOG(ERROR) << "Filter not valid: " << pathFilter;
    mPathFilter = "*";
    return;
  }

  mVersion = version;

  LOG(INFO) << "Querying files valid for run " << mRun << R"( and path ")" << pathFilter << R"(" into CDB storage ")"
            << mType.Data() << "://" << mBaseFolder.Data() << R"(")";

  // In mValidFileIds, clear id for the same 3level path, if any
  LOG(DEBUG) << R"(Clearing list of CDB ConditionId's previously loaded for path ")" << pathFilter << R"(")";
  IdPath filter(pathFilter);
  for (Int_t i = mValidFileIds.GetEntries() - 1; i >= 0; --i) {
    ConditionId* rmMe = dynamic_cast<ConditionId*>(mValidFileIds.At(i));
    IdPath rmPath = rmMe->getPathString();
    if (filter.isSupersetOf(rmPath)) {
      LOG(DEBUG) << R"(Removing id ")" << rmPath.getPathString().Data() << R"(" matching: ")" << pathFilter << R"(")";
      delete mValidFileIds.RemoveAt(i);
    }
  }

  if (mConditionMetaDataFilter) {
    delete mConditionMetaDataFilter;
    mConditionMetaDataFilter = nullptr;
  }
  if (md) {
    mConditionMetaDataFilter = dynamic_cast<ConditionMetaData*>(md->Clone());
  }

  queryValidFiles();

  LOG(INFO) << mValidFileIds.GetEntries() << " valid files found!";
}

void Storage::printrQueryStorages()
{
  // print parameters used to load list of CDB ConditionId's (mRun, mPathFilter, mVersion)

  ConditionId paramId(mPathFilter, mRun, mRun, mVersion);
  LOG(INFO) << "**** queryStorages Parameters **** \n\t\"" << paramId.ToString().Data() << R"(")";

  if (mConditionMetaDataFilter) {
    mConditionMetaDataFilter->printConditionMetaData();
  }

  TString message = "**** ConditionId's of valid objects found *****\n";
  TIter iter(&mValidFileIds);
  ConditionId* anId = nullptr;

  // loop on the list of selection criteria
  while ((anId = dynamic_cast<ConditionId*>(iter.Next()))) {
    message += Form("\t%s\n", anId->ToString().Data());
  }
  message += Form("\n\tTotal: %d objects found\n", mValidFileIds.GetEntriesFast());
  LOG(INFO) << message.Data();
}

void Storage::setMirrorSEs(const char* mirrors)
{
  // if the current storage is not of "alien" type, just issue a warning
  //  GridStorage implements its own setMirrorSEs method, classes for other storage types do not

  TString storageType = getStorageType();
  if (storageType != "alien") {
    LOG(WARNING) << R"(The current storage is of type ")" << storageType.Data() << R"(". Setting of SEs to ")"
                 << mirrors
                 << R"(" skipped!)";
    return;
  }
  LOG(ERROR) << "We should never get here!!  GridStorage must have masked this virtual method!";
  return;
}

const char* Storage::getMirrorSEs() const
{
  // if the current storage is not of "alien" type, just issue a warning
  //  GridStorage implements its own getMirrorSEs method, classes for other storage types do not

  TString storageType = getStorageType();
  if (storageType != "alien") {
    LOG(WARNING) << R"(The current storage is of type ")" << storageType.Data()
                 << R"(" and cannot handle SEs. Returning empty string!)";
    return "";
  }
  LOG(ERROR) << "We should never get here!!  GridStorage must have masked this virtual method!";
  return "";
}

void Storage::loadTreeFromFile(Condition* entry) const
{
  // Checks whether entry contains a TTree and in case loads it into memory

  TObject* obj = (TObject*)entry->getObject();
  if (!obj) {
    LOG(ERROR) << "Cannot retrieve the object:";
    entry->printConditionMetaData();
    return;
  }

  if (!strcmp(obj->ClassName(), TTree::Class_Name())) {

    LOG(WARNING) << "Condition contains a TTree! Loading baskets...";

    TTree* tree = dynamic_cast<TTree*>(obj);

    if (!tree) {
      return;
    }

    tree->LoadBaskets();
    tree->SetDirectory(nullptr);
  } else if (!strcmp(obj->ClassName(), TNtuple::Class_Name())) {

    LOG(WARNING) << "Condition contains a TNtuple! Loading baskets...";

    TNtuple* ntu = dynamic_cast<TNtuple*>(obj);

    if (!ntu) {
      return;
    }

    ntu->LoadBaskets();
    ntu->SetDirectory(nullptr);
  }

  return;
}
