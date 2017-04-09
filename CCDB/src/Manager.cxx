#include "CCDB/Manager.h"
#include <FairLogger.h>    // for LOG
#include <TGrid.h>         // for gGrid, TGrid
#include <TKey.h>          // for TKey
#include <TMessage.h>      // for TMessage
#include <TObjString.h>    // for TObjString
#include <TRegexp.h>       // for TRegexp
#include <TSAXParser.h>    // for TSAXParser
#include <TUUID.h>         // for TUUID
#include "CCDB/Condition.h"     // for Condition
#include "CCDB/FileStorage.h"   // for FileStorageFactory
#include "CCDB/GridStorage.h"   // for GridStorageFactory
#include "CCDB/LocalStorage.h"  // for LocalStorageFactory
#include "TFile.h"         // for TFile
#include "TSystem.h"       // for TSystem, gSystem
#include "CCDB/XmlHandler.h"    // for XmlHandler

using namespace o2::CDB;

ClassImp(StorageParameters)

ClassImp(Manager)

TString Manager::sOcdbFolderXmlFile("alien:///alice/data/OCDBFoldervsIdRunRange.xml");
Manager *Manager::sInstance = nullptr;

Manager *Manager::Instance(TMap *entryCache, Int_t run)
{
  // returns Manager instance (singleton)

  if (!sInstance) {
    sInstance = new Manager();
    if (!entryCache) {
      sInstance->init();
    } else {
      sInstance->initFromCache(entryCache, run);
    }
  }

  return sInstance;
}

void Manager::init()
{
  // factory registering

  registerFactory(new FileStorageFactory());
  registerFactory(new LocalStorageFactory());
  // GridStorageFactory is registered only if AliEn libraries are enabled in Root
  if (!gSystem->Exec("root-config --has-alien 2>/dev/null |grep yes 2>&1 > /dev/null")) { // returns 0 if yes
    LOG(INFO) << "AliEn classes enabled in Root. GridStorage factory registered." << FairLogger::endl;
    registerFactory(new GridStorageFactory());
  }
}

void Manager::initFromCache(TMap *entryCache, Int_t run)
{
  // initialize manager from existing cache
  // used on the slaves in case of parallel reconstruction
  setRun(run);

  TIter iter(entryCache->GetTable());
  TPair *pair = nullptr;

  while ((pair = dynamic_cast<TPair *>(iter.Next()))) {
    mConditionCache.Add(pair->Key(), pair->Value());
  }
  // mCondition is the new owner of the cache
  mConditionCache.SetOwnerKeyValue(kTRUE, kTRUE);
  entryCache->SetOwnerKeyValue(kFALSE, kFALSE);
  LOG(INFO) << mConditionCache.GetEntries() << " cache entries have been loaded" << FairLogger::endl;
}

void Manager::dumpToSnapshotFile(const char *snapshotFileName, Bool_t singleKeys) const
{
  //
  // If singleKeys is true, dump the entries map and the ids list to the snapshot file
  // (provided mostly for historical reasons, the file is then read with initFromSnapshot),
  // otherwise write to file each Condition separately (the is the preferred way, the file
  // is then read with setSnapshotMode).

  // open the file
  TFile *f = TFile::Open(snapshotFileName, "RECREATE");
  if (!f || f->IsZombie()) {
    LOG(ERROR) << "Cannot open file " << snapshotFileName << FairLogger::endl;
    return;
  }

  LOG(INFO) << "Dumping entriesMap (entries'cache) with " << mConditionCache.GetEntries() << " entries!"
            << FairLogger::endl;
  LOG(INFO) << "Dumping entriesList with " << mIds->GetEntries() << "entries!" << FairLogger::endl;

  f->cd();
  if (singleKeys) {
    f->WriteObject(&mConditionCache, "CDBentriesMap");
    f->WriteObject(mIds, "CDBidsList");
  } else {
    // We write the entries one by one named by their calibration path
    TIter iter(mConditionCache.GetTable());
    TPair *pair = nullptr;
    while ((pair = dynamic_cast<TPair *>(iter.Next()))) {
      TObjString *os = dynamic_cast<TObjString *>(pair->Key());
      if (!os) {
        continue;
      }
      TString path = os->GetString();
      Condition *entry = dynamic_cast<Condition *>(pair->Value());
      if (!entry) {
        continue;
      }
      path.ReplaceAll("/", "*");
      entry->Write(path.Data());
    }
  }
  f->Close();
  delete f;
}

void Manager::dumpToLightSnapshotFile(const char *lightSnapshotFileName) const
{
  // The light snapshot does not contain the CDB objects (Entries) but
  // only the information identifying them, that is the map of storages and
  // the list of Ids, as in the UserInfo of AliESDs.root

  // open the file
  TFile *f = TFile::Open(lightSnapshotFileName, "RECREATE");
  if (!f || f->IsZombie()) {
    LOG(ERROR) << "Cannot open file " << lightSnapshotFileName << FairLogger::endl;
    return;
  }

  LOG(INFO) << "Dumping map of storages with " << mStorageMap->GetEntries() << " entries!" << FairLogger::endl;
  LOG(INFO) << "Dumping entriesList with " << mIds->GetEntries() << " entries!" << FairLogger::endl;
  f->WriteObject(mStorageMap, "cdbStoragesMap");
  f->WriteObject(mIds, "CDBidsList");

  f->Close();
  delete f;
}

Bool_t Manager::initFromSnapshot(const char *snapshotFileName, Bool_t overwrite)
{
  // initialize manager from a CDB snapshot, that is add the entries
  // to the entries map and the ids to the ids list taking them from
  // the map and the list found in the input file

  // if the manager is locked it cannot initialize from a snapshot
  if (mLock) {
    LOG(ERROR) << "Being locked I cannot initialize from the snapshot!" << FairLogger::endl;
    return kFALSE;
  }

  // open the file
  TString snapshotFile(snapshotFileName);
  if (snapshotFile.BeginsWith("alien://")) {
    if (!gGrid) {
      TGrid::Connect("alien://", "");
      if (!gGrid) {
        LOG(ERROR) << "Connection to alien failed!" << FairLogger::endl;
        return kFALSE;
      }
    }
  }

  TFile *f = TFile::Open(snapshotFileName);
  if (!f || f->IsZombie()) {
    LOG(ERROR) << "Cannot open file " << snapshotFileName << FairLogger::endl;
    return kFALSE;
  }

  // retrieve entries' map from snapshot file
  TMap *entriesMap = nullptr;
  TIter next(f->GetListOfKeys());
  TKey *key;
  while ((key = (TKey *) next())) {
    if (strcmp(key->GetClassName(), "TMap") != 0) {
      continue;
    }
    entriesMap = (TMap *) key->ReadObj();
    break;
  }
  if (!entriesMap || entriesMap->GetEntries() == 0) {
    LOG(ERROR) << "Cannot get valid map of CDB entries from snapshot file" << FairLogger::endl;
    return kFALSE;
  }

  // retrieve ids' list from snapshot file
  TList *idsList = nullptr;
  TIter nextKey(f->GetListOfKeys());
  TKey *keyN;
  while ((keyN = (TKey *) nextKey())) {
    if (strcmp(keyN->GetClassName(), "TList") != 0) {
      continue;
    }
    idsList = (TList *) keyN->ReadObj();
    break;
  }
  if (!idsList || idsList->GetEntries() == 0) {
    LOG(ERROR) << "Cannot get valid list of CDB entries from snapshot file" << FairLogger::endl;
    return kFALSE;
  }

  // Add each (entry,id) from the snapshot to the memory: entry to the cache, id to the list of ids.
  // If "overwrite" is false: add the entry to the cache and its id to the list of ids
  // only if neither of them is already there.
  // If "overwrite" is true: write the snapshot entry,id in any case. If something
  // was already there for that calibration type, remove it and issue a warning
  TIter iterObj(entriesMap->GetTable());
  TPair *pair = nullptr;
  Int_t nAdded = 0;
  while ((pair = dynamic_cast<TPair *>(iterObj.Next()))) {
    TObjString *os = (TObjString *) pair->Key();
    TString path = os->GetString();
    TIter iterId(idsList);
    ConditionId *id = nullptr;
    ConditionId *correspondingId = nullptr;
    while ((id = dynamic_cast<ConditionId *>(iterId.Next()))) {
      TString idpath(id->getPathString());
      if (idpath == path) {
        correspondingId = id;
        break;
      }
    }
    if (!correspondingId) {
      LOG(ERROR) << R"(id for ")" << path.Data()
                 << R"(" not found in the snapshot (while entry was). This entry is skipped!)" << FairLogger::endl;
      break;
    }
    Bool_t cached = mConditionCache.Contains(path.Data());
    Bool_t registeredId = kFALSE;
    TIter iter(mIds);
    ConditionId *idT = nullptr;
    while ((idT = dynamic_cast<ConditionId *>(iter.Next()))) {
      if (idT->getPathString() == path) {
        registeredId = kTRUE;
        break;
      }
    }

    if (overwrite) {
      if (cached || registeredId) {
        LOG(WARNING) << R"(An entry was already cached for ")" << path.Data()
                     << R"(". Removing it before caching from snapshot)" << FairLogger::endl;
        unloadFromCache(path.Data());
      }
      mConditionCache.Add(pair->Key(), pair->Value());
      mIds->Add(id);
      nAdded++;
    } else {
      if (cached || registeredId) {
        LOG(WARNING) << R"(An entry was already cached for ")" << path.Data()
                     << R"(". Not adding this object from snapshot)" << FairLogger::endl;
      } else {
        mConditionCache.Add(pair->Key(), pair->Value());
        mIds->Add(id);
        nAdded++;
      }
    }
  }

  // mCondition is the new owner of the cache
  mConditionCache.SetOwnerKeyValue(kTRUE, kTRUE);
  entriesMap->SetOwnerKeyValue(kFALSE, kFALSE);
  mIds->SetOwner(kTRUE);
  idsList->SetOwner(kFALSE);
  LOG(INFO) << nAdded << " new (entry,id) cached. Total number " << mConditionCache.GetEntries() << FairLogger::endl;

  f->Close();
  delete f;

  return kTRUE;
}

void Manager::destroy()
{
  // delete ALCDBManager instance and active storages

  if (sInstance) {
    delete sInstance;
    sInstance = nullptr;
  }
}

Manager::Manager()
  : TObject(),
    mFactories(),
    mActiveStorages(),
    mSpecificStorages(),
    mConditionCache(),
    mIds(nullptr),
    mStorageMap(nullptr),
    mDefaultStorage(nullptr),
    mdrainStorage(nullptr),
    mOfficialStorageParameters(nullptr),
    mReferenceStorageParameters(nullptr),
    mRun(-1),
    mCache(kTRUE),
    mLock(kFALSE),
    mSnapshotMode(kFALSE),
    mSnapshotFile(nullptr),
    mOcdbUploadMode(kFALSE),
    mRaw(kFALSE),
    mCvmfsOcdb(""),
    mStartRunLhcPeriod(-1),
    mEndRunLhcPeriod(-1),
    mLhcPeriod(""),
    mKey(0)
{
  // default constuctor
  mFactories.SetOwner(1);
  mActiveStorages.SetOwner(1);
  mSpecificStorages.SetOwner(1);
  mConditionCache.SetName("CDBConditionCache");
  mConditionCache.SetOwnerKeyValue(kTRUE, kTRUE);

  mStorageMap = new TMap();
  mStorageMap->SetOwner(1);
  mIds = new TList();
  mIds->SetOwner(1);
}

Manager::~Manager()
{
  // destructor
  clearCache();
  destroyActiveStorages();
  mFactories.Delete();
  mdrainStorage = nullptr;
  mDefaultStorage = nullptr;
  delete mStorageMap;
  mStorageMap = nullptr;
  delete mIds;
  mIds = nullptr;
  delete mOfficialStorageParameters;
  delete mReferenceStorageParameters;
  if (mSnapshotMode) {
    mSnapshotFile->Close();
    mSnapshotFile = nullptr;
  }
}

void Manager::putActiveStorage(StorageParameters *param, Storage *storage)
{
  // put a storage object into the list of active storages

  mActiveStorages.Add(param, storage);
  LOG(DEBUG) << "Active storages: " << mActiveStorages.GetEntries() << FairLogger::endl;
}

void Manager::registerFactory(StorageFactory *factory)
{
  // add a storage factory to the list of registerd factories

  if (!mFactories.Contains(factory)) {
    mFactories.Add(factory);
  }
}

Bool_t Manager::hasStorage(const char *dbString) const
{
  // check if dbString is a URI valid for one of the registered factories

  TIter iter(&mFactories);

  StorageFactory *factory = nullptr;
  while ((factory = (StorageFactory *) iter.Next())) {

    if (factory->validateStorageUri(dbString)) {
      return kTRUE;
    }
  }

  return kFALSE;
}

StorageParameters *Manager::createStorageParameter(const char *dbString) const
{
  // create  StorageParameters object from URI string

  TString uriString(dbString);

  if (!mCvmfsOcdb.IsNull() && uriString.BeginsWith("alien://")) {
    alienToCvmfsUri(uriString);
  }

  TIter iter(&mFactories);

  StorageFactory *factory = nullptr;
  while ((factory = (StorageFactory *) iter.Next())) {
    StorageParameters *param = factory->createStorageParameter(uriString);
    if (param) {
      return param;
    }
  }

  return nullptr;
}

void Manager::alienToCvmfsUri(TString &uriString) const
{
  // convert alien storage uri to local:///cvmfs storage uri (called when OCDB_PATH is set)

  TObjArray *arr = uriString.Tokenize('?');
  TIter iter(arr);
  TObjString *str = nullptr;
  TString entryKey = "";
  TString entryValue = "";
  TString newUriString = "";
  while ((str = (TObjString *) iter.Next())) {
    TString entry(str->String());
    Int_t indeq = entry.Index('=');
    entryKey = entry(0, indeq + 1);
    entryValue = entry(indeq + 1, entry.Length() - indeq);

    if (entryKey.Contains("folder", TString::kIgnoreCase)) {

      TRegexp re_RawFolder("^/alice/data/20[0-9]+/OCDB");
      TRegexp re_MCFolder("^/alice/simulation/2008/v4-15-Release");
      TString rawFolder = entryValue(re_RawFolder);
      TString mcFolder = entryValue(re_MCFolder);
      if (!rawFolder.IsNull()) {
        entryValue.Replace(0, 6, "/cvmfs/alice-ocdb.cern.ch/calibration");
        // entryValue.Replace(entryValue.Length()-4, entryValue.Length(), "");
      } else if (!mcFolder.IsNull()) {
        entryValue.Replace(0, 36, "/cvmfs/alice-ocdb.cern.ch/calibration/MC");
      } else {
        LOG(FATAL) << "Environment variable for cvmfs OCDB folder set for an invalid OCDB storage:\n   "
                   << entryValue.Data() << FairLogger::endl;
      }
    } else {
      newUriString += entryKey;
    }
    newUriString += entryValue;
    newUriString += '?';
  }
  newUriString.Prepend("local://");
  newUriString.Remove(TString::kTrailing, '?');
  uriString = newUriString;
}

Storage *Manager::getStorage(const char *dbString)
{
  // Get the CDB storage corresponding to the URI string passed as argument
  // If "raw://" is passed, get the storage for the raw OCDB for the current run (mRun)

  TString uriString(dbString);
  if (uriString.EqualTo("raw://")) {
    if (!mLhcPeriod.IsNull() && !mLhcPeriod.IsWhitespace()) {
      return getDefaultStorage();
    } else {
      TString lhcPeriod("");
      Int_t startRun = -1, endRun = -1;
      getLHCPeriodAgainstAlienFile(mRun, lhcPeriod, startRun, endRun);
      return getStorage(lhcPeriod.Data());
    }
  }

  StorageParameters *param = createStorageParameter(dbString);
  if (!param) {
    LOG(ERROR) << "Failed to activate requested storage! Check URI: " << dbString << FairLogger::endl;
    return nullptr;
  }

  Storage *aStorage = getStorage(param);

  delete param;
  return aStorage;
}

Storage *Manager::getStorage(const StorageParameters *param)
{
  // get storage object from  StorageParameters object

  // if the list of active storages already contains
  // the requested storage, return it
  Storage *aStorage = getActiveStorage(param);
  if (aStorage) {
    return aStorage;
  }

  // if lock is ON, cannot activate more storages!
  if (mLock) {
    if (mDefaultStorage) {
      LOG(FATAL) << "Lock is ON, and default storage is already set: cannot reset it or activate "
        "more storages!" << FairLogger::endl;
    }
  }

  // loop on the list of registered factories
  TIter iter(&mFactories);
  StorageFactory *factory = nullptr;
  while ((factory = (StorageFactory *) iter.Next())) {

    // each factory tries to create its storage from the parameter
    aStorage = factory->createStorage(param);
    if (aStorage) {
      putActiveStorage(param->cloneParam(), aStorage);
      aStorage->setUri(param->getUri());
      if (mRun >= 0) {
        if (aStorage->getStorageType() == "alien" || aStorage->getStorageType() == "local") {
          aStorage->queryStorages(mRun);
        }
      }
      return aStorage;
    }
  }

  LOG(ERROR) << "Failed to activate requested storage! Check URI: " << param->getUri().Data() << FairLogger::endl;

  return nullptr;
}

Storage *Manager::getActiveStorage(const StorageParameters *param)
{
  // get a storage object from the list of active storages

  return dynamic_cast<Storage *>(mActiveStorages.GetValue(param));
}

TList *Manager::getActiveStorages()
{
  // return list of active storages
  // user has responsibility to delete returned object

  TList *result = new TList();

  TIter iter(mActiveStorages.GetTable());
  TPair *aPair = nullptr;
  while ((aPair = (TPair *) iter.Next())) {
    result->Add(aPair->Value());
  }

  return result;
}

void Manager::setdrainMode(const char *dbString)
{
  // set drain storage from URI string

  mdrainStorage = getStorage(dbString);
}

void Manager::setdrainMode(const StorageParameters *param)
{
  // set drain storage from  StorageParameters

  mdrainStorage = getStorage(param);
}

void Manager::setdrainMode(Storage *storage)
{
  // set drain storage from another active storage

  mdrainStorage = storage;
}

Bool_t Manager::drain(Condition *entry)
{
  // drain retrieved object to drain storage

  LOG(DEBUG) << "draining into drain storage..." << FairLogger::endl;
  return mdrainStorage->putObject(entry);
}

Bool_t Manager::setOcdbUploadMode()
{
  // Set the framework in official upload mode. This tells the framework to upload
  // objects to cvmfs after they have been uploaded to AliEn OCDBs.
  // It return false if the executable to upload to cvmfs is not found.

  TString cvmfsUploadExecutable("$HOME/bin/ocdb-cvmfs");
  gSystem->ExpandPathName(cvmfsUploadExecutable);
  if (gSystem->AccessPathName(cvmfsUploadExecutable)) {
    return kFALSE;
  }
  mOcdbUploadMode = kTRUE;
  return kTRUE;
}

void Manager::setDefaultStorage(const char *storageUri)
{
  // sets default storage from URI string

  // if in the cvmfs case (triggered by environment variable) check for path validity
  // and modify Uri if it is "raw://"
  TString cvmfsOcdb(gSystem->Getenv("OCDB_PATH"));
  if (!cvmfsOcdb.IsNull()) {
    mCvmfsOcdb = cvmfsOcdb;
    validateCvmfsCase();
  }

  // checking whether we are in the raw case
  TString uriTemp(storageUri);
  if (uriTemp == "raw://") {
    mRaw = kTRUE; // read then by setRun to check if the method has to be called again with expanded uri
    LOG(INFO) << "Setting the run-number will set the corresponding OCDB for raw data reconstruction."
              << FairLogger::endl;
    return;
  }

  Storage *bckStorage = mDefaultStorage;

  mDefaultStorage = getStorage(storageUri);

  if (!mDefaultStorage) {
    return;
  }

  if (bckStorage && (mDefaultStorage != bckStorage)) {
    LOG(WARNING) << "Existing default storage replaced: clearing cache!" << FairLogger::endl;
    clearCache();
  }

  if (mStorageMap->Contains("default")) {
    delete mStorageMap->Remove(((TPair *) mStorageMap->FindObject("default"))->Key());
  }
  mStorageMap->Add(new TObjString("default"), new TObjString(mDefaultStorage->getUri()));
}

void Manager::setDefaultStorage(const StorageParameters *param)
{
  // set default storage from  StorageParameters object

  Storage *bckStorage = mDefaultStorage;

  mDefaultStorage = getStorage(param);

  if (!mDefaultStorage) {
    return;
  }

  if (bckStorage && (mDefaultStorage != bckStorage)) {
    LOG(WARNING) << "Existing default storage replaced: clearing cache!" << FairLogger::endl;
    clearCache();
  }

  if (mStorageMap->Contains("default")) {
    delete mStorageMap->Remove(((TPair *) mStorageMap->FindObject("default"))->Key());
  }
  mStorageMap->Add(new TObjString("default"), new TObjString(mDefaultStorage->getUri()));
}

void Manager::setDefaultStorage(Storage *storage)
{
  // set default storage from another active storage

  // if lock is ON, cannot activate more storages!
  if (mLock) {
    if (mDefaultStorage) {
      LOG(FATAL) << "Lock is ON, and default storage is already set: cannot reset it or activate "
        "more storages!" << FairLogger::endl;
    }
  }

  if (!storage) {
    unsetDefaultStorage();
    return;
  }

  Storage *bckStorage = mDefaultStorage;

  mDefaultStorage = storage;

  if (bckStorage && (mDefaultStorage != bckStorage)) {
    LOG(WARNING) << "Existing default storage replaced: clearing cache!" << FairLogger::endl;
    clearCache();
  }

  if (mStorageMap->Contains("default")) {
    delete mStorageMap->Remove(((TPair *) mStorageMap->FindObject("default"))->Key());
  }
  mStorageMap->Add(new TObjString("default"), new TObjString(mDefaultStorage->getUri()));
}

void Manager::validateCvmfsCase() const
{
  // The OCDB_PATH variable contains the path to the directory in /cvmfs/ which is
  // an AliRoot tag based snapshot of the AliEn file catalogue (e.g.
  // /cvmfs/alice.cern.ch/x86_64-2.6-gnu-4.1.2/Packages/OCDB/v5-05-76-AN).
  // The directory has to contain:
  // 1) <data|MC>/20??.list.gz gzipped text files listing the OCDB files (seen by that AliRoot tag)
  // 2) bin/getOCDBFilesPerRun.sh   (shell+awk) script extracting from 1) the list
  //    of valid files for the given run.

  if (!mCvmfsOcdb.BeginsWith("/cvmfs")) //!!!! to be commented out for testing
    LOG(FATAL) << "OCDB_PATH set to an invalid path: " << mCvmfsOcdb.Data() << FairLogger::endl;

  TString cvmfsUri(mCvmfsOcdb);
  gSystem->ExpandPathName(cvmfsUri);
  if (gSystem->AccessPathName(cvmfsUri))
    LOG(FATAL) << "OCDB_PATH set to an invalid path: " << cvmfsUri.Data() << FairLogger::endl;

  // check that we find the two scripts we need

  LOG(DEBUG) << "OCDB_PATH envvar is set. Changing OCDB storage from alien:// to local:///cvmfs type."
             << FairLogger::endl;
  cvmfsUri = cvmfsUri.Strip(TString::kTrailing, '/');
  cvmfsUri.Append("/bin/getOCDBFilesPerRun.sh");
  if (gSystem->AccessPathName(cvmfsUri))
    LOG(FATAL) << "Cannot find valid script: " << cvmfsUri.Data() << FairLogger::endl;
}

void Manager::setDefaultStorageFromRun(Int_t run)
{
  // set default storage from the run number - to be used only with raw data

  // if lock is ON, cannot activate more storages!
  if (mLock) {
    if (mDefaultStorage) {
      LOG(FATAL) << "Lock is ON, and default storage is already set: cannot activate default "
        "storage from run number" << FairLogger::endl;
    }
  }

  TString lhcPeriod("");
  Int_t startRun = 0, endRun = 0;
  if (!mCvmfsOcdb.IsNull()) { // mRaw and cvmfs case: set LHC period from cvmfs file
    getLHCPeriodAgainstCvmfsFile(run, lhcPeriod, startRun, endRun);
  } else { // mRaw: set LHC period from AliEn XML file
    getLHCPeriodAgainstAlienFile(run, lhcPeriod, startRun, endRun);
  }

  mLhcPeriod = lhcPeriod;
  mStartRunLhcPeriod = startRun;
  mEndRunLhcPeriod = endRun;

  setDefaultStorage(mLhcPeriod.Data());
  if (!mDefaultStorage)
    LOG(FATAL) << mLhcPeriod.Data() << " storage not there! Please check!" << FairLogger::endl;
}

void Manager::getLHCPeriodAgainstAlienFile(Int_t run, TString &lhcPeriod, Int_t &startRun, Int_t &endRun)
{
  // set LHC period (year + first, last run) comparing run number and AliEn XML file

  // retrieve XML file from alien
  if (!gGrid) {
    TGrid::Connect("alien://", "");
    if (!gGrid) {
      LOG(ERROR) << "Connection to alien failed!" << FairLogger::endl;
      return;
    }
  }
  TUUID uuid;
  TString rndname = "/tmp/";
  rndname += "OCDBFolderXML.";
  rndname += uuid.AsString();
  rndname += ".xml";
  LOG(DEBUG) << "file to be copied = " << sOcdbFolderXmlFile.Data() << FairLogger::endl;
  if (!TFile::Cp(sOcdbFolderXmlFile.Data(), rndname.Data())) {
    LOG(FATAL) << "Cannot make a local copy of OCDBFolder xml file in " << rndname.Data() << FairLogger::endl;
  }
  XmlHandler *saxcdb = new XmlHandler();
  saxcdb->setRun(run);
  TSAXParser *saxParser = new TSAXParser();
  saxParser->ConnectToHandler(" Handler", saxcdb);
  saxParser->ParseFile(rndname.Data());
  LOG(INFO) << " LHC folder = " << saxcdb->getOcdbFolder().Data() << FairLogger::endl;
  LOG(INFO) << " LHC period start run = " << saxcdb->getStartIdRunRange() << FairLogger::endl;
  LOG(INFO) << " LHC period end run = " << saxcdb->getEndIdRunRange() << FairLogger::endl;
  lhcPeriod = saxcdb->getOcdbFolder();
  startRun = saxcdb->getStartIdRunRange();
  endRun = saxcdb->getEndIdRunRange();
}

void Manager::getLHCPeriodAgainstCvmfsFile(Int_t run, TString &lhcPeriod, Int_t &startRun, Int_t &endRun)
{
  // set LHC period (year + first, last run) comparing run number and CVMFS file
  // We don't want to connect to AliEn just to set the uri from the runnumber
  // for that we use the script getUriFromYear.sh in the cvmfs AliRoot package

  TString getYearScript(mCvmfsOcdb);
  getYearScript = getYearScript.Strip(TString::kTrailing, '/');
  getYearScript.Append("/bin/getUriFromYear.sh");
  if (gSystem->AccessPathName(getYearScript))
    LOG(FATAL) << "Cannot find valid script: " << getYearScript.Data() << FairLogger::endl;
  TString inoutFile(gSystem->WorkingDirectory());
  inoutFile += "/uri_range_";
  inoutFile += TString::Itoa(run, 10);
  TString command(getYearScript);
  command += ' ';
  command += TString::Itoa(run, 10);
  command += Form(" > %s", inoutFile.Data());
  LOG(DEBUG) << R"(Running command: ")" << command.Data() << R"(")" << FairLogger::endl;
  Int_t result = gSystem->Exec(command.Data());
  if (result != 0) {
    LOG(FATAL) << R"(Was not able to execute ")" << command.Data() << R"(")" << FairLogger::endl;
  }

  // now read the file with the uri and first and last run
  std::ifstream file(inoutFile.Data());
  if (!file.is_open()) {
    LOG(FATAL) << R"(Error opening file ")" << inoutFile.Data() << R"("!)" << FairLogger::endl;
  }
  TString line;
  TObjArray *oStringsArray = nullptr;
  while (line.ReadLine(file)) {
    oStringsArray = line.Tokenize(' ');
  }
  TObjString *oStrUri = dynamic_cast<TObjString *>(oStringsArray->At(0));
  TObjString *oStrFirst = dynamic_cast<TObjString *>(oStringsArray->At(1));
  TString firstRun = oStrFirst->GetString();
  TObjString *oStrLast = dynamic_cast<TObjString *>(oStringsArray->At(2));
  TString lastRun = oStrLast->GetString();

  lhcPeriod = oStrUri->GetString();
  startRun = firstRun.Atoi();
  endRun = lastRun.Atoi();

  file.close();
}

void Manager::unsetDefaultStorage()
{
  // Unset default storage

  // if lock is ON, action is forbidden!
  if (mLock) {
    if (mDefaultStorage) {
      LOG(FATAL) << "Lock is ON: cannot unset default storage!" << FairLogger::endl;
    }
  }

  if (mDefaultStorage) {
    LOG(WARNING) << "Clearing cache!" << FairLogger::endl;
    clearCache();
  }

  mRun = mStartRunLhcPeriod = mEndRunLhcPeriod = -1;
  mRaw = kFALSE;

  mDefaultStorage = nullptr;
}

void Manager::setSpecificStorage(const char *calibType, const char *dbString, Int_t version, Int_t subVersion)
{
  // sets storage specific for detector or calibration type (works with  Manager::getObject(...))

  StorageParameters *aPar = createStorageParameter(dbString);
  if (!aPar) {
    return;
  }
  setSpecificStorage(calibType, aPar, version, subVersion);
  delete aPar;
}

void Manager::setSpecificStorage(const char *calibType, const StorageParameters *param, Int_t version, Int_t subVersion)
{
  // sets storage specific for detector or calibration type (works with  Manager::getObject(...))
  // Default storage should be defined prior to any specific storages, e.g.:
  //  Manager::instance()->setDefaultStorage("alien://");
  //  Manager::instance()->setSpecificStorage("TPC/*","local://DB_TPC");
  //  Manager::instance()->setSpecificStorage("*/Align/*","local://DB_TPCAlign");
  // calibType must be a valid CDB path! (3 level folder structure)
  // Specific version/subversion is set in the uniqueid of the Param value stored in the
  // specific storages map

  if (!mDefaultStorage && !mRaw) {
    LOG(ERROR) << "Please activate a default storage first!" << FairLogger::endl;
    return;
  }

  IdPath aPath(calibType);
  if (!aPath.isValid()) {
    LOG(ERROR) << "Not a valid path: " << calibType << FairLogger::endl;
    return;
  }

  TObjString *objCalibType = new TObjString(aPath.getPathString());
  if (mSpecificStorages.Contains(objCalibType)) {
    LOG(WARNING) << R"(Storage ")" << calibType << R"(" already activated! It will be replaced by the new one)"
                 << FairLogger::endl;
    StorageParameters *checkPar = dynamic_cast<StorageParameters *>(mSpecificStorages.GetValue(calibType));
    if (checkPar) {
      delete checkPar;
    }
    delete mSpecificStorages.Remove(objCalibType);
  }
  Storage *aStorage = getStorage(param);
  if (!aStorage) {
    return;
  }

  // Set the unique id of the AliCDBParam stored in the map to store specific version/subversion
  UInt_t uId = ((subVersion + 1) << 16) + (version + 1);
  StorageParameters *specificParam = param->cloneParam();
  specificParam->SetUniqueID(uId);
  mSpecificStorages.Add(objCalibType, specificParam);

  if (mStorageMap->Contains(objCalibType)) {
    delete mStorageMap->Remove(objCalibType);
  }
  mStorageMap->Add(objCalibType->Clone(), new TObjString(param->getUri()));
}

Storage *Manager::getSpecificStorage(const char *calibType)
{
  // get storage specific for detector or calibration type

  IdPath calibPath(calibType);
  if (!calibPath.isValid()) {
    return nullptr;
  }

  StorageParameters *checkPar = (StorageParameters *) mSpecificStorages.GetValue(calibPath.getPathString());
  if (!checkPar) {
    LOG(ERROR) << calibType << " storage not found!" << FairLogger::endl;
    return nullptr;
  } else {
    return getStorage(checkPar);
  }
}

StorageParameters *Manager::selectSpecificStorage(const TString &path)
{
  // select storage valid for path from the list of specific storages

  IdPath aPath(path);
  if (!aPath.isValid()) {
    return nullptr;
  }

  TIter iter(&mSpecificStorages);
  TObjString *aCalibType = nullptr;
  IdPath tmpPath("null/null/null");
  StorageParameters *aPar = nullptr;
  while ((aCalibType = (TObjString *) iter.Next())) {
    IdPath calibTypePath(aCalibType->GetName());
    if (calibTypePath.isSupersetOf(aPath)) {
      if (calibTypePath.isSupersetOf(tmpPath)) {
        continue;
      }
      aPar = (StorageParameters *) mSpecificStorages.GetValue(aCalibType);
      tmpPath.setPath(calibTypePath.getPathString());
    }
  }
  return aPar;
}

Condition *Manager::getObject(const IdPath &path, Int_t runNumber, Int_t version, Int_t subVersion)
{
  // get an  Condition object from the database

  if (runNumber < 0) {
    // RunNumber is not specified. Try with mRun
    if (mRun < 0) {
      LOG(ERROR) << "Run number neither specified in query nor set in  Manager! Use  Manager::setRun."
                 << FairLogger::endl;
      return nullptr;
    }
    runNumber = mRun;
  }

  return getObject(ConditionId(path, runNumber, runNumber, version, subVersion));
}

Condition *Manager::getObject(const IdPath &path, const IdRunRange &runRange, Int_t version, Int_t subVersion)
{
  // get an  Condition object from the database!

  return getObject(ConditionId(path, runRange, version, subVersion));
}

Condition *Manager::getObject(const ConditionId &queryId, Bool_t forceCaching)
{
  // get an  Condition object from the database

  // check if queryId's path and runRange are valid
  // queryId is invalid also if version is not specified and subversion is!
  if (!queryId.isValid()) {
    LOG(ERROR) << "Invalid query: " << queryId.ToString().Data() << FairLogger::endl;
    return nullptr;
  }

  // query is not specified if path contains wildcard or run range= [-1,-1]
  if (!queryId.isSpecified()) {
    LOG(ERROR) << "Unspecified query: " << queryId.ToString().Data() << FairLogger::endl;
    return nullptr;
  }

  if (mLock && !(mRun >= queryId.getFirstRun() && mRun <= queryId.getLastRun()))
    LOG(FATAL) << "Lock is ON: cannot use different run number than the internal one!" << FairLogger::endl;

  if (mCache && !(mRun >= queryId.getFirstRun() && mRun <= queryId.getLastRun()))
    LOG(WARNING) << "Run number explicitly set in query: CDB cache temporarily disabled!" << FairLogger::endl;

  Condition *entry = nullptr;

  // first look into map of cached objects
  if (mCache && queryId.getFirstRun() == mRun) {
    entry = (Condition *) mConditionCache.GetValue(queryId.getPathString());
  }
  if (entry) {
    LOG(DEBUG) << "Object " << queryId.getPathString().Data() << " retrieved from cache !!" << FairLogger::endl;
    return entry;
  }

  // if snapshot flag is set, try getting from the snapshot
  // but in the case a specific storage is specified for this path
  StorageParameters *aPar = selectSpecificStorage(queryId.getPathString());
  if (!aPar) {
    if (mSnapshotMode && queryId.getFirstRun() == mRun) {
      entry = getConditionFromSnapshot(queryId.getPathString());
      if (entry) {
        LOG(INFO) << R"(Object ")" << queryId.getPathString().Data() << R"(" retrieved from the snapshot.)"
                  << FairLogger::endl;
        if (queryId.getFirstRun() == mRun) { // no need to check mCache, mSnapshotMode not possible otherwise
          cacheCondition(queryId.getPathString(), entry);
        }

        if (!mIds->Contains(&entry->getId())) {
          mIds->Add(entry->getId().Clone());
        }

        return entry;
      }
    }
  }

  // Condition is not in cache (and, in case we are in snapshot mode, not in the snapshot either)
  // => retrieve it from the storage and cache it!!
  if (!mDefaultStorage) {
    LOG(ERROR) << "No storage set!" << FairLogger::endl;
    return nullptr;
  }

  Int_t version = -1, subVersion = -1;
  Storage *aStorage = nullptr;
  if (aPar) {
    aStorage = getStorage(aPar);
    TString str = aPar->getUri();
    UInt_t uId = aPar->GetUniqueID();
    version = Int_t(uId & 0xffff) - 1;
    subVersion = Int_t(uId >> 16) - 1;
    LOG(DEBUG) << "Looking into storage: " << str.Data() << FairLogger::endl;
  } else {
    aStorage = getDefaultStorage();
    LOG(DEBUG) << "Looking into default storage" << FairLogger::endl;
  }

  ConditionId finalQueryId(queryId);
  if (version >= 0) {
    LOG(DEBUG) << "Specific version set to: " << version << FairLogger::endl;
    finalQueryId.setVersion(version);
  }
  if (subVersion >= 0) {
    LOG(DEBUG) << "Specific subversion set to: " << subVersion << FairLogger::endl;
    finalQueryId.setSubVersion(subVersion);
  }
  entry = aStorage->getObject(finalQueryId);

  if (entry && mCache && (queryId.getFirstRun() == mRun || forceCaching)) {
    cacheCondition(queryId.getPathString(), entry);
  }

  if (entry && !mIds->Contains(&entry->getId())) {
    mIds->Add(entry->getId().Clone());
  }

  return entry;
}

Condition *Manager::getConditionFromSnapshot(const char *path)
{
  // get the entry from the open snapshot file

  TString sPath(path);
  sPath.ReplaceAll("/", "*");
  if (!mSnapshotFile) {
    LOG(ERROR) << "No snapshot file is open!" << FairLogger::endl;
    return nullptr;
  }
  Condition *entry = dynamic_cast<Condition *>(mSnapshotFile->Get(sPath.Data()));
  if (!entry) {
    LOG(DEBUG) << R"(Cannot get a CDB entry for ")" << path << R"(" from snapshot file)" << FairLogger::endl;
    return nullptr;
  }

  return entry;
}

Bool_t Manager::setSnapshotMode(const char *snapshotFileName)
{
  // set the manager in snapshot mode

  if (!mCache) {
    LOG(ERROR) << "Cannot set the CDB manage in snapshot mode if the cache is not active!" << FairLogger::endl;
    return kFALSE;
  }

  // open snapshot file
  TString snapshotFile(snapshotFileName);
  if (snapshotFile.BeginsWith("alien://")) {
    if (!gGrid) {
      TGrid::Connect("alien://", "");
      if (!gGrid) {
        LOG(ERROR) << "Connection to alien failed!" << FairLogger::endl;
        return kFALSE;
      }
    }
  }

  mSnapshotFile = TFile::Open(snapshotFileName);
  if (!mSnapshotFile || mSnapshotFile->IsZombie()) {
    LOG(ERROR) << "Cannot open file " << snapshotFileName << FairLogger::endl;
    return kFALSE;
  }

  LOG(INFO) << "The CDB manager is set in snapshot mode!" << FairLogger::endl;
  mSnapshotMode = kTRUE;
  return kTRUE;
}

const char *Manager::getUri(const char *path)
{
  // return the URI of the storage where to look for path

  if (!isDefaultStorageSet()) {
    return nullptr;
  }

  StorageParameters *aPar = selectSpecificStorage(path);

  if (aPar) {
    return aPar->getUri().Data();

  } else {
    return getDefaultStorage()->getUri().Data();
  }

  return nullptr;
}

Int_t Manager::getStartRunLHCPeriod()
{
  // get the first run of validity
  // for the current period
  // if set
  if (mStartRunLhcPeriod == -1)
    LOG(WARNING) << "Run-range not yet set for the current LHC period." << FairLogger::endl;
  return mStartRunLhcPeriod;
}

Int_t Manager::getEndRunLHCPeriod()
{
  // get the last run of validity
  // for the current period
  // if set
  if (mEndRunLhcPeriod == -1)
    LOG(WARNING) << "Run-range not yet set for the current LHC period." << FairLogger::endl;
  return mEndRunLhcPeriod;
}

TString Manager::getLHCPeriod()
{
  // get the current LHC period string
  //
  if (mLhcPeriod.IsWhitespace() || mLhcPeriod.IsNull())
    LOG(WARNING) << "LHC period (OCDB folder) not yet set" << FairLogger::endl;
  return mLhcPeriod;
}

ConditionId *Manager::getId(const IdPath &path, Int_t runNumber, Int_t version, Int_t subVersion)
{
  // get the ConditionId of the valid object from the database (does not retrieve the object)
  // User must delete returned object!

  if (runNumber < 0) {
    // RunNumber is not specified. Try with mRun
    if (mRun < 0) {
      LOG(ERROR) << "Run number neither specified in query nor set in  Manager! Use  Manager::setRun."
                 << FairLogger::endl;
      return nullptr;
    }
    runNumber = mRun;
  }

  return getId(ConditionId(path, runNumber, runNumber, version, subVersion));
}

ConditionId *Manager::getId(const IdPath &path, const IdRunRange &runRange, Int_t version, Int_t subVersion)
{
  // get the ConditionId of the valid object from the database (does not retrieve the object)
  // User must delete returned object!

  return getId(ConditionId(path, runRange, version, subVersion));
}

ConditionId *Manager::getId(const ConditionId &query)
{
  // get the ConditionId of the valid object from the database (does not retrieve the object)
  // User must delete returned object!

  if (!mDefaultStorage) {
    LOG(ERROR) << "No storage set!" << FairLogger::endl;
    return nullptr;
  }

  // check if query's path and runRange are valid
  // query is invalid also if version is not specified and subversion is!
  if (!query.isValid()) {
    LOG(ERROR) << "Invalid query: " << query.ToString().Data() << FairLogger::endl;
    return nullptr;
  }

  // query is not specified if path contains wildcard or run range= [-1,-1]
  if (!query.isSpecified()) {
    LOG(ERROR) << "Unspecified query: " << query.ToString().Data() << FairLogger::endl;
    return nullptr;
  }

  if (mCache && query.getFirstRun() != mRun)
    LOG(WARNING) << "Run number explicitly set in query: CDB cache temporarily disabled!" << FairLogger::endl;

  Condition *entry = nullptr;

  // first look into map of cached objects
  if (mCache && query.getFirstRun() == mRun) {
    entry = (Condition *) mConditionCache.GetValue(query.getPathString());
  }

  if (entry) {
    LOG(DEBUG) << "Object " << query.getPathString().Data() << " retrieved from cache !!" << FairLogger::endl;
    return dynamic_cast<ConditionId *>(entry->getId().Clone());
  }

  // Condition is not in cache -> retrieve it from CDB and cache it!!
  Storage *aStorage = nullptr;
  StorageParameters *aPar = selectSpecificStorage(query.getPathString());

  if (aPar) {
    aStorage = getStorage(aPar);
    TString str = aPar->getUri();
    LOG(DEBUG) << "Looking into storage: " << str.Data() << FairLogger::endl;

  } else {
    aStorage = getDefaultStorage();
    LOG(DEBUG) << "Looking into default storage" << FairLogger::endl;
  }

  return aStorage->getId(query);
}

TList *Manager::getAllObjects(const IdPath &path, Int_t runNumber, Int_t version, Int_t subVersion)
{
  // get multiple  Condition objects from the database

  if (runNumber < 0) {
    // RunNumber is not specified. Try with mRun
    if (mRun < 0) {
      LOG(ERROR) << "Run number neither specified in query nor set in  Manager! Use  Manager::setRun."
                 << FairLogger::endl;
      return nullptr;
    }
    runNumber = mRun;
  }

  return getAllObjects(ConditionId(path, runNumber, runNumber, version, subVersion));
}

TList *Manager::getAllObjects(const IdPath &path, const IdRunRange &runRange, Int_t version, Int_t subVersion)
{
  // get multiple  Condition objects from the database

  return getAllObjects(ConditionId(path, runRange, version, subVersion));
}

TList *Manager::getAllObjects(const ConditionId &query)
{
  // get multiple  Condition objects from the database
  // Warning: this method works correctly only for queries of the type "Detector/*"
  // 		and not for more specific queries e.g. "Detector/Calib/*" !
  // Warning #2: Entries are cached, but getAllObjects will keep on retrieving objects from OCDB!
  // 		To get an object from cache use getObject() function

  if (!mDefaultStorage) {
    LOG(ERROR) << "No storage set!" << FairLogger::endl;
    return nullptr;
  }

  if (!query.isValid()) {
    LOG(ERROR) << "Invalid query: " << query.ToString().Data() << FairLogger::endl;
    return nullptr;
  }

  if ((mSpecificStorages.GetEntries() > 0) && query.getPathString().BeginsWith('*')) {
    // if specific storages are active a query with "*" is ambiguous
    LOG(ERROR) << "Query too generic in this context!" << FairLogger::endl;
    return nullptr;
  }

  if (query.isAnyRange()) {
    LOG(ERROR) << "Unspecified run or runrange: " << query.ToString().Data() << FairLogger::endl;
    return nullptr;
  }

  if (mLock && query.getFirstRun() != mRun)
    LOG(FATAL) << "Lock is ON: cannot use different run number than the internal one!" << FairLogger::endl;

  StorageParameters *aPar = selectSpecificStorage(query.getPathString());

  Storage *aStorage;
  if (aPar) {
    aStorage = getStorage(aPar);
    LOG(DEBUG) << "Looking into storage: " << aPar->getUri().Data() << FairLogger::endl;

  } else {
    aStorage = getDefaultStorage();
    LOG(DEBUG) << "Looking into default storage: " << aStorage->getUri().Data() << FairLogger::endl;
  }

  TList *result = nullptr;
  if (aStorage) {
    result = aStorage->getAllObjects(query);
  }
  if (!result) {
    return nullptr;
  }

  // loop on result to check whether entries should be re-queried with specific storages
  if (mSpecificStorages.GetEntries() > 0 && !(mSpecificStorages.GetEntries() == 1 && aPar)) {
    LOG(INFO) << "Now look into all other specific storages..." << FairLogger::endl;

    TIter iter(result);
    Condition *chkCondition = nullptr;

    while ((chkCondition = dynamic_cast<Condition *>(iter.Next()))) {
      ConditionId &chkId = chkCondition->getId();
      LOG(DEBUG) << "Checking id " << chkId.getPathString().Data() << " " << FairLogger::endl;
      StorageParameters *chkPar = selectSpecificStorage(chkId.getPathString());
      if (!chkPar || aPar == chkPar) {
        continue;
      }
      Storage *chkStorage = getStorage(chkPar);
      LOG(DEBUG) << "Found specific storage! " << chkPar->getUri().Data() << FairLogger::endl;

      chkId.setIdRunRange(query.getFirstRun(), query.getLastRun());
      UInt_t uId = chkPar->GetUniqueID();
      Int_t version = -1, subVersion = -1;
      version = Int_t(uId & 0xffff) - 1;
      subVersion = Int_t(uId >> 16) - 1;
      if (version != -1) {
        chkId.setVersion(version);
      } else {
        chkId.setVersion(query.getVersion());
      }
      if (subVersion != -1) {
        chkId.setSubVersion(subVersion);
      } else {
        chkId.setSubVersion(query.getSubVersion());
      }

      Condition *newCondition = nullptr;

      if (chkStorage) {
        newCondition = chkStorage->getObject(chkId);
      }
      if (!newCondition) {
        continue;
      }

      // object is found in specific storage: replace entry in the result list!
      chkCondition->setOwner(1);
      delete result->Remove(chkCondition);
      result->AddFirst(newCondition);
    }

    Int_t nEntries = result->GetEntries();
    LOG(INFO) << "After look into other specific storages, result list is:" << FairLogger::endl;
    for (int i = 0; i < nEntries; i++) {
      Condition *entry = (Condition *) result->At(i);
      LOG(INFO) << entry->getId().ToString().Data() << FairLogger::endl;
    }
  }

  // caching entries
  TIter iter(result);
  Condition *entry = nullptr;
  while ((entry = dynamic_cast<Condition *>(iter.Next()))) {

    if (!mIds->Contains(&entry->getId())) {
      mIds->Add(entry->getId().Clone());
    }
    if (mCache && (query.getFirstRun() == mRun)) {
      cacheCondition(entry->getId().getPathString(), entry);
    }
  }

  return result;
}

Bool_t Manager::putObject(TObject *object, const ConditionId &id, ConditionMetaData *metaData, const char *mirrors)
{
  // store an  Condition object into the database

  if (object == nullptr) {
    LOG(ERROR) << "Null Condition! No storage will be done!" << FairLogger::endl;
    return kFALSE;
  }

  Condition anCondition(object, id, metaData);
  return putObject(&anCondition, mirrors);
}

Bool_t Manager::putObject(Condition *entry, const char *mirrors)
{
  // store an  Condition object into the database

  if (!mDefaultStorage) {
    LOG(ERROR) << "No storage set!" << FairLogger::endl;
    return kFALSE;
  }

  if (!entry) {
    LOG(ERROR) << "No entry!" << FairLogger::endl;
    return kFALSE;
  }

  if (entry->getObject() == nullptr) {
    LOG(ERROR) << "No valid object in CDB entry!" << FairLogger::endl;
    return kFALSE;
  }

  if (!entry->getId().isValid()) {
    LOG(ERROR) << "Invalid entry ID: " << entry->getId().ToString().Data() << FairLogger::endl;
    return kFALSE;
  }

  if (!entry->getId().isSpecified()) {
    LOG(ERROR) << "Unspecified entry ID: " << entry->getId().ToString().Data() << FairLogger::endl;
    return kFALSE;
  }

  ConditionId id = entry->getId();
  StorageParameters *aPar = selectSpecificStorage(id.getPathString());

  Storage *aStorage = nullptr;

  if (aPar) {
    aStorage = getStorage(aPar);
  } else {
    aStorage = getDefaultStorage();
  }

  LOG(DEBUG) << "Storing object into storage: " << aStorage->getUri().Data() << FairLogger::endl;

  TString strMirrors(mirrors);
  Bool_t result = kFALSE;
  if (!strMirrors.IsNull() && !strMirrors.IsWhitespace()) {
    result = aStorage->putObject(entry, mirrors);
  } else {
    result = aStorage->putObject(entry, "");
  }

  if (mRun >= 0) {
    queryStorages();
  }

  return result;
}

void Manager::setMirrorSEs(const char *mirrors)
{
  // set mirror Storage Elements for the default storage, if it is of type "alien"
  if (mDefaultStorage->getStorageType() != "alien") {
    LOG(INFO) << R"(The default storage is not of type "alien". Settings for Storage Elements are not taken into account!)" << FairLogger::endl;
    return;
  }
  mDefaultStorage->setMirrorSEs(mirrors);
}

const char *Manager::getMirrorSEs() const
{
  // get mirror Storage Elements for the default storage, if it is of type "alien"
  if (mDefaultStorage->getStorageType() != "alien") {
    LOG(INFO) << R"(The default storage is not of type "alien". Settings for Storage Elements are not taken into account!)" << FairLogger::endl;
    return "";
  }
  return mDefaultStorage->getMirrorSEs();
}

void Manager::cacheCondition(const char *path, Condition *entry)
{
  // cache  Condition. Cache is valid until run number is changed.

  Condition *chkCondition = dynamic_cast<Condition *>(mConditionCache.GetValue(path));

  if (chkCondition) {
    LOG(DEBUG) << "Object " << path << " already in cache !!" << FairLogger::endl;
    return;
  } else {
    LOG(DEBUG) << "Caching entry " << path << FairLogger::endl;
  }

  mConditionCache.Add(new TObjString(path), entry);
  LOG(DEBUG) << "Cache entries: " << mConditionCache.GetEntries() << FairLogger::endl;
}

void Manager::print(Option_t * /*option*/) const
{
  // Print list of active storages and their URIs

  TString output = Form("Run number = %d; ", mRun);
  output += "Cache is ";
  if (!mCache) {
    output += "NOT ";
  }
  output += Form("ACTIVE; Number of active storages: %d\n", mActiveStorages.GetEntries());

  if (mDefaultStorage) {
    output += Form("\t*** Default Storage URI: \"%s\"\n", mDefaultStorage->getUri().Data());
  }
  if (mSpecificStorages.GetEntries() > 0) {
    TIter iter(mSpecificStorages.GetTable());
    TPair *aPair = nullptr;
    Int_t i = 1;
    while ((aPair = (TPair *) iter.Next())) {
      output += Form("\t*** Specific storage %d: Path \"%s\" -> URI \"%s\"\n", i++,
                     ((TObjString *) aPair->Key())->GetName(), ((StorageParameters *) aPair->Value())->getUri().Data());
    }
  }
  if (mdrainStorage) {
    output += Form("*** drain Storage URI: %s\n", mdrainStorage->getUri().Data());
  }
  LOG(INFO) << output.Data() << FairLogger::endl;
}

void Manager::setRun(Int_t run)
{
  // Sets current run number.
  // When the run number changes the caching is cleared.

  if (mRun == run) {
    return;
  }

  if (mLock && mRun >= 0) {
    LOG(FATAL) << "Lock is ON, cannot reset run number!" << FairLogger::endl;
  }

  mRun = run;

  if (mRaw) {
    // here the LHCPeriod xml file is parsed; the string containing the correct period is returned;
    // the default storage is set
    if (mStartRunLhcPeriod <= run && mEndRunLhcPeriod >= run) {
      LOG(INFO) << "LHCPeriod alien folder for current run already in memory" << FairLogger::endl;
    } else {
      setDefaultStorageFromRun(mRun);
      if (mConditionCache.GetEntries() != 0) {
        clearCache();
      }
      return;
    }
  }
  clearCache();
  queryStorages();
}

void Manager::clearCache()
{
  // clear  Condition cache

  LOG(DEBUG) << "Cache entries to be deleted: " << mConditionCache.GetEntries() << FairLogger::endl;

  /*
  // To clean entries one by one
  TIter iter(mConditionCache.GetTable());
  TPair* pair=0;
  while((pair= dynamic_cast<TPair*> (iter.Next()))){

  TObjString* key = dynamic_cast<TObjString*> (pair->Key());
   Condition* entry = dynamic_cast< Condition*> (pair->Value());
  LOG(DEBUG) << "Deleting entry: " << key->GetName() << FairLogger::endl;
  if (entry) delete entry;
  delete mConditionCache.Remove(key);
  }
  */
  mConditionCache.DeleteAll();
  LOG(DEBUG) << "After deleting - Cache entries: " << mConditionCache.GetEntries() << FairLogger::endl;
}

void Manager::unloadFromCache(const char *path)
{
  // unload cached object
  // that is remove the entry from the cache and the id from the list of ids
  //
  if (!mActiveStorages.GetEntries()) {
    LOG(DEBUG) << R"(No active storages. Object ")" << path << R"(" is not unloaded from cache)" << FairLogger::endl;
    return;
  }

  IdPath queryPath(path);
  if (!queryPath.isValid()) {
    return;
  }

  if (!queryPath.isWildcard()) { // path is not wildcard, get it directly from the cache and unload it!
    if (mConditionCache.Contains(path)) {
      LOG(DEBUG) << R"(Unloading object ")" << path << R"(" from cache and from list of ids)" << FairLogger::endl;
      TObjString pathStr(path);
      delete mConditionCache.Remove(&pathStr);
      // we do not remove from the list of ConditionId's (it's not very coherent but we leave the
      // id for the benefit of the userinfo)
      /*
         TIter iter(mIds);
         ConditionId *id = 0;
         while((id = dynamic_cast<ConditionId*> (iter.Next()))){
         if(queryPath.isSupersetOf(id->getPath()))
         delete mIds->Remove(id);
         }*/
    } else {
      LOG(WARNING) << R"(Cache does not contain object ")" << path << R"("!)" << FairLogger::endl;
    }
    LOG(DEBUG) << "Cache entries: " << mConditionCache.GetEntries() << FairLogger::endl;
    return;
  }

  // path is wildcard: loop on the cache and unload all comprised objects!
  TIter iter(mConditionCache.GetTable());
  TPair *pair = nullptr;
  Int_t removed = 0;

  while ((pair = dynamic_cast<TPair *>(iter.Next()))) {
    IdPath entryPath = pair->Key()->GetName();
    if (queryPath.isSupersetOf(entryPath)) {
      LOG(DEBUG) << R"(Unloading object ")" << entryPath.getPathString().Data() << R"(" from cache and from list of ids)"
                 << FairLogger::endl;
      TObjString pathStr(entryPath.getPathString());
      delete mConditionCache.Remove(&pathStr);
      removed++;

      // we do not remove from the list of ConditionId's (it's not very coherent but we leave the
      // id for the benefit of the userinfo)
      /*
         TIter iterids(mIds);
         ConditionId *anId = 0;
         while((anId = dynamic_cast<ConditionId*> (iterids.Next()))){
          IdPath aPath = anId->getPath();
         TString aPathStr = aPath.getPath();
         if(queryPath.isSupersetOf(aPath)) {
         delete mIds->Remove(anId);
         }
         }*/
    }
  }
  LOG(DEBUG) << "Cache entries and ids removed: " << removed << " Remaining: " << mConditionCache.GetEntries()
             << FairLogger::endl;
}

void Manager::destroyActiveStorages()
{
  // delete list of active storages

  mActiveStorages.DeleteAll();
  mSpecificStorages.DeleteAll();
}

void Manager::destroyActiveStorage(Storage * /*storage*/)
{
  // destroys active storage

  /*
     TIter iter(mActiveStorages.GetTable());
     TPair* aPair;
     while ((aPair = (TPair*) iter.Next())) {
     if(storage == ( Storage*) aPair->Value())
     delete mActiveStorages.Remove(aPair->Key());
     storage->Delete(); storage=0x0;
     }
     */
}

void Manager::queryStorages()
{
  // query default and specific storages for files valid for mRun. Every storage loads the Ids into
  // its list.

  if (mRun < 0) {
    LOG(ERROR) << "Run number not yet set! Use  Manager::setRun." << FairLogger::endl;
    return;
  }
  if (!mDefaultStorage) {
    LOG(ERROR) << "Default storage is not set! Use  Manager::setDefaultStorage" << FairLogger::endl;
    return;
  }
  if (mDefaultStorage->getStorageType() == "alien" || mDefaultStorage->getStorageType() == "local") {
    mDefaultStorage->queryStorages(mRun);
    //} else {
    //	LOG(DEBUG) << "Skipping query for valid files, it used only in grid..." << FairLogger::endl;
  }

  TIter iter(&mSpecificStorages);
  TObjString *aCalibType = nullptr;
  StorageParameters *aPar = nullptr;
  while ((aCalibType = dynamic_cast<TObjString *>(iter.Next()))) {
    aPar = (StorageParameters *) mSpecificStorages.GetValue(aCalibType);
    if (aPar) {
      LOG(DEBUG) << "Querying specific storage " << aCalibType->GetName() << FairLogger::endl;
      Storage *aStorage = getStorage(aPar);
      if (aStorage->getStorageType() == "alien" || aStorage->getStorageType() == "local") {
        aStorage->queryStorages(mRun, aCalibType->GetName());
      } else {
        LOG(DEBUG) << "Skipping query for valid files, it is used only in grid..." << FairLogger::endl;
      }
    }
  }
}

Bool_t Manager::diffObjects(const char *cdbFile1, const char *cdbFile2) const
{
  // Compare byte-by-byte the objects contained in the CDB entry in two different files,
  // whose name is passed as input
  // Return value:
  //   kTRUE - in case the content of the OCDB object (persistent part) is exactly the same
  //   kFALSE - otherwise

  TString f1Str(cdbFile1);
  TString f2Str(cdbFile2);
  if (!gGrid && (f1Str.BeginsWith("alien://") || f2Str.BeginsWith("alien://"))) {
    TGrid::Connect("alien://");
  }

  TFile *f1 = TFile::Open(cdbFile1);
  if (!f1) {
    Printf(R"(Cannot open file "%s")", cdbFile1);
    return kFALSE;
  }
  TFile *f2 = TFile::Open(cdbFile2);
  if (!f2) {
    Printf(R"(Cannot open file "%s")", cdbFile2);
    return kFALSE;
  }

  Condition *entry1 = (Condition *) f1->Get(" Condition");
  if (!entry1) {
    Printf(R"(Cannot get CDB entry from file "%s")", cdbFile1);
    return kFALSE;
  }
  Condition *entry2 = (Condition *) f2->Get(" Condition");
  if (!entry2) {
    Printf(R"(Cannot get CDB entry from file "%s")", cdbFile2);
    return kFALSE;
  }

  // stream the two objects in the buffer of two TMessages
  TObject *object1 = entry1->getObject();
  TObject *object2 = entry2->getObject();
  TMessage *file1 = new TMessage(TBuffer::kWrite);
  file1->WriteObject(object1);
  Int_t size1 = file1->Length();
  TMessage *file2 = new TMessage(TBuffer::kWrite);
  file2->WriteObject(object2);
  Int_t size2 = file2->Length();
  if (size1 != size2) {
    Printf("Problem 2:  OCDB entry of different size (%d,%d)", size1, size2);
    return kFALSE;
  }

  // if the two buffers have the same size, check that they are the same byte-by-byte
  Int_t countDiff = 0;
  char *buf1 = file1->Buffer();
  char *buf2 = file2->Buffer();
  // for (Int_t i=0; i<size1; i++)    if (file1->Buffer()[i]!=file2->Buffer()[i]) countDiff++;
  for (Int_t i = 0; i < size1; i++) {
    if (buf1[i] != buf2[i]) {
      countDiff++;
    }
  }

  if (countDiff > 0) {
    Printf("The CDB objects differ by %d bytes.", countDiff);
    return kFALSE;
  }

  Printf("The CDB objects are the same in the two files.");
  return kTRUE;
}

ULong64_t Manager::setLock(Bool_t lock, ULong64_t key)
{
  // To lock/unlock user must provide the key. A new key is provided after
  // each successful lock. User should always backup the returned key and
  // use it on next access.
  if (mLock == lock) {
    return 0;
  } // nothing to be done
  if (lock) {
    // User wants to lock - check his identity
    if (mKey) {
      // Lock has a user - check his key
      if (mKey != key) {
        LOG(FATAL) << "Wrong key provided to lock CDB. Please remove CDB lock access from your code !"
                   << FairLogger::endl;
        return 0;
      }
    }
    // Provide new key
    mKey = gSystem->Now();
    mLock = kTRUE;
    return mKey;
  }
  // User wants to unlock - check the provided key
  if (key != mKey) {
    LOG(FATAL) << "Lock is ON: wrong key provided" << FairLogger::endl;
    return 0;
  }
  mLock = kFALSE;
  return key;
}

void Manager::extractBaseFolder(TString &url)
{
  // TBD RS
  // remove everything but the url -
  // Exact copy of the AliReconstuction::Rectify.... (to be removed)
  //
  //
  TString sbs;
  if (!(sbs = url(R"(\?User=[^?]*)")).IsNull()) {
    url.ReplaceAll(sbs, "");
  }
  if (!(sbs = url(R"(\?DBFolder=[^?]*)")).IsNull()) {
    url.ReplaceAll("?DB", "");
  }
  if (!(sbs = url(R"(\?SE=[^?]*)")).IsNull()) {
    url.ReplaceAll(sbs, "");
  }
  if (!(sbs = url(R"(\?CacheFolder=[^?]*)")).IsNull()) {
    url.ReplaceAll(sbs, "");
  }
  if (!(sbs = url(R"(\?OperateDisconnected=[^?]*)")).IsNull()) {
    url.ReplaceAll(sbs, "");
  }
  if (!(sbs = url(R"(\?CacheSize=[^?]*)")).IsNull()) {
    url.ReplaceAll(sbs, "");
  }
  if (!(sbs = url(R"(\?CleanupInterval=[^?]*)")).IsNull()) {
    url.ReplaceAll(sbs, "");
  }
  Bool_t slash = kFALSE, space = kFALSE;
  while ((slash = url.EndsWith("/")) || (space = url.EndsWith(" "))) {
    if (slash) {
      url = url.Strip(TString::kTrailing, '/');
    }
    if (space) {
      url = url.Strip(TString::kTrailing, ' ');
    }
  }
  // url.ToLower();
  //
}

// interface to specific  Parameter class           //
// (GridStorageParam,  LocalStorageParam,  DumpParam)  //

StorageParameters::StorageParameters() : mType(), mURI()
{
  // constructor
}

StorageParameters::~StorageParameters()
{
  // destructor
}
