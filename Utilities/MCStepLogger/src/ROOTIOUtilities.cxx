// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCStepLogger/ROOTIOUtilities.h"

ClassImp(o2::mcstepanalysis::ROOTIOUtilities);

using namespace o2::mcstepanalysis;

const std::unordered_map<ETFileMode, const char*> ROOTIOUtilities::mTFileModesNames = {
  { ETFileMode::kREAD, "READ" },
  { ETFileMode::kUPDATE, "UPDATE" },
  { ETFileMode::kRECREATE, "RECREATE" }
};

ROOTIOUtilities::ROOTIOUtilities(const std::string& path, ETFileMode mode)
  : mFilepath(path), mTFile(nullptr), mTFileOpened(false), mTFileMode(mode), mTDirectory(nullptr), mTDirectoryName(""), mObjectList(nullptr), mTDirectoryEntries(0), mTDirectoryCounter(0), mTTree(nullptr), mTTreeOpened(false), mTTreeCounter(0), mTTreeEntries(0)
{
}

ROOTIOUtilities::~ROOTIOUtilities()
{
  close();
}

void ROOTIOUtilities::changeTFile(const std::string& path, ETFileMode mode)
{
  closeTFile();
  mFilepath = path;
  mTFileMode = mode;
  mTDirectoryName = "";
}

void ROOTIOUtilities::changeTFileMode(ETFileMode mode)
{
  if (mode == mTFileMode) {
    return;
  }
  mTFileMode = mode;
  openTFile();
  changeToTDirectory();
  closeTTree();
}

bool ROOTIOUtilities::openTFile()
{
  auto it = mTFileModesNames.find(mTFileMode);

  if (mTFileOpened) {
    // check whether changing the mode is successful, if not, close the TFile
    if (mTFile->ReOpen(it->second) < 0) {
      closeTFile();
    }
  }
  // if no TFile was opened or if it was closed due to failed mode change, try to open it again from scratch
  if (!mTFileOpened && !mFilepath.empty()) {
    mTFile = new TFile(mFilepath.c_str(), it->second);
  }
  if (mTFile) {
    mTFile->cd();
    mTDirectory = mTFile->GetDirectory(0);
    mTFileOpened = true;
  }
  return mTFileOpened;
}

void ROOTIOUtilities::closeTFile(bool finalAction)
{
  if (mTFileOpened) {
    // do final stuff if finalAction flag is set
    if (finalAction) {
      finalizeTTree();
    }
    // disconnect addresses from TTree and reset internals
    closeTTree();
    mTFile->Close();
    // free memory and reset open status
    delete mTFile;
    mTFile = nullptr;
    mObjectList = nullptr;
    mTDirectory = nullptr;
    mTFileOpened = false;
  }
}
void ROOTIOUtilities::close(bool finalAction)
{
  // so far, only the TFile needs to be closed
  closeTFile(finalAction);
}

bool ROOTIOUtilities::changeToTDirectory(const std::string& dirname)
{
  // if file not yet opened
  if (!mTFileOpened) {
    openTFile();
  }
  // if it was already opened, check if we are already in the desired directory
  else if (dirname.compare(mTDirectoryName) == 0) {
    return true;
  }
  // now the file might be open
  if (mTFileOpened) {
    // update pointer to directory
    mTDirectory = mTFile->GetDirectory(dirname.c_str());

    if (mTDirectory) {
      mTFile->cd(dirname.c_str());
      mTDirectoryName = dirname;
    } else if (mTFileMode == ETFileMode::kRECREATE || mTFileMode == ETFileMode::kUPDATE) {
      mTDirectory = mTFile->mkdir(dirname.c_str());
      // this could also fail e.g. if a directory above the root directory was desired
      if (mTDirectory) {
        mTFile->cd(dirname.c_str());
        mTDirectoryName = dirname;
      }
    }
  }
  resetKeyList();
  return (mTDirectory != nullptr);
}

bool ROOTIOUtilities::hasObject(const std::string& name)
{
  changeToTDirectory(mTDirectoryName);
  TKey* key = nullptr;
  TIter next(mObjectList);
  while ((key = dynamic_cast<TKey*>(next()))) {
    if (name.compare(key->GetName()) == 0) {
      return true;
    }
  }
  return false;
}

void ROOTIOUtilities::resetKeyList()
{
  if (mTDirectory) {
    mObjectList = mTDirectory->GetListOfKeys();
    mTDirectoryEntries = mObjectList->GetEntries();
  } else {
    mObjectList = nullptr;
    mTDirectoryEntries = 0;
  }
  mTDirectoryCounter = 0;
}

bool ROOTIOUtilities::changeToTTree(const std::string& treename)
{
  // if file not yet opened
  if (!mTFileOpened) {
    // if it's not possible to open the desired file, just return
    if (!openTFile()) {
      return false;
    }
  }

  // If desired treename is the same as the opened tree, keep it open but reset counters etc.
  else if (mTTreeOpened) {
    if (treename.compare(mTTree->GetName()) == 0) {
      resetTTreeConnection();
      return true;
    }
    // if it's a different tree, close it entirely
    closeTTree();
  }

  // first, try to get existing TTree by name
  mTTree = dynamic_cast<TTree*>(mTFile->Get(treename.c_str()));
  // If it does not exist, try to create one depending on the file mode
  if (!mTTree && (mTFileMode == ETFileMode::kRECREATE || mTFileMode == ETFileMode::kUPDATE)) {
    mTTree = new TTree(treename.c_str(), treename.c_str());
  }
  // check, whether on of the former attempts was successful
  if (mTTree) {
    mTTreeOpened = true;
    resetTTreeCounter();
  }
  return mTTreeOpened;
}
void ROOTIOUtilities::resetTTreeCounter()
{
  mTTreeCounter = 0;
  if (mTTreeOpened) {
    mTTreeEntries = mTTree->GetEntries();
  } else {
    mTTreeEntries = 0;
  }
}
void ROOTIOUtilities::resetTTreeConnection()
{
  if (mTTreeOpened) {
    mTTree->ResetBranchAddresses();
    resetTTreeCounter();
  }
}
bool ROOTIOUtilities::closeTTree(bool finalAction)
{
  bool success = true;
  if (mTTreeOpened) {
    if (finalAction) {
      success = finalizeTTree();
    }
    resetTTreeConnection();
    delete mTTree;
    mTTreeOpened = false;
    resetTTreeCounter();
  }
  return success;
}

bool ROOTIOUtilities::fetchData(int event)
{
  // is the tree even open?
  if (!mTTreeOpened) {
    return false;
  }
  // out of bounds?
  if (event >= mTTreeEntries) {
    return false;
  }
  int gotten = 0;
  // specific entry required
  if (event > -1) {
    gotten = mTTree->GetEntry(event);
  }
  // just get the next one
  else {
    gotten = mTTree->GetEntry(mTTreeCounter++);
  }
  return (gotten > 0);
}
bool ROOTIOUtilities::flushToTTree()
{
  if (!mTTreeOpened) {
    return false;
  }
  int success = mTTree->Fill();
  return (success > -1);
}
bool ROOTIOUtilities::processTTree(int event)
{
  if ((mTFileOpened && mTFileMode == ETFileMode::kRECREATE) || mTFileMode == ETFileMode::kUPDATE) {
    return flushToTTree();
  } else {
    return fetchData(event);
  }
}
bool ROOTIOUtilities::finalizeTTree()
{
  if ((mTFileOpened && mTFileMode == ETFileMode::kRECREATE) || mTFileMode == ETFileMode::kUPDATE) {
    return (mTFile->Write() > 0);
  }
  return false;
}
std::string ROOTIOUtilities::getTTreename() const
{
  if (mTTree) {
    return mTTree->GetName();
  }
  return "UNKNOWNTTREE";
}

int ROOTIOUtilities::nEntries() const
{
  if (mTTreeOpened) {
    return mTTreeEntries;
  }
  return -1;
}

bool ROOTIOUtilities::writeObject(const TObject* object)
{
  changeToTDirectory(mTDirectoryName);
  return (object->Write() > 0);
}
