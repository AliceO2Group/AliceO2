// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/* Wrapping some functionality of handling ROOT files
 * Hides details of tree and directory access and modification
 */
#ifndef ROOT_IO_UTILITIES_H_
#define ROOT_IO_UTILITIES_H_

#include <type_traits> // for std::is_pointer
#include <string>
#include <unordered_map>

#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"

namespace o2
{
namespace mcstepanalysis
{

enum class ETFileMode : int { kREAD = 0,
                              kUPDATE = 1,
                              kRECREATE = 2 };

class ROOTIOUtilities
{
 public:
  /// default constructor
  ROOTIOUtilities(const std::string& path = "", ETFileMode mode = ETFileMode::kREAD);
  /// provide explicit desctructor
  ~ROOTIOUtilities();
  //
  // additional steering
  //
  /// close everything and reset
  void close(bool finalAction = true);
  //
  // TDirectory operations
  //
  /// change to directory sticking to current ETFileMode
  bool changeToTDirectory(const std::string& dirname = "");
  /// check whether an object with desired name is in current directory
  bool hasObject(const std::string& name);
  /// write method for standard ROOT TObjects
  bool writeObject(const TObject* object);
  /// write any object to a directory
  template <typename T>
  bool writeObject(const T object, const std::string& name)
  {
    static_assert(std::is_pointer<T>::value, "ROOTIOUtilities::writeObject: Expected a pointer for the object to be written.");
    changeToTDirectory(mTDirectoryName);
    return (mTDirectory->WriteObject(object, name.c_str()) > 0);
  }
  /// reading objects of any type by number of entry
  template <typename T>
  void readObject(T*& object, int entry = -1)
  {
    static_assert(std::is_pointer<T*>::value, "ROOTIOUtilities::readObject: Expected a pointer for the object to be read.");
    changeToTDirectory(mTDirectoryName);
    TKey* key = nullptr;

    if (entry > -1 && entry < mTDirectoryEntries) {
      key = dynamic_cast<TKey*>(mObjectList->At(entry));
      object = key->ReadObject<T>();
    } else if (entry < 0 && mTDirectoryCounter < mTDirectoryEntries) {
      key = dynamic_cast<TKey*>(mObjectList->At(mTDirectoryCounter++));
      object = key->ReadObject<T>();
    } else {
      object = nullptr;
    }
  }

  template <typename T>
  void readObject(T& object, int entry = -1)
  {
    changeToTDirectory(mTDirectoryName);
    TKey* key = nullptr;
    if (entry > -1 && entry < mTDirectoryEntries) {
      key = dynamic_cast<TKey*>(mObjectList->At(entry));
      object = *(key->ReadObject<T>());
    } else if (entry < 0 && mTDirectoryCounter < mTDirectoryEntries) {
      key = dynamic_cast<TKey*>(mObjectList->At(mTDirectoryCounter++));
      object = *(key->ReadObject<T>());
    }
  }

  /// reading objects of any type by name
  template <typename T>
  void readObject(T*& object, const std::string& name)
  {
    static_assert(std::is_pointer<T*>::value, "ROOTIOUtilities::readObject: Expected a pointer for the object to be read.");
    changeToTDirectory(mTDirectoryName);
    if (mTDirectoryEntries > 0) {
      mTDirectory->GetObject(name.c_str(), object);
    } else {
      object = nullptr;
    }
  }

  /// reading objects of any type by name
  template <typename T>
  void readObject(T& object, const std::string& name)
  {
    changeToTDirectory(mTDirectoryName);
    if (mTDirectoryEntries > 0) {
      T* objectCast = nullptr;
      mTDirectory->GetObject(name.c_str(), objectCast);
      if (objectCast) {
        object = *(objectCast);
      }
    }
  }
  //
  // TTree operations
  //
  /// silently change TTree sticking to current ETFileMode
  bool changeToTTree(const std::string& treename);
  /// silently change TTree changing ETFileMode on the fly
  bool changeToTTree(const std::string& treename, ETFileMode mode);
  /// set an address to some branchname
  template <typename T>
  bool setBranch(const std::string& branchname, T** address)
  {
    if (mTTreeOpened) {
      // check whether branch is available
      if (mTTree->GetBranch(branchname.c_str())) {
        mTTree->SetBranchAddress(branchname.c_str(), address);
        return true;
      }
      // if update or recreate, the branch might not be there yet, create it
      else if (mTFileMode == ETFileMode::kRECREATE || mTFileMode == ETFileMode::kUPDATE) {
        mTTree->Branch(branchname.c_str(), address);
        return true;
      }
    }
    return false;
  }
  /// reset internal counters
  void resetTTreeCounter();
  /// reset branch addresses
  void resetTTreeConnection();
  /// disconect from current TTree
  bool closeTTree(bool finalAction = true);
  /// either read or write. The decision is made automatically by checking the opening mode
  /// of the TFile
  bool processTTree(int event = -1);
  /// do final actions depending on TFile mode
  bool finalizeTTree();
  //
  // setting
  //
  /// set path to file to be accessed
  void changeTFile(const std::string& path, ETFileMode mode = ETFileMode::kREAD);
  /// set another ETFileMode
  void changeTFileMode(ETFileMode mode);
  //
  // getting
  //
  /// return treename
  std::string getTTreename() const;
  /// get number of entries in TTree
  int nEntries() const;

 private:
  /// open and close the TFile
  bool openTFile();
  /// close the TFile
  void closeTFile(bool finalAction = true);
  /// reset the object list read?written to a TDirectory
  void resetKeyList();
  /// fetch data from tree, event-wise
  bool fetchData(int event = -1);
  /// flush data to TTree
  bool flushToTTree();

 private:
  /// from where the ROOT file was opened
  std::string mFilepath;
  /// internal pointer to ROOT file
  TFile* mTFile;
  /// additional flag to check whether ROOT file is opened
  // \todo may be removed
  bool mTFileOpened;
  /// mode of the opened ROOT file
  ETFileMode mTFileMode;
  /// internal pointer to current TDirectory in ROOT file to easily jump
  /// between directories
  TDirectory* mTDirectory;
  /// current name of directory
  std::string mTDirectoryName;
  /// list of objects in the current TDirectory
  TList* mObjectList;
  /// number of entries in the current TDirectory
  int mTDirectoryEntries;
  /// internal counter/position in current TDirectory
  int mTDirectoryCounter;
  /// pointer to current TTree
  TTree* mTTree;
  /// additional flag to check whether TTree is in use
  bool mTTreeOpened;
  /// internal position in current TTree
  int mTTreeCounter;
  /// number of entries in current TTree
  int mTTreeEntries;
  /// file modes
  static const std::unordered_map<ETFileMode, const char*> mTFileModesNames;

  ClassDefNV(ROOTIOUtilities, 1);
};
} // end of namespace mcstepanalysis
} // end of namespace o2
#endif /* ROOT_IO_UTILITIES_H_ */
