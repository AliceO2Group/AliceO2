// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   MemFileUtils.h
/// @brief  Utilities to create a memory image of the object stored to file

#ifndef O2_MEMFILE_UTILS_H
#define O2_MEMFILE_UTILS_H

#include <typeinfo>
#include <utility>
#include <TMemFile.h>
#include <TTree.h>
#include "Framework/Logger.h"
#include "CommonUtils/StringUtils.h"

namespace o2
{
namespace utils
{

/// Static class for with methods to dump any root-persistent class to a TMemFile or its
/// binary image

struct MemFileHelper {
  using FileImage = std::vector<char>;

  //________________________________________________________________
  /// get the class name of the object
  template <typename T>
  inline static std::string getClassName(const T& obj)
  {
    return getClassName(typeid(T));
  }

  //________________________________________________________________
  /// dump object into the TMemFile named fileName. The stored object will be named according to the optName or its className
  template <typename T>
  inline static std::unique_ptr<TMemFile> createTMemFile(const T& obj, const std::string& fName, const std::string& optName = "")
  {
    return createTMemFile(&obj, typeid(T), fName, optName);
  }

  //________________________________________________________________
  /// create binary image of the TMemFile containing the object and named fileName.
  /// The stored object will be named according to the objName
  static std::unique_ptr<FileImage> createFileImage(const TObject& obj, const std::string& fileName, const std::string& objName)
  {
    auto memfUPtr = createTMemFile(obj, fileName, objName);
    std::unique_ptr<FileImage> img = std::make_unique<FileImage>(memfUPtr->GetSize());
    auto sz = memfUPtr->CopyTo(img->data(), memfUPtr->GetSize());
    img->resize(sz);
    return img;
  }

  //________________________________________________________________
  /// create binary image of the TMemFile containing the object and named fileName.
  /// The stored object will be named according to the objName
  template <typename T>
  inline static std::unique_ptr<FileImage> createFileImage(const T& obj, const std::string& fileName, const std::string& optName = "")
  {
    return createFileImage(&obj, typeid(T), fileName, optName);
  }

  //________________________________________________________________
  /// get the class name of the object
  static std::string getClassName(const std::type_info& tinfo)
  {
    auto tcl = TClass::GetClass(tinfo);
    std::string clname;
    if (!tcl) {
      LOG(error) << "Could not retrieve ROOT dictionary for type " << tinfo.name();
    } else {
      clname = tcl->GetName();
      o2::utils::Str::trim(clname);
    }
    return clname;
  }

  //________________________________________________________________
  /// dump TObject the TMemFile named fileName. The stored object will be named according to the optName or its className
  static std::unique_ptr<TMemFile> createTMemFile(const TObject& rootObject, const std::string& fName, const std::string& objName)
  {
    std::unique_ptr<TMemFile> uptr;
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 18, 0)
    uptr = std::make_unique<TMemFile>(fName.c_str(), "RECREATE");
#else
    uptr = std::make_unique<TMemFile>(fName.c_str(), "RECREATE", "", ROOT::RCompressionSetting::EDefaults::kUseGeneralPurpose, 1024);
#endif
    if (uptr->IsZombie()) {
      uptr->Close();
      uptr.reset();
      throw std::runtime_error(std::string("Error opening memory file ") + fName.c_str());
    } else {
      rootObject.Write(objName.c_str());
      uptr->Close();
    }
    return uptr;
  }

  //________________________________________________________________
  /// dump object into the TMemFile named fileName. The stored object will be named according to the optName or its className
  static std::unique_ptr<TMemFile> createTMemFile(const void* obj, const std::type_info& tinfo, const std::string& fName, const std::string& optName = "")
  {
    std::unique_ptr<TMemFile> uptr;
    auto tcl = TClass::GetClass(tinfo);
    std::string clsName;
    if (!tcl) {
      LOG(error) << "Could not retrieve ROOT dictionary for type " << tinfo.name();
      return uptr;
    } else {
      clsName = tcl->GetName();
      o2::utils::Str::trim(clsName);
    }
    bool isTree = tcl->InheritsFrom("TTree");
    int compressLevel = isTree ? ROOT::RCompressionSetting::EDefaults::kUseCompiledDefault : ROOT::RCompressionSetting::EDefaults::kUseGeneralPurpose;
    Long64_t blSize = isTree ? 0LL : 1024LL;

    std::string objName = optName.empty() ? clsName : optName;
    std::string fileName = fName.empty() ? (objName + ".root") : fName;
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 18, 0)
    uptr = std::make_unique<TMemFile>(fileName.c_str(), "RECREATE");
#else
    uptr = std::make_unique<TMemFile>(fileName.c_str(), "RECREATE", "", compressLevel, blSize);
#endif
    if (uptr->IsZombie()) {
      uptr->Close();
      uptr.reset();
      throw std::runtime_error(std::string("Error opening memory file ") + fileName.c_str());
    } else {
      const TTree* treePtr = nullptr;
      if (isTree) { // trees are special: need to create a new file-resident tree
        treePtr = const_cast<TTree*>((const TTree*)obj)->CloneTree(-1, "");
        obj = treePtr;
      }
      uptr->WriteObjectAny(obj, clsName.c_str(), objName.c_str());
      delete treePtr;
      uptr->Close();
    }
    return uptr;
  }

  //________________________________________________________________
  /// create binary image of the TMemFile containing the object and named fileName.
  /// The stored object will be named according to the optName or its className
  static std::unique_ptr<FileImage> createFileImage(const void* obj, const std::type_info& tinfo, const std::string& fileName, const std::string& optName = "")
  {
    auto memfUPtr = createTMemFile(obj, tinfo, fileName, optName);
    std::unique_ptr<FileImage> img = std::make_unique<FileImage>(memfUPtr->GetSize());
    auto sz = memfUPtr->CopyTo(img->data(), memfUPtr->GetSize());
    img->resize(sz);
    return img;
  }

  ClassDefNV(MemFileHelper, 1);
};

} // namespace utils
} // namespace o2

#endif
