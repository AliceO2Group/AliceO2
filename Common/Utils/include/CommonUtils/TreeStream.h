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

/// @brief Class for creating debug root trees with std::cout like intervace
/// @author Marian Ivanov, marian.ivanov@cern.ch (original code in AliRoot)
///         Ruben Shahoyan, ruben.shahoyan@cern.ch (porting to O2).

#ifndef ALICEO2_TREESTREAM_H
#define ALICEO2_TREESTREAM_H

#include <TString.h>
#include <TTree.h>
#include <vector>
#include "GPUCommonDef.h"

class TBranch;
class TClass;
class TDataType;

namespace o2
{
namespace utils
{
/// The TreeStream class allows creating a root tree of any objects having root
/// dictionary, using operator<< interface, and w/o prior tree declaration.
/// The format is:
/// treeStream << "branchName0="<<objPtr
///            <<"branchName1="<<objRed
///            <<"branchName2="
///            <<elementaryTypeVar<<"\n"
///
/// See testTreeStream.cxx for functional example
///
class TreeStream
{
 public:
  struct TreeDataElement {
    char type = 0;               ///< type of data element
    const TClass* cls = nullptr; ///< data type pointer
    const void* ptr = nullptr;   ///< pointer to element
    std::string name;            ///< name of the element
  };

  TreeStream(const char* treename);
  TreeStream() = default;
  virtual ~TreeStream() = default;
  void Close() { mTree.Write(); }
  Int_t CheckIn(Char_t type, const void* pointer);
  void BuildTree();
  void Fill();
  Double_t getSize() { return mTree.GetZipBytes(); }
  TreeStream& Endl();

  TTree& getTree() { return mTree; }
  const char* getName() const { return mTree.GetName(); }
  void setID(int id) { mID = id; }
  int getID() const { return mID; }
  TreeStream& operator<<(const Bool_t& b)
  {
    CheckIn('B', &b);
    return *this;
  }

  TreeStream& operator<<(const Char_t& c)
  {
    CheckIn('B', &c);
    return *this;
  }

  TreeStream& operator<<(const UChar_t& c)
  {
    CheckIn('b', &c);
    return *this;
  }

  TreeStream& operator<<(const Short_t& h)
  {
    CheckIn('S', &h);
    return *this;
  }

  TreeStream& operator<<(const UShort_t& h)
  {
    CheckIn('s', &h);
    return *this;
  }

  TreeStream& operator<<(const Int_t& i)
  {
    CheckIn('I', &i);
    return *this;
  }

  TreeStream& operator<<(const UInt_t& i)
  {
    CheckIn('i', &i);
    return *this;
  }

  TreeStream& operator<<(const Long_t& l)
  {
    CheckIn('L', &l);
    return *this;
  }

  TreeStream& operator<<(const ULong_t& l)
  {
    CheckIn('l', &l);
    return *this;
  }

  TreeStream& operator<<(const Long64_t& l)
  {
    CheckIn('L', &l);
    return *this;
  }

  TreeStream& operator<<(const ULong64_t& l)
  {
    CheckIn('l', &l);
    return *this;
  }

  TreeStream& operator<<(const Float_t& f)
  {
    CheckIn('F', &f);
    return *this;
  }

  TreeStream& operator<<(const Double_t& d)
  {
    CheckIn('D', &d);
    return *this;
  }

  TreeStream& operator<<(const Char_t* name);

  template <class T>
  TreeStream& operator<<(const T* obj)
  {
    CheckIn(obj);
    return *this;
  }

  template <class T, typename std::enable_if<!std::is_pointer<GPUgeneric() T>::value, bool>::type* = nullptr>
  TreeStream& operator<<(const T& obj)
  {
    CheckIn(&obj);
    return *this;
  }

  template <class T>
  Int_t CheckIn(const T* obj);

 private:
  //
  std::vector<TreeDataElement> mElements;
  std::vector<TBranch*> mBranches; ///< pointers to branches
  TTree mTree;                     ///< data storage
  int mCurrentIndex = 0;           ///< index of current element
  int mID = -1;                    ///< identifier of layout
  int mNextNameCounter = 0;        ///< next name counter
  int mStatus = 0;                 ///< status of the layout
  TString mNextName;               ///< name for next entry

  ClassDefNV(TreeStream, 0);
};

template <class T>
Int_t TreeStream::CheckIn(const T* obj)
{
  // check in arbitrary class having dictionary
  TClass* pClass = nullptr;
  if (obj) {
    pClass = TClass::GetClass(typeid(*obj));
  }

  if (mCurrentIndex >= static_cast<int>(mElements.size())) {
    mElements.emplace_back();
    auto& element = mElements.back();
    element.cls = pClass;
    TString name = mNextName;
    if (name.Length()) {
      if (mNextNameCounter > 0) {
        name += mNextNameCounter;
      }
    } else {
      name = TString::Format("B%d", static_cast<int>(mElements.size()));
    }
    element.name = name.Data();
    element.ptr = obj;
  } else {
    auto& element = mElements[mCurrentIndex];
    if (!element.cls) {
      element.cls = pClass;
    } else {
      if (element.cls != pClass && pClass) {
        mStatus++;
        return 1; // mismatched data element
      }
    }
    element.ptr = obj;
  }
  mCurrentIndex++;
  return 0;
}

} // namespace utils
} // namespace o2

#endif
