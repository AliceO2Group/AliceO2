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

#ifndef ALICEO2_CDB_PATH_H_
#define ALICEO2_CDB_PATH_H_

//  Path string identifying the object:  			   //
//  "level0/level1/level2" 					   //
//  (example: "ZDC/Calib/Pedestals") 		         	   //
#include <TObject.h> // for TObject
#include <TString.h> // for TString
#include "Rtypes.h"  // for Bool_t, IdPath::Class, ClassDef, etc

namespace o2
{
namespace ccdb
{

class IdPath : public TObject
{

 public:
  IdPath();

  IdPath(const IdPath& other);

  IdPath(const char* level0, const char* level1, const char* level2);

  IdPath(const char* path);

  IdPath(const TString& path);

  ~IdPath() override;

  const TString& getPathString() const
  {
    return mPath;
  }

  void setPath(const char* path)
  {
    mPath = path;
    InitPath();
  }

  const char* getLevel(Int_t i) const;

  Bool_t isValid() const
  {
    return mValid;
  }

  Bool_t isWildcard() const
  {
    return mWildcard;
  }

  Bool_t doesLevel0Contain(const TString& str) const;

  Bool_t doesLevel1Contain(const TString& str) const;

  Bool_t doesLevel2Contain(const TString& str) const;

  Bool_t isSupersetOf(const IdPath& other) const;

 private:
  Bool_t isWord(const TString& str);

  void InitPath();

  void init();

  TString mPath;   // detector pathname (Detector/DBType/SpecType)
  TString mLevel0; // level0 name (ex. detector: ZDC, TPC...)
  TString mLevel1; // level1 name (ex. DB type, Calib, Align)
  TString mLevel2; // level2 name (ex. DetSpecType, pedestals, gain...)

  Bool_t mValid;    // validity flag
  Bool_t mWildcard; // wildcard flag

  ClassDefOverride(IdPath, 1);
};
} // namespace ccdb
} // namespace o2
#endif
