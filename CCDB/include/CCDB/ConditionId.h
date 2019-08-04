// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_CDB_OBJECTID_H_
#define ALICEO2_CDB_OBJECTID_H_

//  Identity of an object stored into a database:  		   //
//  path, run validity range, version, subVersion 		   //
#include <TObject.h>         // for TObject
#include "CCDB/IdPath.h"     // for IdPath
#include "CCDB/IdRunRange.h" // for IdRunRange
#include "Rtypes.h"          // for Int_t, Bool_t, ConditionId::Class, ClassDef, etc
#include "TString.h"         // for TString

namespace o2
{
namespace ccdb
{

class ConditionId : public TObject
{

 public:
  ConditionId();

  ConditionId(const ConditionId& other);

  ConditionId(const IdPath& path, const IdRunRange& runRange, Int_t version = -1, Int_t subVersion = -1);

  ConditionId(const IdPath& path, Int_t firstRun, Int_t lastRun, Int_t verison = -1, Int_t subVersion = -1);

  static ConditionId* makeFromString(const TString& idString);

  ~ConditionId() override;

  const IdPath& getPath() const
  {
    return mPath;
  }

  const TString& getPathString() const
  {
    return mPath.getPathString();
  }

  const TString getPathLevel(Int_t i) const
  {
    return mPath.getLevel(i);
  }

  Bool_t isWildcard() const
  {
    return mPath.isWildcard();
  }

  void setPath(const char* path)
  {
    mPath.setPath(path);
  }

  const IdRunRange& getIdRunRange() const
  {
    return mIdRunRange;
  }

  IdRunRange& getIdRunRange()
  {
    return mIdRunRange;
  }

  Int_t getFirstRun() const
  {
    return mIdRunRange.getFirstRun();
  }

  Int_t getLastRun() const
  {
    return mIdRunRange.getLastRun();
  }

  void setFirstRun(Int_t firstRun)
  {
    mIdRunRange.setFirstRun(firstRun);
  }

  void setLastRun(Int_t lastRun)
  {
    mIdRunRange.setLastRun(lastRun);
  }

  void setIdRunRange(Int_t firstRun, Int_t lastRun)
  {
    mIdRunRange.setIdRunRange(firstRun, lastRun);
  }

  Bool_t isAnyRange() const
  {
    return mIdRunRange.isAnyRange();
  }

  Int_t getVersion() const
  {
    return mVersion;
  }

  Int_t getSubVersion() const
  {
    return mSubVersion;
  }

  void setVersion(Int_t version)
  {
    mVersion = version;
  }

  void setSubVersion(Int_t subVersion)
  {
    mSubVersion = subVersion;
  }

  const TString& getLastStorage() const
  {
    return mLastStorage;
  }

  void setLastStorage(TString& lastStorage)
  {
    mLastStorage = lastStorage;
  }

  Bool_t isValid() const;

  Bool_t isSpecified() const
  {
    return !(isWildcard() || isAnyRange());
  }

  Bool_t hasVersion() const
  {
    return mVersion >= 0;
  }

  Bool_t hasSubVersion() const
  {
    return mSubVersion >= 0;
  }

  Bool_t isSupersetOf(const ConditionId& other) const
  {
    return mPath.isSupersetOf(other.mPath) && mIdRunRange.isSupersetOf(other.mIdRunRange);
  }

  virtual Bool_t isEqual(const TObject* obj) const;

  TString ToString() const;

  void print(Option_t* option = "") const;

  Int_t Compare(const TObject* obj) const override;

  Bool_t IsSortable() const override;

  const char* GetName() const override
  {
    return mPath.getPathString().Data();
  }

 private:
  IdPath mPath;           // path
  IdRunRange mIdRunRange; // run range
  Int_t mVersion;         // version
  Int_t mSubVersion;      // subversion
  TString mLastStorage;   // previous storage place (new, grid, local, dump)

  ClassDefOverride(ConditionId, 1);
};
} // namespace ccdb
} // namespace o2
#endif
