// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALI_META_DATA_H
#define ALI_META_DATA_H

#include <TMap.h>    // for TMap
#include <TObject.h> // for TObject
#include "Rtypes.h"  // for UInt_t, ConditionMetaData::Class, Bool_t, etc
#include "TString.h" // for TString

namespace o2
{
namespace ccdb
{
//  Set of data describing the object  				   //
//  but not used to identify the object 			   //

class ConditionMetaData : public TObject
{

 public:
  ConditionMetaData();

  ConditionMetaData(const char* responsible, UInt_t beamPeriod = 0, const char* alirootVersion = "",
                    const char* comment = "");

  ~ConditionMetaData() override;

  void setObjectClassName(const char* name)
  {
    mObjectClassName = name;
  };

  const char* getObjectClassName() const
  {
    return mObjectClassName.Data();
  };

  void setResponsible(const char* yourName)
  {
    mResponsible = yourName;
  };

  const char* getResponsible() const
  {
    return mResponsible.Data();
  };

  void setBeamPeriod(UInt_t period)
  {
    mBeamPeriod = period;
  };

  UInt_t getBeamPeriod() const
  {
    return mBeamPeriod;
  };

  void setAliRootVersion(const char* version)
  {
    mAliRootVersion = version;
  };

  const char* getAliRootVersion() const
  {
    return mAliRootVersion.Data();
  };

  void setComment(const char* comment)
  {
    mComment = comment;
  };

  const char* getComment() const
  {
    return mComment.Data();
  };

  void addDateToComment();

  void setProperty(const char* property, TObject* object);

  TObject* getProperty(const char* property) const;

  Bool_t removeProperty(const char* property);

  void printConditionMetaData();

 private:
  TString mObjectClassName; // object's class name
  TString mResponsible;     // object's responsible person
  UInt_t mBeamPeriod;       // beam period
  TString mAliRootVersion;  // AliRoot version
  TString mComment;         // extra comments
  // TList mCalibRuns;

  TMap mProperties; // list of object specific properties

  ClassDefOverride(ConditionMetaData, 1);
};
} // namespace ccdb
} // namespace o2
#endif
