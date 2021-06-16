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

#ifndef O2_CalPadWrapper_H
#define O2_CalPadWrapper_H

#include <TClass.h>
#include <TObject.h>

#include "TPCBase/CalDet.h"

namespace o2
{
namespace tpc
{
namespace qc
{
/// Temporary solution until objects not inheriting from TObject can be handled in QualityControl
/// A wrapper class to easily promote CalDet<float> objects to a TObject
/// does not take ownership of wrapped object and should not be used
/// in tight loops since construction expensive
class CalPadWrapper : public TObject
{
 public:
  CalPadWrapper(o2::tpc::CalDet<float>* obj) : mObj(obj), TObject()
  {
  }

  CalPadWrapper() = default;

  void setObj(o2::tpc::CalDet<float>* obj)
  {
    mObj = obj;
  }

  o2::tpc::CalDet<float>* getObj()
  {
    return mObj;
  }

  virtual const char* GetName() const override { return mObj ? mObj->getName().data() : "unset"; }

  virtual ~CalPadWrapper() override = default;

 private:
  o2::tpc::CalDet<float>* mObj{}; ///< wrapped CalDet<float> (aka CalPad)

  ClassDefOverride(CalPadWrapper, 1);
};
} // namespace qc
} // namespace tpc
} // namespace o2

#endif
