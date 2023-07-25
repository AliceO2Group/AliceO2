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

/// \file CustomMergeableTObject.h
/// \brief An example of a custom TObject inheriting MergeInterface
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#ifndef O2_CUSTOMMERGEABLETOBJECT_H
#define O2_CUSTOMMERGEABLETOBJECT_H

#include <TObject.h>
#include "Mergers/MergeInterface.h"

namespace o2::mergers
{

class CustomMergeableTObject : public TObject, public MergeInterface
{
 public:
  CustomMergeableTObject() = default;
  CustomMergeableTObject(std::string name, int secret = 9000)
    : TObject(), MergeInterface(), mSecret(secret), mName(name)
  {
  }

  ~CustomMergeableTObject() override = default;

  void merge(MergeInterface* const other) override
  {
    mSecret += dynamic_cast<const CustomMergeableTObject* const>(other)->getSecret();
  }

  int getSecret() const
  {
    return mSecret;
  }

  const char* GetName() const override
  {
    return mName.c_str();
  }

  MergeInterface* cloneMovingWindow() const override
  {
    return new CustomMergeableTObject{mName + "_mw", mSecret};
  }

 private:
  int mSecret = 0;
  std::string mName;

  ClassDefOverride(CustomMergeableTObject, 1);
};

} // namespace o2::mergers

#endif //O2_CUSTOMMERGEABLETOBJECT_H
