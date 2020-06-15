// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_MERGEINTERFACEOVERRIDEEXAMPLE_H
#define ALICEO2_MERGEINTERFACEOVERRIDEEXAMPLE_H

/// \file CustomMergeableObject.h
/// \brief An example of overriding O2 Mergers merging interface, v0.1
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "Mergers/MergeInterface.h"

namespace o2::mergers
{

class CustomMergeableObject : public MergeInterface
{
 public:
  CustomMergeableObject(int secret = 9000) : MergeInterface(), mSecret(secret) {}
  ~CustomMergeableObject() override = default;

  void merge(MergeInterface* const other) override
  {
    mSecret += dynamic_cast<const CustomMergeableObject* const>(other)->getSecret();
  }

  int getSecret() const { return mSecret; }

 private:
  int mSecret = 0;

  ClassDefOverride(CustomMergeableObject, 1);
};

} // namespace o2::mergers

#endif //ALICEO2_MERGEINTERFACEOVERRIDEEXAMPLE_H
