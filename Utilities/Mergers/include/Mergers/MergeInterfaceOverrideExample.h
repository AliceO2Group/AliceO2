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

/// \file MergeInterfaceOverrideExample.h
/// \brief An example of overriding O2 Mergers merging interface, v0.1
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "Mergers/MergeInterface.h"

namespace o2
{
namespace experimental::mergers
{

class MergeInterfaceOverrideExample : public TObject, public MergeInterface
{
 public:
  MergeInterfaceOverrideExample(int secret = 9000) : TObject(), MergeInterface(), mSecret(secret) {}
  ~MergeInterfaceOverrideExample() override = default;

  std::vector<TObject*> unpack() override
  {
    return {this};
  }

  Long64_t merge(TCollection* list) override
  {
    auto iter = list->MakeIterator();
    while (auto element = iter->Next()) {
      mSecret += reinterpret_cast<MergeInterfaceOverrideExample*>(element)->getSecret();
    }
    return 0;
  }

  double getTimestamp() override
  {
    return 0;
  };

  int getSecret() const { return mSecret; }

 private:
  int mSecret = 0;

  ClassDefOverride(MergeInterfaceOverrideExample, 1);
};

} // namespace experimental::mergers
} // namespace o2

#endif //ALICEO2_MERGEINTERFACEOVERRIDEEXAMPLE_H
