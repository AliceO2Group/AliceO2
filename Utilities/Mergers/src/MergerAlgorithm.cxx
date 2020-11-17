// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Merger.cxx
/// \brief Implementation of O2 Mergers, v0.1
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "Mergers/MergerAlgorithm.h"

#include "Mergers/MergeInterface.h"

#include <TH1.h>
#include <TH2.h>
#include <TH3.h>
#include <THn.h>
#include <TTree.h>
#include <THnSparse.h>
#include <TObjArray.h>
#include <TGraph.h>

namespace o2::mergers::algorithm
{

void merge(TObject* const target, TObject* const other)
{
  if (target == nullptr) {
    throw std::runtime_error("Merging target is nullptr");
  }
  if (other == nullptr) {
    throw std::runtime_error("Object to be merged in is nullptr");
  }
  if (other == target) {
    throw std::runtime_error("Merging target and the other object point to the same address");
  }
  // fixme: should we check if names match?

  // We expect that both objects follow the same structure, but we allow to add missing objects to TCollections.
  // First we check if an object contains a MergeInterface, as it should overlap default Merge() methods of TObject.
  if (auto custom = dynamic_cast<MergeInterface*>(target)) {

    custom->merge(dynamic_cast<MergeInterface* const>(other));

  } else if (auto targetCollection = dynamic_cast<TCollection*>(target)) {

    auto otherCollection = dynamic_cast<TCollection*>(other);
    if (otherCollection == nullptr) {
      throw std::runtime_error(std::string("The target object '") + target->GetName() +
                               "' is a TCollection, while the other object '" + other->GetName() + "' is not.");
    }

    auto otherIterator = otherCollection->MakeIterator();
    while (auto otherObject = otherIterator->Next()) {
      TObject* targetObject = targetCollection->FindObject(otherObject->GetName());
      if (targetObject) {
        // That might be another collection or a concrete object to be merged, we walk on the collection recursively.
        merge(targetObject, otherObject);
      } else {
        // We prefer to clone instead of passing the pointer in order to simplify deleting the `other`.
        targetCollection->Add(otherObject->Clone());
      }
    }
    delete otherIterator;
  } else {
    Long64_t errorCode = 0;
    TObjArray otherCollection;
    otherCollection.SetOwner(false);
    otherCollection.Add(other);

    if (target->InheritsFrom(TH1::Class())) {
      // this includes TH1, TH2, TH3
      errorCode = reinterpret_cast<TH1*>(target)->Merge(&otherCollection);
    } else if (target->InheritsFrom(THnBase::Class())) {
      // this includes THn and THnSparse
      errorCode = reinterpret_cast<THnBase*>(target)->Merge(&otherCollection);
    } else if (target->InheritsFrom(TTree::Class())) {
      errorCode = reinterpret_cast<TTree*>(target)->Merge(&otherCollection);
    } else if (target->InheritsFrom(TGraph::Class())) {
      errorCode = reinterpret_cast<TGraph*>(target)->Merge(&otherCollection);
    } else {
      throw std::runtime_error("Object with type '" + std::string(target->ClassName()) + "' is not one of the mergeable types.");
    }
    if (errorCode == -1) {
      throw std::runtime_error("Merging object of type '" + std::string(target->ClassName()) + "' failed.");
    }
  }
}

void deleteTCollections(TObject* obj)
{
  if (auto c = dynamic_cast<TCollection*>(obj)) {
    c->SetOwner(false);
    auto iter = c->MakeIterator();
    while (auto element = iter->Next()) {
      deleteTCollections(element);
    }
    delete iter;
    delete c;
  } else {
    delete obj;
  }
}

} // namespace o2::mergers::algorithm
