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

/// \file MergerAlgorithm.cxx
/// \brief Implementation of O2 Mergers, v0.1
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "Mergers/MergerAlgorithm.h"

#include "Mergers/MergeInterface.h"
#include "Mergers/ObjectStore.h"
#include "Framework/Logger.h"

#include <TEfficiency.h>
#include <TGraph.h>
#include <TH1.h>
#include <TH2.h>
#include <TH3.h>
#include <THn.h>
#include <THnSparse.h>
#include <TObjArray.h>
#include <TObject.h>
#include <TTree.h>
#include <TPad.h>
#include <TCanvas.h>
#include <algorithm>
#include <stdexcept>

namespace o2::mergers::algorithm
{

size_t estimateTreeSize(TTree* tree)
{
  size_t totalSize = 0;
  auto branchList = tree->GetListOfBranches();
  for (const auto* branch : *branchList) {
    totalSize += dynamic_cast<const TBranch*>(branch)->GetTotalSize();
  }
  return totalSize;
}

// Mergeable objects are kept as primitives in TCanvas object in underlying TPad.
// TPad is a linked list of primitives of any type (https://root.cern.ch/doc/master/classTPad.html)
// including other TPads. So in order to collect all mergeable objects from TCanvas
// we need to recursively transverse whole TPad structure.
auto collectUnderlyingObjects(TCanvas* canvas) -> std::vector<TObject*>
{
  auto collectFromTPad = [](TPad* pad, std::vector<TObject*>& objects, const auto& collectFromTPad) {
    if (!pad) {
      return;
    }
    auto* primitives = pad->GetListOfPrimitives();
    for (int i = 0; i < primitives->GetSize(); ++i) {
      auto* primitive = primitives->At(i);
      if (auto* primitivePad = dynamic_cast<TPad*>(primitive)) {
        collectFromTPad(primitivePad, objects, collectFromTPad);
      } else {
        objects.push_back(primitive);
      }
    }
  };

  std::vector<TObject*> collectedObjects;
  collectFromTPad(canvas, collectedObjects, collectFromTPad);

  return collectedObjects;
}

struct MatchedCollectedObjects {
  MatchedCollectedObjects(TObject* t, TObject* o) : target(t), other(o) {}

  TObject* target;
  TObject* other;
};

auto matchCollectedToPairs(const std::vector<TObject*>& targetObjects, const std::vector<TObject*> otherObjects) -> std::vector<MatchedCollectedObjects>
{
  std::vector<MatchedCollectedObjects> matchedObjects;
  matchedObjects.reserve(std::max(targetObjects.size(), otherObjects.size()));
  for (const auto& targetObject : targetObjects) {
    if (const auto found_it = std::ranges::find_if(otherObjects, [&targetObject](TObject* obj) { return std::string_view(targetObject->GetName()) == std::string_view(obj->GetName()); });
        found_it != otherObjects.end()) {
      matchedObjects.emplace_back(targetObject, *found_it);
    }
  }
  return matchedObjects;
}

// calls the default Merge methods of TObjects
Long64_t mergeDefault(TObject* const target, TObject* const other)
{
  Long64_t errorCode = 0;

  TObjArray otherCollection;
  otherCollection.SetOwner(false);
  otherCollection.Add(other);

  if (target->InheritsFrom(TH1::Class())) {
    // this includes TH1, TH2, TH3
    auto targetTH1 = reinterpret_cast<TH1*>(target);
    if (targetTH1->TestBit(TH1::kIsAverage)) {
      // Merge() does not support averages, we have to use Add()
      // this will break if collection.size != 1
      if (auto otherTH1 = dynamic_cast<TH1*>(otherCollection.First())) {
        errorCode = targetTH1->Add(otherTH1) == kFALSE ? -1 : 0;
      }
    } else {
      // Add() does not support histograms with labels, thus we resort to Merge() by default
      errorCode = targetTH1->Merge(&otherCollection);
    }
  } else if (target->InheritsFrom(THnBase::Class())) {
    // this includes THn and THnSparse
    errorCode = reinterpret_cast<THnBase*>(target)->Merge(&otherCollection);
  } else if (target->InheritsFrom(TTree::Class())) {
    auto targetTree = reinterpret_cast<TTree*>(target);
    auto otherTree = reinterpret_cast<TTree*>(other);
    auto targetTreeSize = estimateTreeSize(targetTree);
    auto otherTreeSize = estimateTreeSize(otherTree);
    if (auto totalSize = targetTreeSize + otherTreeSize; totalSize > 100000000) {
      LOG(warn) << "The tree '" << targetTree->GetName() << "' would be larger than 100MB (" << totalSize << "B) after merging, skipping to let the system survive";
      errorCode = 0;
    } else {
      errorCode = targetTree->Merge(&otherCollection);
    }
  } else if (target->InheritsFrom(TGraph::Class())) {
    errorCode = reinterpret_cast<TGraph*>(target)->Merge(&otherCollection);
  } else if (target->InheritsFrom(TEfficiency::Class())) {
    errorCode = reinterpret_cast<TEfficiency*>(target)->Merge(&otherCollection);
  } else {
    LOG(warn) << "Object '" + std::string(target->GetName()) + "' with type '" + std::string(target->ClassName()) + "' is not one of the mergeable types, skipping";
  }
  return errorCode;
}

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
  } else if (auto targetCanvas = dynamic_cast<TCanvas*>(target)) {

    auto otherCanvas = dynamic_cast<TCanvas*>(other);
    if (otherCanvas == nullptr) {
      throw std::runtime_error(std::string("The target object '") + target->GetName() +
                               "' is a TCanvas, while the other object '" + other->GetName() + "' is not.");
    }

    const auto targetObjects = collectUnderlyingObjects(targetCanvas);
    const auto otherObjects = collectUnderlyingObjects(otherCanvas);
    if (targetObjects.size() != otherObjects.size()) {
      throw std::runtime_error(std::string("Trying to merge canvas: ") + targetCanvas->GetName() + " and canvas " + otherObjects.size() + "but contents are not the same");
    }

    const auto matched = matchCollectedToPairs(targetObjects, otherObjects);
    if (targetObjects.size() != matched.size()) {
      throw std::runtime_error(std::string("Trying to merge canvas: ") + targetCanvas->GetName() + " and canvas " + otherObjects.size() + "but contents are not the same");
    }

    for (const auto& [targetObject, otherObject] : matched) {
      merge(targetObject, otherObject);
    }

  } else {
    Long64_t errorCode = mergeDefault(target, other);

    if (errorCode == -1) {
      LOG(error) << "Failed to merge the input object '" + std::string(other->GetName()) + "' of type '" + std::string(other->ClassName()) //
                      + " and the target object '" + std::string(target->GetName()) + "' of type '" + std::string(target->ClassName()) + "'";

      // we retry with debug options enabled in ROOT in hopes to get some logs explaining the issue
      gDebug = true;
      errorCode = mergeDefault(target, other);
      gDebug = false;
      if (errorCode == -1) {
        LOG(error) << "Merging '" + std::string(other->GetName()) + "' and '" + std::string(target->GetName()) //
                        + "' failed again after a retry for debugging purposes. See ROOT warnings for details.";
      } else {
        LOG(warn) << "Merging '" + std::string(other->GetName()) + "' and '" + std::string(target->GetName()) //
                       + "' succeeded after retrying for debugging purposes.";
      }
    }
  }
}

void merge(VectorOfTObjectPtrs& targets, const VectorOfTObjectPtrs& others)
{
  for (const auto& other : others) {
    if (const auto targetSameName = std::find_if(targets.begin(), targets.end(), [&other](const auto& target) {
          return std::string_view{other->GetName()} == std::string_view{target->GetName()};
        });
        targetSameName != targets.end()) {
      merge(targetSameName->get(), other.get());
    } else {
      targets.push_back(std::shared_ptr<TObject>(other->Clone(), deleteTCollections));
    }
  }
}

void deleteRecursive(TCollection* Coll)
{
  // I can iterate a collection
  Coll->SetOwner(false);
  auto ITelem = Coll->MakeIterator();
  while (auto* element = ITelem->Next()) {
    if (auto* Coll2 = dynamic_cast<TCollection*>(element)) {
      Coll2->SetOwner(false);
      deleteRecursive(Coll2);
    }
    Coll->Remove(element); // Remove from mother collection
    delete element;        // Delete payload
  }
  delete ITelem;
}

void deleteTCollections(TObject* obj)
{
  if (auto* L = dynamic_cast<TCollection*>(obj)) {
    deleteRecursive(L);
    delete L;
  } else {
    delete obj;
  }
}

} // namespace o2::mergers::algorithm
