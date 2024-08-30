// Copyright 2024 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <TCollection.h>
#include <TEfficiency.h>
#include <TGraph.h>
#include <TH1.h>
#include <THnBase.h>
#include <TObject.h>
#include <TTree.h>
#include "Mergers/MergeInterface.h"
#include "Mergers/Mergeable.h"

namespace o2::mergers
{

bool isMergeable(TObject* obj)
{
  return obj->InheritsFrom(mergers::MergeInterface::Class()) ||
         obj->InheritsFrom(TCollection::Class()) ||
         obj->InheritsFrom(TH1::Class()) ||
         obj->InheritsFrom(THnBase::Class()) ||
         obj->InheritsFrom(TTree::Class()) ||
         obj->InheritsFrom(TGraph::Class()) ||
         obj->InheritsFrom(TEfficiency::Class());
}

} // namespace o2::mergers
