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

#ifndef ALICEO2_MERGERS_H
#define ALICEO2_MERGERS_H

/// \file Mergeable.h
/// \brief Mergeable concept.
///
/// \author Michal Tichak, michal.tichak@cern.ch

#include <concepts>

class TObject;
class TH1;
class TCollection;
class TObjArray;
class TH1;
class TTree;
class THnBase;
class TEfficiency;
class TGraph;
class TCanvas;

namespace o2::mergers
{

class MergeInterface;

template <typename T, typename... Ts>
constexpr bool IsDerivedFrom = (std::derived_from<T, Ts> || ...);

// \brief Concept to be used to test if some parameter is mergeable
//
// \parameter Ignore can disable whole concept, if user is really sure that he wants to pass anything into this
// \parameter T type to be restricted
template <typename T>
concept Mergeable = IsDerivedFrom<std::remove_pointer_t<T>, mergers::MergeInterface, TCollection, TH1, TTree, TGraph, TEfficiency, THnBase>;

// \brief runtime check whether TObject is mergeable
bool isMergeable(TObject* obj);

} // namespace o2::mergers

#endif
