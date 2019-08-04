// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_MERGERCONFIG_H
#define ALICEO2_MERGERCONFIG_H

/// \file MergerConfig.h
/// \brief Definition of O2 MergerConfig, v0.1
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

namespace o2
{
namespace experimental::mergers
{

// This is a set of Mergers' options that user can choose from.

enum class OwnershipMode { // todo better name
  Full,                    // Mergers expect full objects each time.
  Integral                 // Mergers expect objects' differences (what has changed since the previous was sent).
};

enum class MergingMode {
  Binwise,     // Bins of histograms are added, TTree branches are attached, objects inside TCollections are merged correspondingly.
  Timewise,    // NOT SUPPORTED YET. Arriving objects are merged into one T*****, ordered in time.
  Concatenate, // Arriving objects are merged into one TObjArray, no particular order.
};

enum class MergingTime {
  AfterArrival,      // Merging just after new object arrives.
  WhenXInputsCached, // Merging when 0 < X < 1 of the inputs are cached.
                     //  EachNSeconds, // NOT SUPPORTED. Merging the most up-to-date inputs each N seconds //todo: maybe later (dpl makes it hard to have two timers)
  BeforePublication  // Merging just before publication.
};

enum class Timespan {
  FullHistory,    // Merged object should consist of all the partial data that Mergers received..
  LastDifference, // Merged object should consist of only data received after last publication. Merged object is reset after published.
  //  MovingWindow // NOT SUPPORTED YET. Merged object should consist of object from last N seconds. Inherit MergeInterface and implement time().
};

enum class PublicationDecision {
  EachNSeconds,       // Merged object is published each N seconds.
  WhenXInputsUpdated, // Merged object is published when 0 < X < 1 of the inputs are updated since previous publication.
  // AfterEachMerge // NOT SUPPORTED. todo
};

enum class TopologySize {
  NumberOfLayers, // User specifies the number of layers in topology.
  ReductionFactor // User specifies how many sources should be handled by one merger (by maximum).
};

enum class UnpackingMethod {
  NoUnpackingNeeded, // Merger treats object as it is.
  TCollection,       // NOT SUPPORTED YET. Merger treats each object as TCollection and merges each member accordingly. todo
};

template <typename V, typename P = double>
struct ConfigEntry {
  V value;
  P param;
};

// \brief Merger configuration structure. Default configuration should work in most cases, out of the box.
struct MergerConfig {
  ConfigEntry<OwnershipMode> ownershipMode = {OwnershipMode::Full};
  ConfigEntry<MergingMode> mergingMode = {MergingMode::Binwise};
  ConfigEntry<MergingTime> mergingTime = {MergingTime::BeforePublication};
  ConfigEntry<Timespan> timespan = {Timespan::FullHistory};
  ConfigEntry<PublicationDecision> publicationDecision = {PublicationDecision::WhenXInputsUpdated, 0.999999};
  ConfigEntry<TopologySize, int> topologySize = {TopologySize::NumberOfLayers, 1};
  ConfigEntry<UnpackingMethod> unpackingMethod = {UnpackingMethod::NoUnpackingNeeded};
};

} // namespace experimental::mergers
} // namespace o2

#endif //ALICEO2_MERGERCONFIG_H
