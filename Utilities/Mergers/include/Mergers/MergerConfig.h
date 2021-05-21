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

#include <string>

namespace o2::mergers
{

// This is a set of Mergers' options that user can choose from.

enum class InputObjectsTimespan {
  FullHistory,   // Mergers expect objects with all data accumulated so far each time.
  LastDifference // Mergers expect objects' differences (what has changed since the previous were sent).
};

enum class MergedObjectTimespan {
  // Merged object should be an sum of differences received since the beginning
  // or a sum of latest versions of objects received on each input.
  FullHistory,
  // Merged object should be an sum of differences received after last publication.
  // Merged object is reset after published. It won't produce meaningful results
  // when InputObjectsTimespan::FullHstory is set.
  LastDifference
};

enum class PublicationDecision {
  EachNSeconds,       // Merged object is published each N seconds.
};

enum class TopologySize {
  NumberOfLayers, // User specifies the number of layers in topology.
  ReductionFactor // User specifies how many sources should be handled by one merger (by maximum).
};

template <typename V, typename P = double>
struct ConfigEntry {
  V value;
  P param = P();
};

// \brief MergerAlgorithm configuration structure. Default configuration should work in most cases, out of the box.
struct MergerConfig {
  ConfigEntry<InputObjectsTimespan> inputObjectTimespan = {InputObjectsTimespan::FullHistory};
  ConfigEntry<MergedObjectTimespan> mergedObjectTimespan = {MergedObjectTimespan::FullHistory};
  ConfigEntry<PublicationDecision> publicationDecision = {PublicationDecision::EachNSeconds, 10};
  ConfigEntry<TopologySize, int> topologySize = {TopologySize::NumberOfLayers, 1};
  std::string monitoringUrl = "infologger:///debug?qc";
};

} // namespace o2::mergers

#endif //ALICEO2_MERGERCONFIG_H
