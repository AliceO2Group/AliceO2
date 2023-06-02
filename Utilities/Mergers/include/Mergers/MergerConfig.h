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

#ifndef ALICEO2_MERGERCONFIG_H
#define ALICEO2_MERGERCONFIG_H

/// \file MergerConfig.h
/// \brief Definition of O2 MergerConfig, v0.1
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include <string>
#include <vector>
#include <variant>

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
  // when InputObjectsTimespan::FullHistory is set.
  LastDifference,
  // Generalisation of the two above. Resets all objects in Mergers after n cycles (0 - infinite).
  // The the above will be removed once we switch to NCycles in QC.
  NCycles
};

enum class PublicationDecision {
  EachNSeconds, // Merged object is published each N seconds. This can evolve over time, thus we expect pairs specifying N:duration1, M:duration2...
};

enum class TopologySize {
  NumberOfLayers,  // User specifies the number of layers in topology.
  ReductionFactor, // User specifies how many sources should be handled by one merger (by maximum).
  MergersPerLayer  // User specifies how many Mergers should be spawned in each layer.
};

enum class ParallelismType {
  SplitInputs, // Splits the provided vector of InputSpecs evenly among Mergers.
  RoundRobin   // Mergers receive their input messages in round robin order. Useful when there is one InputSpec with a wildcard.
};

template <typename V, typename P = double>
struct ConfigEntry {
  V value;
  P param = P();
};

/**
 * This class just serves the purpose of allowing for both the old and the new way of specifying the
 * cycles duration.
 */
class PublicationDecisionParameter
{
 public:
  PublicationDecisionParameter(size_t param) : decision({{param, 1}}) {}
  PublicationDecisionParameter(const std::vector<std::pair<size_t, size_t>>& decision) : decision(decision) {}

  std::vector<std::pair<size_t, size_t>> decision;
};

// todo rework configuration in a way that user cannot create an invalid configuration
// \brief MergerAlgorithm configuration structure. Default configuration should work in most cases, out of the box.
struct MergerConfig {
  ConfigEntry<InputObjectsTimespan> inputObjectTimespan = {InputObjectsTimespan::FullHistory};
  ConfigEntry<MergedObjectTimespan, int> mergedObjectTimespan = {MergedObjectTimespan::FullHistory};
  ConfigEntry<PublicationDecision, PublicationDecisionParameter> publicationDecision = {PublicationDecision::EachNSeconds, {10}};
  ConfigEntry<TopologySize, std::variant<int, std::vector<size_t>>> topologySize = {TopologySize::NumberOfLayers, 1};
  std::string monitoringUrl = "infologger:///debug?qc";
  std::string detectorName = "TST";
  ConfigEntry<ParallelismType> parallelismType = {ParallelismType::SplitInputs};
  bool expendable = false;
};

} // namespace o2::mergers

#endif //ALICEO2_MERGERCONFIG_H
