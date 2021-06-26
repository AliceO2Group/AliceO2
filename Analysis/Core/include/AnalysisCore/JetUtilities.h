// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file JetUtilities.h
/// \brief Jet related utilities
///
/// \author Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL

#ifndef O2_ANALYSIS_JETUTILITIES_H
#define O2_ANALYSIS_JETUTILITIES_H

#include <tuple>
#include <vector>

namespace JetMatching
{
/**
 * Geometrical jet matching.
 *
 * Match jets in the "base" collection with those in the "tag" collection. Jets are matched within
 * the provided matching distance. Jets are required to match uniquely - namely: base <-> tag.
 * Only one direction of matching isn't enough.
 *
 * If no unique match was found for a jet, an index of -1 is stored.
 *
 * @param jetsBasePhi Base jet collection phi.
 * @param jetsBaseEta Base jet collection eta.
 * @param jetsTagPhi Tag jet collection phi.
 * @param jetsTagEta Tag jet collection eta.
 * @param maxMatchingDistance Maximum matching distance.
 *
 * @returns (Base to tag index map, tag to base index map) for uniquely matched jets.
 */
template<typename T>
std::tuple<std::vector<int>, std::vector<int>> MatchJetsGeometrically(
  std::vector<T> jetsBasePhi,
  std::vector<T> jetsBaseEta,
  std::vector<T> jetsTagPhi,
  std::vector<T> jetsTagEta,
  double maxMatchingDistance
);

/**
 * Implementation of geometrical jet matching.
 *
 * Jets are required to match uniquely - namely: base <-> tag. Only one direction of matching isn't enough.
 * Unless special conditions are required, it's better to use `MatchJetsGeometrically`, which has an
 * easier to use interface.
 *
 * NOTE: The vectors for matching could all be const, except that SetData() doesn't take a const.
 *
 * @param jetsBasePhi Base jet collection phi.
 * @param jetsBaseEta Base jet collection eta.
 * @param jetsBasePhiForMatching Base jet collection phi to use for matching.
 * @param jetsBaseEtaForMatching Base jet collection eta to use for matching.
 * @param jetMapBaseToJetIndex Base jet collection index map from duplicated jets to original jets.
 * @param jetsTagPhi Tag jet collection phi.
 * @param jetsTagEta Tag jet collection eta.
 * @param jetsTagPhiForMatching Tag jet collection phi to use for matching.
 * @param jetsTagEtaForMatching Tag jet collection eta to use for matching.
 * @param jetMapTagToJetIndex Tag jet collection index map from duplicated jets to original jets.
 * @param maxMatchingDistance Maximum matching distance.
 *
 * @returns (Base to tag index map, tag to base index map) for uniquely matched jets.
 */
template<typename T>
std::tuple<std::vector<int>, std::vector<int>> MatchJetsGeometricallyImpl(
    const std::vector<T> & jetsBasePhi,
    const std::vector<T> & jetsBaseEta,
    std::vector<T> jetsBasePhiForMatching,
    std::vector<T> jetsBaseEtaForMatching,
    const std::vector<std::size_t> jetMapBaseToJetIndex,
    const std::vector<T> & jetsTagPhi,
    const std::vector<T> & jetsTagEta,
    std::vector<T> jetsTagPhiForMatching,
    std::vector<T> jetsTagEtaForMatching,
    const std::vector<std::size_t> jetMapTagToJetIndex,
    double maxMatchingDistance
    );

/**
 * Duplicates jets around the phi boundary which are within the matching distance.
 *
 * NOTE: Assumes, but does not validate, that 0 <= phi < 2pi.
 *
 * @param jetsPhi Jets phi
 * @param jetsEta Jets eta
 * @param maxMatchingDistance Maximum matching distance. Only duplicate jets within this distance of the boundary.
 */
template<typename T>
std::tuple<std::vector<std::size_t>, std::vector<T>, std::vector<T>> DuplicateJetsAroundPhiBoundary(
  std::vector<T> & jetsPhi,
  std::vector<T> & jetsEta,
  double maxMatchingDistance,
  // TODO: Remove additional margin after additional testing.
  double additionalMargin = 0.05
);
};

#endif
