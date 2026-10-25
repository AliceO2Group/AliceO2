// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/ASoA.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/GroupedCombinations.h"
#include "Framework/TableBuilder.h"

#include <catch_amalgamated.hpp>

#include <array>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace o2::framework;
using namespace o2::soa;

namespace o2::aod
{
namespace groupedcombinations
{
DECLARE_SOA_COLUMN(Category, category, int);
DECLARE_SOA_COLUMN(EventValue, eventValue, int);
DECLARE_SOA_COLUMN(DaughterValue, daughterValue, int);
} // namespace groupedcombinations
DECLARE_SOA_TABLE(GroupedEvents, "AOD", "GCEVENT", o2::soa::Index<>, groupedcombinations::Category, groupedcombinations::EventValue);
namespace groupedcombinations
{
DECLARE_SOA_INDEX_COLUMN_FULL_CUSTOM(Event, event, int, GroupedEvents, "GCEVENT", "");
} // namespace groupedcombinations
DECLARE_SOA_TABLE(GroupedLeftRows, "AOD", "GCLEFT", groupedcombinations::EventId, groupedcombinations::DaughterValue);
DECLARE_SOA_TABLE(GroupedRightRows, "AOD", "GCRIGHT", groupedcombinations::EventId, groupedcombinations::DaughterValue);
DECLARE_SOA_TABLE(GroupedThirdRows, "AOD", "GCTHIRD", groupedcombinations::EventId, groupedcombinations::DaughterValue);
} // namespace o2::aod

namespace
{
using namespace o2::aod::groupedcombinations;
using Events = o2::aod::GroupedEvents;
using LeftRows = o2::aod::GroupedLeftRows;
using RightRows = o2::aod::GroupedRightRows;
using ThirdRows = o2::aod::GroupedThirdRows;
using Binning = ColumnBinningPolicy<Category>;
using UpperPolicy = CombinationsBlockUpperSameIndexPolicy<Binning, int, Events, Events>;
using FullPolicy = CombinationsBlockFullSameIndexPolicy<Binning, int, Events, Events>;
using StrictPair = Pair<Events, LeftRows, RightRows, Binning>;
using UpperPair = Pair<Events, LeftRows, RightRows, Binning, int, UpperPolicy>;
using FullPair = Pair<Events, LeftRows, RightRows, Binning, int, FullPolicy>;
using StrictTriple = Triple<Events, LeftRows, RightRows, ThirdRows, Binning>;
using JoinedRows = o2::soa::Join<LeftRows, RightRows>;
using FilteredRows = o2::soa::Filtered<LeftRows>;
using ConcatenatedRows = o2::soa::Concat<LeftRows, RightRows>;

template <typename FirstAssociated>
concept CanConstructSkipPair = requires(const Binning& binning, const Events& events,
                                        const std::tuple<FirstAssociated, RightRows>& associated,
                                        SliceCache* cache) {
  Pair<Events, FirstAssociated, RightRows, Binning>{binning, 1, -1, events, associated, cache,
                                                    GroupedSlicePolicy::SkipEmpty};
};

static_assert(CanConstructSkipPair<LeftRows>);
static_assert(!CanConstructSkipPair<JoinedRows>);
static_assert(!CanConstructSkipPair<FilteredRows>);
static_assert(!CanConstructSkipPair<ConcatenatedRows>);
static_assert(!CanConstructSkipPair<ConcatenatedRows::base>);

struct Fixture {
  std::shared_ptr<arrow::Table> eventsArrow;
  std::shared_ptr<arrow::Table> leftArrow;
  std::shared_ptr<arrow::Table> rightArrow;
  std::shared_ptr<arrow::Table> thirdArrow;
  Events events;
  LeftRows left;
  RightRows right;
  ThirdRows third;
  ArrowTableSlicingCache arrowCache;
  SliceCache cache;
  Binning binning{std::array<std::vector<double>, 1>{std::vector<double>{VARIABLE_WIDTH, -0.5, 0.5, 1.5}}, true};

  Fixture(std::vector<int> leftIds = {0, 2, 3, 4},
          std::vector<int> rightIds = {1, 2, 3, 5},
          std::vector<int> thirdIds = {0, 1, 4, 5})
    : eventsArrow(makeEvents()), leftArrow(makeValues<LeftRows>(leftIds, 200)), rightArrow(makeValues<RightRows>(rightIds, 300)), thirdArrow(makeValues<ThirdRows>(thirdIds, 400)), events(eventsArrow), left(leftArrow), right(rightArrow), third(thirdArrow), arrowCache(makeCache()), cache{&arrowCache}
  {
    left.bindExternalIndices(&events);
    right.bindExternalIndices(&events);
    third.bindExternalIndices(&events);
  }

  template <typename Table>
  static std::shared_ptr<arrow::Table> makeValues(const std::vector<int>& ids, int base)
  {
    TableBuilder builder;
    auto writer = builder.cursor<Table>();
    for (const auto id : ids) {
      writer(0, id, base + id);
    }
    return builder.finalize();
  }

  static std::shared_ptr<arrow::Table> makeEvents()
  {
    TableBuilder builder;
    auto writer = builder.cursor<Events>();
    for (const auto [category, value] : std::array<std::pair<int, int>, 7>{{{0, 100}, {0, 101}, {0, 102}, {1, 103}, {1, 104}, {1, 105}, {99, 199}}}) {
      writer(0, category, value);
    }
    return builder.finalize();
  }

  ArrowTableSlicingCache makeCache()
  {
    const auto key = EventId::mLabel;
    auto leftMatcher = getMatcherFromTypeForKey<LeftRows>(key);
    auto rightMatcher = getMatcherFromTypeForKey<RightRows>(key);
    auto thirdMatcher = getMatcherFromTypeForKey<ThirdRows>(key);
    ArrowTableSlicingCache result{{{getLabelFromType<LeftRows>(), leftMatcher, key},
                                   {getLabelFromType<RightRows>(), rightMatcher, key},
                                   {getLabelFromType<ThirdRows>(), thirdMatcher, key}},
                                  {},
                                  o2::header::DataOrigin{"AOD"}};
    REQUIRE(result.updateCacheEntry(0, leftArrow).ok());
    REQUIRE(result.updateCacheEntry(1, rightArrow).ok());
    REQUIRE(result.updateCacheEntry(2, thirdArrow).ok());
    return result;
  }
};

template <typename PairType>
std::vector<std::string> readPair(PairType& pairs, bool filterEmpty)
{
  std::vector<std::string> values;
  for (auto& [leftEvent, leftRows, rightEvent, rightRows] : pairs) {
    if (filterEmpty && (leftRows.size() == 0 || rightRows.size() == 0)) {
      continue;
    }
    REQUIRE(leftRows.size() > 0);
    REQUIRE(rightRows.size() > 0);
    REQUIRE(leftRows.begin().eventId() == leftEvent.globalIndex());
    REQUIRE(rightRows.begin().eventId() == rightEvent.globalIndex());
    REQUIRE(leftRows.begin().template event_as<Events>().eventValue() == 100 + leftEvent.globalIndex());
    REQUIRE(rightRows.begin().template event_as<Events>().eventValue() == 100 + rightEvent.globalIndex());
    values.push_back(std::to_string(leftEvent.globalIndex()) + ":" + std::to_string(rightEvent.globalIndex()) + ":" +
                     std::to_string(leftRows.begin().daughterValue()) + ":" + std::to_string(rightRows.begin().daughterValue()));
  }
  return values;
}

TEST_CASE("Grouped combinations preserve default stream and skip empty slices on opt-in")
{
  Fixture fixture;
  const auto associated = std::make_tuple(fixture.left, fixture.right);
  const std::vector<std::string> expectedStrictDepth1{"0:1:200:301", "4:5:204:305"};
  const std::vector<std::string> expectedStrictDepth2{"0:1:200:301", "0:2:200:302", "3:5:203:305", "4:5:204:305"};
  const std::vector<std::string> expectedStrictDepth10 = expectedStrictDepth2;

  for (const int depth : {1, 2, 10}) {
    StrictPair baseline{fixture.binning, depth, -1, fixture.events, associated, &fixture.cache};
    const auto eligible = readPair(baseline, true);
    REQUIRE(eligible == (depth == 1 ? expectedStrictDepth1 : depth == 2 ? expectedStrictDepth2
                                                                        : expectedStrictDepth10));

    std::vector<std::string> full;
    for (auto& [first, left, second, right] : baseline) {
      full.push_back(std::to_string(first.globalIndex()) + ":" + std::to_string(second.globalIndex()) + ":" +
                     std::to_string(left.size()) + ":" + std::to_string(right.size()));
    }
    REQUIRE(full.size() == (depth == 1 ? 4 : 6));
    REQUIRE(full.front() == "0:1:1:1");
    REQUIRE(full.back() == "4:5:1:1");

    StrictPair optimized{fixture.binning, depth, -1, fixture.events, associated, &fixture.cache,
                         GroupedSlicePolicy::SkipEmpty};
    REQUIRE(readPair(optimized, false) == eligible);

    if (depth == 1) {
      auto iterator = baseline.begin();
      auto copy = iterator;
      auto previous = iterator++;
      REQUIRE(std::get<0>(*previous).globalIndex() == 0);
      REQUIRE(std::get<0>(*iterator).globalIndex() == 1);
      REQUIRE(std::get<0>(*copy).globalIndex() == 0);
      auto advancedCopy = iterator;
      REQUIRE(std::get<0>(*advancedCopy).globalIndex() == 1);

      auto optimizedIterator = optimized.begin();
      auto independent = optimizedIterator;
      auto saved = optimizedIterator++;
      REQUIRE(std::get<0>(*saved).globalIndex() == 0);
      REQUIRE(std::get<2>(*saved).globalIndex() == 1);
      REQUIRE(std::get<0>(*optimizedIterator).globalIndex() == 4);
      REQUIRE(std::get<0>(*saved).globalIndex() == 0);
      REQUIRE(std::get<0>(*independent).globalIndex() == 0);
      auto advancedOptimizedCopy = optimizedIterator;
      ++optimizedIterator;
      REQUIRE(optimizedIterator == optimized.end());
      REQUIRE(std::get<0>(*advancedOptimizedCopy).globalIndex() == 4);
      REQUIRE(std::get<1>(*advancedOptimizedCopy).begin().template event_as<Events>().eventValue() == 104);
      REQUIRE(std::get<0>(*independent).globalIndex() == 0);
      ++advancedOptimizedCopy;
      REQUIRE(advancedOptimizedCopy == optimized.end());
    }
  }

  UpperPair upper{fixture.binning, 2, -1, fixture.events, associated, &fixture.cache};
  const auto upperEligible = readPair(upper, true);
  REQUIRE(upperEligible == (std::vector<std::string>{"0:1:200:301", "0:2:200:302", "2:2:202:302", "3:3:203:303", "3:5:203:305", "4:5:204:305"}));
  UpperPair upperOptimized{fixture.binning, 2, -1, fixture.events, associated, &fixture.cache,
                           GroupedSlicePolicy::SkipEmpty};
  REQUIRE(readPair(upperOptimized, false) == upperEligible);
  for (const auto& tuple : upperEligible) {
    REQUIRE(tuple.find("6:") != 0);
    REQUIRE(tuple.find(":6:") == std::string::npos);
  }
  FullPair full{fixture.binning, 2, -1, fixture.events, associated, &fixture.cache};
  const auto fullEligible = readPair(full, true);
  REQUIRE(fullEligible == (std::vector<std::string>{"0:1:200:301", "0:2:200:302", "2:1:202:301", "2:2:202:302", "3:3:203:303", "3:5:203:305", "4:3:204:303", "4:5:204:305"}));
  FullPair fullOptimized{fixture.binning, 2, -1, fixture.events, associated, &fixture.cache,
                         GroupedSlicePolicy::SkipEmpty};
  REQUIRE(readPair(fullOptimized, false) == fullEligible);
}

TEST_CASE("Grouped triple skips if any associated slice is empty")
{
  Fixture fixture;
  const auto associated = std::make_tuple(fixture.left, fixture.right, fixture.third);
  StrictTriple baseline{fixture.binning, 10, -1, fixture.events, associated, &fixture.cache};
  std::vector<std::string> eligible;
  std::vector<std::string> full;
  for (auto& [first, left, second, right, thirdEvent, third] : baseline) {
    const auto tuple = std::to_string(first.globalIndex()) + ":" + std::to_string(second.globalIndex()) + ":" +
                       std::to_string(thirdEvent.globalIndex());
    full.push_back(tuple);
    if (left.size() && right.size() && third.size()) {
      eligible.push_back(tuple);
      REQUIRE(left.begin().eventId() == first.globalIndex());
      REQUIRE(right.begin().eventId() == second.globalIndex());
      REQUIRE(third.begin().eventId() == thirdEvent.globalIndex());
      REQUIRE(third.begin().daughterValue() == 400 + thirdEvent.globalIndex());
    }
  }
  REQUIRE(full.size() == 2);
  REQUIRE(eligible.empty());
  StrictTriple optimized{fixture.binning, 10, -1, fixture.events, associated, &fixture.cache,
                         GroupedSlicePolicy::SkipEmpty};
  size_t count = 0;
  for (auto& tuple : optimized) {
    (void)tuple;
    ++count;
  }
  REQUIRE(count == eligible.size());
}

TEST_CASE("Grouped combinations handle empty and singleton strict-upper tables")
{
  Fixture fixture;
  TableBuilder emptyBuilder;
  auto emptyWriter = emptyBuilder.cursor<Events>();
  (void)emptyWriter;
  Events empty{emptyBuilder.finalize()};
  const auto associated = std::make_tuple(fixture.left, fixture.right);
  StrictPair emptyDefault{fixture.binning, 10, -1, empty, associated, &fixture.cache};
  REQUIRE(emptyDefault.begin() == emptyDefault.end());
  StrictPair emptySkip{fixture.binning, 10, -1, empty, associated, &fixture.cache, GroupedSlicePolicy::SkipEmpty};
  REQUIRE(emptySkip.begin() == emptySkip.end());

  TableBuilder singletonBuilder;
  auto singletonWriter = singletonBuilder.cursor<Events>();
  singletonWriter(0, 0, 700);
  Events singleton{singletonBuilder.finalize()};
  auto left = fixture.left;
  auto right = fixture.right;
  left.bindExternalIndices(&singleton);
  right.bindExternalIndices(&singleton);
  const auto singletonAssociated = std::make_tuple(left, right);
  StrictPair singletonDefault{fixture.binning, 10, -1, singleton, singletonAssociated, &fixture.cache};
  REQUIRE(singletonDefault.begin() == singletonDefault.end());
  StrictPair singletonSkip{fixture.binning, 10, -1, singleton, singletonAssociated, &fixture.cache,
                           GroupedSlicePolicy::SkipEmpty};
  REQUIRE(singletonSkip.begin() == singletonSkip.end());

  StrictPair reused{fixture.binning, 2, -1, empty, associated, &fixture.cache};
  REQUIRE(reused.begin() == reused.end());
  reused.setTables(fixture.events, associated);
  REQUIRE(readPair(reused, true) == (std::vector<std::string>{"0:1:200:301", "0:2:200:302", "3:5:203:305", "4:5:204:305"}));
  StrictPair reusedSkip{fixture.binning, 2, -1, empty, associated, &fixture.cache, GroupedSlicePolicy::SkipEmpty};
  reusedSkip.setTables(fixture.events, associated);
  REQUIRE(readPair(reusedSkip, false) == readPair(reused, true));
}

template <typename TripleType>
std::vector<std::array<int64_t, 3>> readTriple(TripleType& triples, bool filterEmpty)
{
  std::vector<std::array<int64_t, 3>> values;
  for (auto& [first, left, second, right, thirdEvent, third] : triples) {
    if (filterEmpty && (left.size() == 0 || right.size() == 0 || third.size() == 0)) {
      continue;
    }
    REQUIRE(left.size() == 1);
    REQUIRE(right.size() == 1);
    REQUIRE(third.size() == 1);
    REQUIRE(left.begin().daughterValue() == 200 + first.globalIndex());
    REQUIRE(right.begin().daughterValue() == 300 + second.globalIndex());
    REQUIRE(third.begin().daughterValue() == 400 + thirdEvent.globalIndex());
    REQUIRE(left.begin().template event_as<Events>().eventValue() == first.eventValue());
    REQUIRE(right.begin().template event_as<Events>().eventValue() == second.eventValue());
    REQUIRE(third.begin().template event_as<Events>().eventValue() == thirdEvent.eventValue());
    values.push_back({first.globalIndex(), second.globalIndex(), thirdEvent.globalIndex()});
  }
  return values;
}

TEST_CASE("Grouped triples preserve policy streams and reject each empty position")
{
  using Upper = CombinationsBlockUpperSameIndexPolicy<Binning, int, Events, Events, Events>;
  using Full = CombinationsBlockFullSameIndexPolicy<Binning, int, Events, Events, Events>;
  using UpperTriple = Triple<Events, LeftRows, RightRows, ThirdRows, Binning, int, Upper>;
  using FullTriple = Triple<Events, LeftRows, RightRows, ThirdRows, Binning, int, Full>;
  const std::vector<int> populated{0, 1, 2, 3, 4, 5};
  const std::vector<std::array<int64_t, 3>> expectedAll{{0, 1, 2}, {3, 4, 5}};
  const std::vector<std::array<int64_t, 3>> expectedOne{{3, 4, 5}};
  for (int missing = -1; missing < 3; ++missing) {
    auto left = populated;
    auto right = populated;
    auto third = populated;
    if (missing == 0) {
      left.erase(left.begin());
    }
    if (missing == 1) {
      right.erase(right.begin() + 1);
    }
    if (missing == 2) {
      third.erase(third.begin() + 2);
    }
    Fixture fixture{left, right, third};
    const auto associated = std::make_tuple(fixture.left, fixture.right, fixture.third);
    StrictTriple baseline{fixture.binning, 10, -1, fixture.events, associated, &fixture.cache};
    StrictTriple optimized{fixture.binning, 10, -1, fixture.events, associated, &fixture.cache,
                           GroupedSlicePolicy::SkipEmpty};
    const auto expected = missing < 0 ? expectedAll : expectedOne;
    REQUIRE(readTriple(baseline, true) == expected);
    REQUIRE(readTriple(optimized, false) == expected);
    auto first = baseline.begin();
    auto& tuple = *first;
    REQUIRE(std::get<1>(tuple).size() == (missing == 0 ? 0 : 1));
    REQUIRE(std::get<3>(tuple).size() == (missing == 1 ? 0 : 1));
    REQUIRE(std::get<5>(tuple).size() == (missing == 2 ? 0 : 1));

    UpperTriple upper{fixture.binning, 2, -1, fixture.events, associated, &fixture.cache};
    UpperTriple upperSkip{fixture.binning, 2, -1, fixture.events, associated, &fixture.cache,
                          GroupedSlicePolicy::SkipEmpty};
    const auto upperRows = readTriple(upper, true);
    REQUIRE(readTriple(upperSkip, false) == upperRows);
    FullTriple full{fixture.binning, 2, -1, fixture.events, associated, &fixture.cache};
    FullTriple fullSkip{fixture.binning, 2, -1, fixture.events, associated, &fixture.cache,
                        GroupedSlicePolicy::SkipEmpty};
    const auto fullRows = readTriple(full, true);
    REQUIRE(readTriple(fullSkip, false) == fullRows);
    if (missing < 0) {
      REQUIRE(upperRows.size() == 20);
      REQUIRE(fullRows.size() == 54);
      REQUIRE(upperRows.front() == (std::array<int64_t, 3>{0, 0, 0}));
      REQUIRE(upperRows.back() == (std::array<int64_t, 3>{5, 5, 5}));
      REQUIRE(fullRows.front() == (std::array<int64_t, 3>{0, 0, 0}));
      REQUIRE(fullRows.back() == (std::array<int64_t, 3>{5, 5, 5}));
    }
  }
}

TEST_CASE("Grouped skip policy survives an empty associated dataframe")
{
  Fixture fixture;
  auto emptyArrow = Fixture::makeValues<LeftRows>({}, 200);
  LeftRows empty{emptyArrow};
  empty.bindExternalIndices(&fixture.events);
  REQUIRE(fixture.arrowCache.updateCacheEntry(0, emptyArrow).ok());
  StrictPair reused{fixture.binning, 2, -1, fixture.events, std::make_tuple(empty, fixture.right),
                    &fixture.cache, GroupedSlicePolicy::SkipEmpty};
  REQUIRE(reused.begin() == reused.end());
  REQUIRE(fixture.arrowCache.updateCacheEntry(0, fixture.leftArrow).ok());
  reused.setTables(fixture.events, std::make_tuple(fixture.left, fixture.right));
  REQUIRE(readPair(reused, false) == (std::vector<std::string>{"0:1:200:301", "0:2:200:302", "3:5:203:305", "4:5:204:305"}));
}

template <typename PairType>
std::vector<std::array<int64_t, 4>> readWindowState(PairType& pairs, bool filterEmpty)
{
  std::vector<std::array<int64_t, 4>> states;
  for (auto iterator = pairs.begin(); iterator != pairs.end(); ++iterator) {
    auto& [first, left, second, right] = *iterator;
    if (filterEmpty && (left.size() == 0 || right.size() == 0)) {
      continue;
    }
    states.push_back({first.globalIndex(), second.globalIndex(), iterator.isNewWindow(), iterator.currentWindowNeighbours()});
  }
  return states;
}

TEST_CASE("Grouped skip retains original window metadata")
{
  Fixture fixture{{0}, {2}, {0, 1, 2}};
  const auto associated = std::make_tuple(fixture.left, fixture.right);
  StrictPair baseline{fixture.binning, 2, -1, fixture.events, associated, &fixture.cache};
  StrictPair optimized{fixture.binning, 2, -1, fixture.events, associated, &fixture.cache,
                       GroupedSlicePolicy::SkipEmpty};
  const std::vector<std::array<int64_t, 4>> expected{{0, 2, 0, 2}};
  REQUIRE(readWindowState(baseline, true) == expected);
  REQUIRE(readWindowState(optimized, false) == expected);
}

TEST_CASE("Grouped skip uses original indices for filtered grouping rows")
{
  Fixture fixture;
  const auto associated = std::make_tuple(fixture.left, fixture.right);
  using SelectedEvents = o2::soa::Filtered<Events>;
  using SelectedPair = Pair<SelectedEvents, LeftRows, RightRows, Binning>;
  SelectedEvents selected{{fixture.events.asArrowTableRef()}, SelectionVector{0, 2, 3, 5}};
  SelectedPair selectedDefault{fixture.binning, 10, -1, selected, associated, &fixture.cache};
  SelectedPair selectedSkip{fixture.binning, 10, -1, selected, associated, &fixture.cache,
                            GroupedSlicePolicy::SkipEmpty};
  const std::vector<std::string> expectedSelected{"0:2:200:302", "3:5:203:305"};
  REQUIRE(readPair(selectedDefault, true) == expectedSelected);
  REQUIRE(readPair(selectedSkip, false) == expectedSelected);
}
} // namespace
