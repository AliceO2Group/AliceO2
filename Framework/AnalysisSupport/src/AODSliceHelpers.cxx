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

#include "AODSliceHelpers.h"

#include "Framework/ArrowTableSlicingCache.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DanglingEdgesContext.h"
#include "Framework/DataAllocator.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/InputRecord.h"
#include "Framework/TableConsumer.h"

namespace o2::framework::helpers
{
namespace
{
Entry sourceEntry(InputSpec const& spec)
{
  auto source = DataSpecUtils::fromMetadataString(std::ranges::find_if(spec.metadata, [](ConfigParamSpec const& cps) { return cps.name.starts_with("slice-source"); })->defaultValue.get<std::string>());
  return {source.binding, DataSpecUtils::asConcreteDataMatcher(source), std::ranges::find_if(spec.metadata, [](ConfigParamSpec const& cps) { return cps.name.starts_with("slice-key"); })->defaultValue.get<std::string>()};
}

struct Sliceable {
  Entry entry;
  ConcreteDataMatcher output;
  bool sorted;

  explicit Sliceable(InputSpec const& spec)
    : entry{sourceEntry(spec)},
      output{DataSpecUtils::asConcreteDataMatcher(spec)},
      sorted{std::ranges::find_if(spec.metadata, [](ConfigParamSpec const& cps) { return cps.name.starts_with("sorted"); })->defaultValue.get<bool>()}
  {
  }

  std::shared_ptr<arrow::Table> materialize(ProcessingContext& pc) const
  {
    auto source = pc.inputs().get<TableConsumer>(entry.matcher)->asArrowTable();
    return sorted ? SliceInfo::makeSorted(entry, source) : SliceInfo::makeUnsorted(entry, source);
  }
};
} // namespace

AlgorithmSpec AODSliceHelpers::arrowTablesSlicerCallback(ConfigContext const& /*ctx*/)
{
  return AlgorithmSpec::InitCallback{[](InitContext& ic) {
    // each slicer handles the group of slice infos for the tables from a single provider
    auto const& requested = ic.services().get<DanglingEdgesContext>().slicerGroups[ic.options().get<int>("slicer-group")];
    std::vector<Sliceable> sliceables;
    sliceables.reserve(requested.size());
    std::ranges::transform(requested, std::back_inserter(sliceables), [](auto const& i) { return Sliceable{i}; });
    return [sliceables](ProcessingContext& pc) {
      auto outputs = pc.outputs();
      std::ranges::for_each(sliceables, [&pc, &outputs](auto const& sliceable) { outputs.adopt(Output{sliceable.output.origin, sliceable.output.description, sliceable.output.subSpec}, sliceable.materialize(pc)); });
    };
  }};
}
} // namespace o2::framework::helpers
