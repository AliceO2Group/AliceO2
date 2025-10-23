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

#include "Framework/AODReaderHelpers.h"
#include "Framework/AnalysisHelpers.h"
#include "Framework/AnalysisDataModelHelpers.h"
#include "Framework/DataProcessingHelpers.h"
#include "Framework/ExpressionHelpers.h"
#include "Framework/AlgorithmSpec.h"
#include "Framework/ControlService.h"
#include "Framework/CallbackService.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/DataSpecUtils.h"

#include <Monitoring/Monitoring.h>

#include <TGrid.h>
#include <TFile.h>
#include <TTreeCache.h>

#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>
#include <arrow/io/interfaces.h>
#include <arrow/table.h>
#include <arrow/util/key_value_metadata.h>

namespace o2::framework::readers
{
auto setEOSCallback(InitContext& ic)
{
  ic.services().get<CallbackService>().set<CallbackService::Id::EndOfStream>(
    [](EndOfStreamContext& eosc) {
      auto& control = eosc.services().get<ControlService>();
      control.endOfStream();
      control.readyToQuit(QuitRequest::Me);
    });
}

template <typename... Ts>
static inline auto doExtractOriginal(framework::pack<Ts...>, ProcessingContext& pc)
{
  if constexpr (sizeof...(Ts) == 1) {
    return pc.inputs().get<TableConsumer>(aod::MetadataTrait<framework::pack_element_t<0, framework::pack<Ts...>>>::metadata::tableLabel())->asArrowTable();
  } else {
    return std::vector{pc.inputs().get<TableConsumer>(aod::MetadataTrait<Ts>::metadata::tableLabel())->asArrowTable()...};
  }
}

template <typename... Os>
static inline auto extractOriginalsTuple(framework::pack<Os...>, ProcessingContext& pc)
{
  return std::make_tuple(extractTypedOriginal<Os>(pc)...);
}

template <typename... Os>
static inline auto extractOriginalsVector(framework::pack<Os...>, ProcessingContext& pc)
{
  return std::vector{extractOriginal<Os>(pc)...};
}

template <size_t N, std::array<soa::TableRef, N> refs>
static inline auto extractOriginals(ProcessingContext& pc)
{
  return [&]<size_t... Is>(std::index_sequence<Is...>) -> std::vector<std::shared_ptr<arrow::Table>> {
    return {pc.inputs().get<TableConsumer>(o2::aod::label<refs[Is]>())->asArrowTable()...};
  }(std::make_index_sequence<refs.size()>());
}
namespace
{
template <typename D>
  requires(D::exclusive)
auto make_build(D metadata, InputSpec const& input, ProcessingContext& pc)
{
  using metadata_t = decltype(metadata);
  using Key = typename metadata_t::Key;
  using index_pack_t = typename metadata_t::index_pack_t;
  constexpr auto sources = metadata_t::sources;
  return o2::framework::IndexBuilder<o2::framework::Exclusive>::indexBuilder<Key, sources.size(), sources>(input.binding.c_str(),
                                                                                                           extractOriginals<sources.size(), sources>(pc),
                                                                                                           index_pack_t{});
}

template <typename D>
  requires(!D::exclusive)
auto make_build(D metadata, InputSpec const& input, ProcessingContext& pc)
{
  using metadata_t = decltype(metadata);
  using Key = typename metadata_t::Key;
  using index_pack_t = typename metadata_t::index_pack_t;
  constexpr auto sources = metadata_t::sources;
  return o2::framework::IndexBuilder<o2::framework::Sparse>::indexBuilder<Key, sources.size(), sources>(input.binding.c_str(),
                                                                                                        extractOriginals<sources.size(), sources>(pc),
                                                                                                        index_pack_t{});
}
} // namespace

AlgorithmSpec AODReaderHelpers::indexBuilderCallback(std::vector<InputSpec>& requested)
{
  return AlgorithmSpec::InitCallback{[requested](InitContext& /*ic*/) {
    return [requested](ProcessingContext& pc) {
      auto outputs = pc.outputs();
      // spawn tables
      for (auto& input : requested) {
        auto&& [origin, description, version] = DataSpecUtils::asConcreteDataMatcher(input);
        if (description == header::DataDescription{"MA_RN2_EX"}) {
          outputs.adopt(Output{origin, description, version}, make_build(o2::aod::Run2MatchedExclusiveMetadata{}, input, pc));
        } else if (description == header::DataDescription{"MA_RN2_SP"}) {
          outputs.adopt(Output{origin, description, version}, make_build(o2::aod::Run2MatchedSparseMetadata{}, input, pc));
        } else if (description == header::DataDescription{"MA_RN3_EX"}) {
          outputs.adopt(Output{origin, description, version}, make_build(o2::aod::Run3MatchedExclusiveMetadata{}, input, pc));
        } else if (description == header::DataDescription{"MA_RN3_SP"}) {
          outputs.adopt(Output{origin, description, version}, make_build(o2::aod::Run3MatchedSparseMetadata{}, input, pc));
        } else if (description == header::DataDescription{"MA_BCCOL_EX"}) {
          outputs.adopt(Output{origin, description, version}, make_build(o2::aod::MatchedBCCollisionsExclusiveMetadata{}, input, pc));
        } else if (description == header::DataDescription{"MA_BCCOL_SP"}) {
          outputs.adopt(Output{origin, description, version}, make_build(o2::aod::MatchedBCCollisionsSparseMetadata{}, input, pc));
        } else if (description == header::DataDescription{"MA_BCCOLS_EX"}) {
          outputs.adopt(Output{origin, description, version}, make_build(o2::aod::MatchedBCCollisionsExclusiveMultiMetadata{}, input, pc));
        } else if (description == header::DataDescription{"MA_BCCOLS_SP"}) {
          outputs.adopt(Output{origin, description, version}, make_build(o2::aod::MatchedBCCollisionsSparseMultiMetadata{}, input, pc));
        } else if (description == header::DataDescription{"MA_RN3_BC_SP"}) {
          outputs.adopt(Output{origin, description, version}, make_build(o2::aod::Run3MatchedToBCSparseMetadata{}, input, pc));
        } else if (description == header::DataDescription{"MA_RN3_BC_EX"}) {
          outputs.adopt(Output{origin, description, version}, make_build(o2::aod::Run3MatchedToBCExclusiveMetadata{}, input, pc));
        } else if (description == header::DataDescription{"MA_RN2_BC_SP"}) {
          outputs.adopt(Output{origin, description, version}, make_build(o2::aod::Run2MatchedToBCSparseMetadata{}, input, pc));
        } else {
          throw std::runtime_error("Not an index table");
        }
      }
    };
  }};
}

namespace
{
template <o2::aod::is_aod_hash D>
auto make_spawn(InputSpec const& input, ProcessingContext& pc)
{
  using metadata_t = o2::aod::MetadataTrait<D>::metadata;
  constexpr auto sources = metadata_t::sources;
  static std::shared_ptr<gandiva::Projector> projector = nullptr;
  static std::shared_ptr<arrow::Schema> schema = std::make_shared<arrow::Schema>(o2::soa::createFieldsFromColumns(typename metadata_t::expression_pack_t{}));
  static auto projectors = []<typename... C>(framework::pack<C...>) -> std::array<expressions::Projector, sizeof...(C)>
  {
    return {{std::move(C::Projector())...}};
  }
  (typename metadata_t::expression_pack_t{});
  return o2::framework::spawner<D>(extractOriginals<sources.size(), sources>(pc), input.binding.c_str(), projectors.data(), projector, schema);
}
} // namespace

AlgorithmSpec AODReaderHelpers::aodSpawnerCallback(std::vector<InputSpec>& requested)
{
  return AlgorithmSpec::InitCallback{[requested](InitContext& /*ic*/) {
    return [requested](ProcessingContext& pc) {
      auto outputs = pc.outputs();
      // spawn tables
      for (auto& input : requested) {
        auto&& [origin, description, version] = DataSpecUtils::asConcreteDataMatcher(input);
        if (description == header::DataDescription{"EXTRACK"}) {
          outputs.adopt(Output{origin, description, version}, make_spawn<o2::aod::Hash<"EXTRACK/0"_h>>(input, pc));
        } else if (description == header::DataDescription{"EXTRACK_IU"}) {
          outputs.adopt(Output{origin, description, version}, make_spawn<o2::aod::Hash<"EXTRACK_IU/0"_h>>(input, pc));
        } else if (description == header::DataDescription{"EXTRACKCOV"}) {
          outputs.adopt(Output{origin, description, version}, make_spawn<o2::aod::Hash<"EXTRACKCOV/0"_h>>(input, pc));
        } else if (description == header::DataDescription{"EXTRACKCOV_IU"}) {
          outputs.adopt(Output{origin, description, version}, make_spawn<o2::aod::Hash<"EXTRACKCOV_IU/0"_h>>(input, pc));
        } else if (description == header::DataDescription{"EXTRACKEXTRA"}) {
          if (version == 0U) {
            outputs.adopt(Output{origin, description, version}, make_spawn<o2::aod::Hash<"EXTRACKEXTRA/0"_h>>(input, pc));
          } else if (version == 1U) {
            outputs.adopt(Output{origin, description, version}, make_spawn<o2::aod::Hash<"EXTRACKEXTRA/1"_h>>(input, pc));
          } else if (version == 2U) {
            outputs.adopt(Output{origin, description, version}, make_spawn<o2::aod::Hash<"EXTRACKEXTRA/2"_h>>(input, pc));
          }
        } else if (description == header::DataDescription{"EXMFTTRACK"}) {
          if (version == 0U) {
            outputs.adopt(Output{origin, description, version}, make_spawn<o2::aod::Hash<"EXMFTTRACK/0"_h>>(input, pc));
          } else if (version == 1U) {
            outputs.adopt(Output{origin, description, version}, make_spawn<o2::aod::Hash<"EXMFTTRACK/1"_h>>(input, pc));
          }
        } else if (description == header::DataDescription{"EXMFTTRACKCOV"}) {
          outputs.adopt(Output{origin, description, version}, make_spawn<o2::aod::Hash<"EXMFTTRACKCOV/0"_h>>(input, pc));
        } else if (description == header::DataDescription{"EXFWDTRACK"}) {
          outputs.adopt(Output{origin, description, version}, make_spawn<o2::aod::Hash<"EXFWDTRACK/0"_h>>(input, pc));
        } else if (description == header::DataDescription{"EXFWDTRACKCOV"}) {
          outputs.adopt(Output{origin, description, version}, make_spawn<o2::aod::Hash<"EXFWDTRACKCOV/0"_h>>(input, pc));
        } else if (description == header::DataDescription{"EXMCPARTICLE"}) {
          if (version == 0U) {
            outputs.adopt(Output{origin, description, version}, make_spawn<o2::aod::Hash<"EXMCPARTICLE/0"_h>>(input, pc));
          } else if (version == 1U) {
            outputs.adopt(Output{origin, description, version}, make_spawn<o2::aod::Hash<"EXMCPARTICLE/1"_h>>(input, pc));
          }
        } else {
          throw runtime_error("Not an extended table");
        }
      }
    };
  }};
}

} // namespace o2::framework::readers
