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
#include "Framework/TableTreeHelpers.h"
#include "Framework/AnalysisHelpers.h"
#include "AnalysisDataModelHelpers.h"
#include "Framework/DataProcessingHelpers.h"
#include "Framework/ExpressionHelpers.h"
#include "Framework/RootTableBuilderHelpers.h"
#include "Framework/AlgorithmSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/CallbackService.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/DeviceSpec.h"
#include "Framework/RawDeviceService.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/DataInputDirector.h"
#include "Framework/SourceInfoHeader.h"
#include "Framework/ChannelInfo.h"
#include "Framework/Logger.h"

#include <Monitoring/Monitoring.h>

#include <TGrid.h>
#include <TFile.h>
#include <TTreeCache.h>
#include <TTreePerfStats.h>

#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>
#include <arrow/io/interfaces.h>
#include <arrow/table.h>
#include <arrow/util/key_value_metadata.h>

#include <thread>

using o2::monitoring::Metric;
using o2::monitoring::Monitoring;
using o2::monitoring::tags::Key;
using o2::monitoring::tags::Value;

namespace o2::framework::readers
{
auto setEOSCallback(InitContext& ic)
{
  ic.services().get<CallbackService>().set(CallbackService::Id::EndOfStream,
                                           [](EndOfStreamContext& eosc) {
                                             auto& control = eosc.services().get<ControlService>();
                                             control.endOfStream();
                                             control.readyToQuit(QuitRequest::Me);
                                           });
}

template <typename... Ts>
static inline auto doExtractTypedOriginal(framework::pack<Ts...>, ProcessingContext& pc)
{
  if constexpr (sizeof...(Ts) == 1) {
    return pc.inputs().get<TableConsumer>(aod::MetadataTrait<framework::pack_element_t<0, framework::pack<Ts...>>>::metadata::tableLabel())->asArrowTable();
  } else {
    return std::vector{pc.inputs().get<TableConsumer>(aod::MetadataTrait<Ts>::metadata::tableLabel())->asArrowTable()...};
  }
}

template <typename O>
static inline auto extractTypedOriginal(ProcessingContext& pc)
{
  return O{doExtractTypedOriginal(soa::make_originals_from_type<O>(), pc)};
}

template <typename... Os>
static inline auto extractOriginalsTuple(framework::pack<Os...>, ProcessingContext& pc)
{
  return std::make_tuple(extractTypedOriginal<Os>(pc)...);
}

AlgorithmSpec AODReaderHelpers::indexBuilderCallback(std::vector<InputSpec>& requested)
{
  return AlgorithmSpec::InitCallback{[requested](InitContext& ic) {
    return [requested](ProcessingContext& pc) {
      auto outputs = pc.outputs();
      // spawn tables
      for (auto& input : requested) {
        auto&& [origin, description, version] = DataSpecUtils::asConcreteDataMatcher(input);
        auto maker = [&](auto metadata) {
          using metadata_t = decltype(metadata);
          using Key = typename metadata_t::Key;
          using index_pack_t = typename metadata_t::index_pack_t;
          using originals = typename metadata_t::originals;
          if constexpr (metadata_t::exclusive == true) {
            return o2::framework::IndexExclusive::indexBuilder(input.binding.c_str(), index_pack_t{},
                                                               extractTypedOriginal<Key>(pc),
                                                               extractOriginalsTuple(originals{}, pc));
          } else {
            return o2::framework::IndexSparse::indexBuilder(input.binding.c_str(), index_pack_t{},
                                                            extractTypedOriginal<Key>(pc),
                                                            extractOriginalsTuple(originals{}, pc));
          }
        };

        if (description == header::DataDescription{"MA_RN2_EX"}) {
          outputs.adopt(Output{origin, description, version}, maker(o2::aod::Run2MatchedExclusiveMetadata{}));
        } else if (description == header::DataDescription{"MA_RN2_SP"}) {
          outputs.adopt(Output{origin, description, version}, maker(o2::aod::Run2MatchedSparseMetadata{}));
        } else if (description == header::DataDescription{"MA_RN3_EX"}) {
          outputs.adopt(Output{origin, description, version}, maker(o2::aod::Run3MatchedExclusiveMetadata{}));
        } else if (description == header::DataDescription{"MA_RN3_SP"}) {
          outputs.adopt(Output{origin, description, version}, maker(o2::aod::Run3MatchedSparseMetadata{}));
        } else if (description == header::DataDescription{"MA_BCCOL_EX"}) {
          outputs.adopt(Output{origin, description, version}, maker(o2::aod::MatchedBCCollisionsExclusiveMetadata{}));
        } else if (description == header::DataDescription{"MA_BCCOL_SP"}) {
          outputs.adopt(Output{origin, description, version}, maker(o2::aod::MatchedBCCollisionsSparseMetadata{}));
        } else if (description == header::DataDescription{"MA_RN3_BC_SP"}) {
          outputs.adopt(Output{origin, description, version}, maker(o2::aod::Run3MatchedToBCSparseMetadata{}));
        } else if (description == header::DataDescription{"MA_RN3_BC_EX"}) {
          outputs.adopt(Output{origin, description, version}, maker(o2::aod::Run3MatchedToBCExclusiveMetadata{}));
        } else if (description == header::DataDescription{"MA_RN2_BC_SP"}) {
          outputs.adopt(Output{origin, description, version}, maker(o2::aod::Run2MatchedToBCSparseMetadata{}));
        } else {
          throw std::runtime_error("Not an index table");
        }
      }
    };
  }};
}

AlgorithmSpec AODReaderHelpers::aodSpawnerCallback(std::vector<InputSpec>& requested)
{
  return AlgorithmSpec::InitCallback{[requested](InitContext& ic) {
    return [requested](ProcessingContext& pc) {
      auto outputs = pc.outputs();
      // spawn tables
      for (auto& input : requested) {
        auto&& [origin, description, version] = DataSpecUtils::asConcreteDataMatcher(input);

        auto maker = [&](auto metadata) {
          using metadata_t = decltype(metadata);
          using expressions = typename metadata_t::expression_pack_t;
          std::vector<std::shared_ptr<arrow::Table>> originalTables;
          for (auto& i : input.metadata) {
            if ((i.type == VariantType::String) && (i.name.find("input:") != std::string::npos)) {
              auto spec = DataSpecUtils::fromMetadataString(i.defaultValue.get<std::string>());
              originalTables.push_back(pc.inputs().get<TableConsumer>(spec.binding)->asArrowTable());
            }
          }
          return o2::framework::spawner(expressions{}, std::move(originalTables), input.binding.c_str());
        };

        if (description == header::DataDescription{"TRACK"}) {
          outputs.adopt(Output{origin, description, version}, maker(o2::aod::TracksExtensionMetadata{}));
        } else if (description == header::DataDescription{"TRACK_IU"}) {
          outputs.adopt(Output{origin, description, version}, maker(o2::aod::TracksIUExtensionMetadata{}));
        } else if (description == header::DataDescription{"TRACKCOV"}) {
          outputs.adopt(Output{origin, description, version}, maker(o2::aod::TracksCovExtensionMetadata{}));
        } else if (description == header::DataDescription{"TRACKCOV_IU"}) {
          outputs.adopt(Output{origin, description, version}, maker(o2::aod::TracksCovIUExtensionMetadata{}));
        } else if (description == header::DataDescription{"TRACKEXTRA"}) {
          outputs.adopt(Output{origin, description, version}, maker(o2::aod::TracksExtraExtensionMetadata{}));
        } else if (description == header::DataDescription{"MFTTRACK"}) {
          outputs.adopt(Output{origin, description, version}, maker(o2::aod::MFTTracksExtensionMetadata{}));
        } else if (description == header::DataDescription{"FWDTRACK"}) {
          outputs.adopt(Output{origin, description, version}, maker(o2::aod::FwdTracksExtensionMetadata{}));
        } else if (description == header::DataDescription{"FWDTRACKCOV"}) {
          outputs.adopt(Output{origin, description, version}, maker(o2::aod::FwdTracksCovExtensionMetadata{}));
        } else if (description == header::DataDescription{"MCPARTICLE"}) {
          outputs.adopt(Output{origin, description, version}, maker(o2::aod::McParticles_000ExtensionMetadata{}));
        } else if (description == header::DataDescription{"MCPARTICLE_001"}) {
          outputs.adopt(Output{origin, description, version}, maker(o2::aod::McParticles_001ExtensionMetadata{}));
        } else {
          throw runtime_error("Not an extended table");
        }
      }
    };
  }};
}

} // namespace o2::framework::readers
