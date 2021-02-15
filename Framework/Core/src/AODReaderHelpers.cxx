// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/AODReaderHelpers.h"
#include "Framework/TableTreeHelpers.h"
#include "Framework/AnalysisHelpers.h"
#include "AnalysisDataModelHelpers.h"
#include "DataProcessingHelpers.h"
#include "ExpressionHelpers.h"
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
#include "Framework/DetectorResponse.h"

#include <Monitoring/Monitoring.h>

#include <ROOT/RDataFrame.hxx>
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
static inline auto doExtractOriginal(framework::pack<Ts...>, ProcessingContext& pc)
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
  return O{doExtractOriginal(soa::make_originals_from_type<O>(), pc)};
}

template <typename... Os>
static inline auto extractTypedOriginalsTuple(framework::pack<Os...>, ProcessingContext& pc)
{
  return std::make_tuple(extractTypedOriginal<Os>(pc)...);
}

template <typename O>
static inline auto extractOriginal(ProcessingContext& pc)
{
  return doExtractOriginal(soa::make_originals_from_type<O>(), pc);
}

template <typename... Os>
static inline auto extractOriginals(framework::pack<Os...>, ProcessingContext& pc)
{
  return std::vector{extractOriginal<Os>(pc)...};
}

AlgorithmSpec AODReaderHelpers::indexBuilderCallback(std::vector<InputSpec> requested)
{
  return AlgorithmSpec::InitCallback{[requested](InitContext& ic) {
    return [requested](ProcessingContext& pc) {
      auto outputs = pc.outputs();
      // spawn tables
      for (auto& input : requested) {
        auto description = std::visit(
          overloaded{
            [](ConcreteDataMatcher const& matcher) { return matcher.description; },
            [](auto&&) { return header::DataDescription{""}; }},
          input.matcher);

        auto origin = std::visit(
          overloaded{
            [](ConcreteDataMatcher const& matcher) { return matcher.origin; },
            [](auto&&) { return header::DataOrigin{""}; }},
          input.matcher);

        auto maker = [&](auto metadata) {
          using metadata_t = decltype(metadata);
          using Key = typename metadata_t::Key;
          using index_pack_t = typename metadata_t::index_pack_t;
          using sources = typename metadata_t::originals;
          if constexpr (metadata_t::exclusive == true) {
            return o2::framework::IndexExclusive::indexBuilder(index_pack_t{},
                                                               extractTypedOriginal<Key>(pc),
                                                               extractTypedOriginalsTuple(sources{}, pc));
          } else {
            return o2::framework::IndexSparse::indexBuilder(index_pack_t{},
                                                            extractTypedOriginal<Key>(pc),
                                                            extractTypedOriginalsTuple(sources{}, pc));
          }
        };

        if (description == header::DataDescription{"MA_RN2_EX"}) {
          outputs.adopt(Output{origin, description}, maker(o2::aod::Run2MatchedExclusiveMetadata{}));
        } else if (description == header::DataDescription{"MA_RN2_SP"}) {
          outputs.adopt(Output{origin, description}, maker(o2::aod::Run2MatchedSparseMetadata{}));
        } else if (description == header::DataDescription{"MA_RN3_EX"}) {
          outputs.adopt(Output{origin, description}, maker(o2::aod::Run3MatchedExclusiveMetadata{}));
        } else if (description == header::DataDescription{"MA_RN3_SP"}) {
          outputs.adopt(Output{origin, description}, maker(o2::aod::Run3MatchedSparseMetadata{}));
        } else if (description == header::DataDescription{"MA_BCCOL_EX"}) {
          outputs.adopt(Output{origin, description}, maker(o2::aod::MatchedBCCollisionsExclusiveMetadata{}));
        } else if (description == header::DataDescription{"MA_BCCOL_SP"}) {
          outputs.adopt(Output{origin, description}, maker(o2::aod::MatchedBCCollisionsSparseMetadata{}));
        } else if (description == header::DataDescription{"MA_RN3_BC_SP"}) {
          outputs.adopt(Output{origin, description}, maker(o2::aod::Run3MatchedToBCSparseMetadata{}));
        } else if (description == header::DataDescription{"MA_RN3_BC_EX"}) {
          outputs.adopt(Output{origin, description}, maker(o2::aod::Run3MatchedToBCExclusiveMetadata{}));
        } else {
          throw std::runtime_error("Not an index table");
        }
      }
    };
  }};
}

AlgorithmSpec AODReaderHelpers::aodSpawnerCallback(std::vector<InputSpec> requested)
{
  return AlgorithmSpec::InitCallback{[requested](InitContext& ic) {
    return [requested](ProcessingContext& pc) {
      auto outputs = pc.outputs();
      // spawn tables
      for (auto& input : requested) {
        auto description = std::visit(
          overloaded{
            [](ConcreteDataMatcher const& matcher) { return matcher.description; },
            [](auto&&) { return header::DataDescription{""}; }},
          input.matcher);

        auto origin = std::visit(
          overloaded{
            [](ConcreteDataMatcher const& matcher) { return matcher.origin; },
            [](auto&&) { return header::DataOrigin{""}; }},
          input.matcher);

        auto maker = [&](auto metadata) {
          using metadata_t = decltype(metadata);
          using expressions = typename metadata_t::expression_pack_t;
          auto original_table = pc.inputs().get<TableConsumer>(input.binding)->asArrowTable();
          return o2::framework::spawner(expressions{}, original_table.get());
        };

        if (description == header::DataDescription{"TRACK:PAR"}) {
          outputs.adopt(Output{origin, description}, maker(o2::aod::TracksExtensionMetadata{}));
        } else if (description == header::DataDescription{"TRACK:PARCOV"}) {
          outputs.adopt(Output{origin, description}, maker(o2::aod::TracksCovExtensionMetadata{}));
        } else if (description == header::DataDescription{"MUON"}) {
          outputs.adopt(Output{origin, description}, maker(o2::aod::MuonsExtensionMetadata{}));
        } else {
          throw runtime_error("Not an extended table");
        }
      }
    };
  }};
}

AlgorithmSpec AODReaderHelpers::pidBuilderCallback(std::vector<InputSpec> requested)
{
  return AlgorithmSpec::InitCallback{[requested](InitContext& ic) {
    // what.value = context.options().get<T>(what.name.c_str());
    // std::string signalname = ic.options().get<std::string>("signalname");
    // return [requested, signalname](ProcessingContext& pc) {
    return [requested](ProcessingContext& pc) {
      auto outputs = pc.outputs();
      // spawn tables
      for (auto& input : requested) {
        auto description = std::visit(
          overloaded{
            [](ConcreteDataMatcher const& matcher) { return matcher.description; },
            [](auto&&) { return header::DataDescription{""}; }},
          input.matcher);

        auto origin = std::visit(
          overloaded{
            [](ConcreteDataMatcher const& matcher) { return matcher.origin; },
            [](auto&&) { return header::DataOrigin{""}; }},
          input.matcher);

        auto maker = [&](const o2::track::pid_constants::ID id, auto metadata) {
          using metadata_t = decltype(metadata);
          using sources = typename metadata_t::sources_t;
          using Trks = soa::Join<aod::Tracks, aod::TracksExtra>;
          using Coll = aod::Collisions;

          TableBuilder builder;
          // The cursor will be filled as a table
          auto cursor = framework::FFL(builder.cursor<typename metadata_t::table_t>());
          // These are the input tables
          auto tables = extractOriginals(sources{}, pc);
          // Getting the sub-tables
          auto collisions = Coll{tables[0]};
          auto tracks = Trks{{tables[1], tables[2]}};
          tracks.bindExternalIndices(&collisions);

          // Setting up the response
          o2::pid::DetectorResponse response;
          response.LoadParamFromFile("/tmp/TPCParam.root", "BetheBloch", response.kSignal);
          response.LoadParamFromFile("/tmp/TPCParam.root", "TPCReso", response.kSigma);
          // const std::string path = "Analysis/PID/TPC";
          // response.LoadParam(response.kSignal, ccdb->getForTimeStamp<Parametrization>(path + "/" + signalname.value, timestamp.value));
          // response.LoadParam(response.kSigma, ccdb->getForTimeStamp<Parametrization>(path + "/" + sigmaname.value, timestamp.value));
          // Service<o2::ccdb::BasicCCDBManager> ccdb;

          for (auto& trk : tracks) {
            const float xsignal[2] = {trk.tpcInnerParam() / o2::track::PID::getMass(id), (float)o2::track::PID::getCharge(id)};
            const float exp_signal = response(response.kSignal, xsignal);
            const float xsigma[2] = {trk.tpcSignal(), (float)trk.tpcNClsFound()};
            cursor(0, exp_signal,
                   (trk.tpcSignal() - exp_signal) / response(response.kSigma, xsigma));
          }

          return builder.finalize();
        };
        // Dispatch
        if (description == header::DataDescription{"AutoPIDTPCEl"}) {
          outputs.adopt(Output{origin, description}, maker(o2::track::PID::Electron, o2::aod::AutoPIDTPCElMetadata{}));
        } else if (description == header::DataDescription{"AutoPIDTPCMu"}) {
          outputs.adopt(Output{origin, description}, maker(o2::track::PID::Muon, o2::aod::AutoPIDTPCMuMetadata{}));
        } else if (description == header::DataDescription{"AutoPIDTPCPi"}) {
          outputs.adopt(Output{origin, description}, maker(o2::track::PID::Pion, o2::aod::AutoPIDTPCPiMetadata{}));
        } else if (description == header::DataDescription{"AutoPIDTPCKa"}) {
          outputs.adopt(Output{origin, description}, maker(o2::track::PID::Kaon, o2::aod::AutoPIDTPCKaMetadata{}));
        } else if (description == header::DataDescription{"AutoPIDTPCPr"}) {
          outputs.adopt(Output{origin, description}, maker(o2::track::PID::Proton, o2::aod::AutoPIDTPCPrMetadata{}));
        } else if (description == header::DataDescription{"AutoPIDTPCDe"}) {
          outputs.adopt(Output{origin, description}, maker(o2::track::PID::Deuteron, o2::aod::AutoPIDTPCDeMetadata{}));
        } else if (description == header::DataDescription{"AutoPIDTPCTr"}) {
          outputs.adopt(Output{origin, description}, maker(o2::track::PID::Triton, o2::aod::AutoPIDTPCTrMetadata{}));
        } else if (description == header::DataDescription{"AutoPIDTPCHe"}) {
          outputs.adopt(Output{origin, description}, maker(o2::track::PID::Helium3, o2::aod::AutoPIDTPCHeMetadata{}));
        } else if (description == header::DataDescription{"AutoPIDTPCAl"}) {
          outputs.adopt(Output{origin, description}, maker(o2::track::PID::Alpha, o2::aod::AutoPIDTPCAlMetadata{}));
        } else {
          throw std::runtime_error("Not a PID table");
        }
      }
    };
  }};
}

} // namespace o2::framework::readers
