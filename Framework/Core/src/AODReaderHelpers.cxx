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

#include <ROOT/RDataFrame.hxx>
#include <TGrid.h>
#include <TFile.h>

#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>
#include <arrow/io/interfaces.h>
#include <arrow/table.h>
#include <arrow/util/key_value_metadata.h>

#include <thread>

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

template <typename O>
static inline auto extractTypedOriginal(ProcessingContext& pc)
{
  ///FIXME: this should be done in invokeProcess() as some of the originals may be compound tables
  return O{pc.inputs().get<TableConsumer>(aod::MetadataTrait<O>::metadata::tableLabel())->asArrowTable()};
}

template <typename... Os>
static inline auto extractOriginalsTuple(framework::pack<Os...>, ProcessingContext& pc)
{
  return std::make_tuple(extractTypedOriginal<Os>(pc)...);
}

AlgorithmSpec AODReaderHelpers::indexBuilderCallback(std::vector<InputSpec> requested)
{
  return AlgorithmSpec::InitCallback{[requested](InitContext& ic) {
    setEOSCallback(ic);

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
                                                               extractOriginalsTuple(sources{}, pc));
          } else {
            return o2::framework::IndexSparse::indexBuilder(index_pack_t{},
                                                            extractTypedOriginal<Key>(pc),
                                                            extractOriginalsTuple(sources{}, pc));
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
          outputs.adopt(Output{origin, description}, maker(o2::aod::BCCollisionsExclusiveMetadata{}));
        } else if (description == header::DataDescription{"MA_BCCOL_SP"}) {
          outputs.adopt(Output{origin, description}, maker(o2::aod::BCCollisionsSparseMetadata{}));
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
    setEOSCallback(ic);

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

AlgorithmSpec AODReaderHelpers::rootFileReaderCallback()
{
  auto callback = AlgorithmSpec{adaptStateful([](ConfigParamRegistry const& options,
                                                 DeviceSpec const& spec) {
    auto filename = options.get<std::string>("aod-file");

    // create a DataInputDirector
    auto didir = std::make_shared<DataInputDirector>(filename);
    if (options.isSet("json-file")) {
      auto jsonFile = options.get<std::string>("json-file");
      if (!didir->readJson(jsonFile)) {
        LOGP(ERROR, "Check the JSON document! Can not be properly parsed!");
      }
    }

    // get the run time watchdog
    auto* watchdog = new RuntimeWatchdog(options.get<int64_t>("time-limit"));

    // create list of requested tables
    std::vector<OutputRoute> requestedTables(spec.outputs);

    auto counter = std::make_shared<int>(0);
    return adaptStateless([requestedTables,
                           watchdog,
                           didir](DataAllocator& outputs, ControlService& control, DeviceSpec const& device) {
      // check if RuntimeLimit is reached
      if (!watchdog->update()) {
        LOGP(INFO, "Run time exceeds run time limit of {} seconds!", watchdog->runTimeLimit);
        LOGP(INFO, "Stopping after time frame {}.", watchdog->numberTimeFrames - 1);
        didir->closeInputFiles();
        control.endOfStream();
        control.readyToQuit(QuitRequest::Me);
        return;
      }

      // Each parallel reader reads the files whose index is associated to
      // their inputTimesliceId
      assert(device.inputTimesliceId < device.maxInputTimeslices);
      size_t fi = (watchdog->numberTimeFrames * device.maxInputTimeslices) + device.inputTimesliceId;

      // check if EoF is reached
      if (didir->atEnd(fi)) {
        LOGP(INFO, "All input files processed");
        didir->closeInputFiles();
        control.endOfStream();
        control.readyToQuit(QuitRequest::Me);
        return;
      }

      // loop over requested tables
      for (auto route : requestedTables) {

        // create a TreeToTable object
        auto concrete = DataSpecUtils::asConcreteDataMatcher(route.matcher);
        auto dh = header::DataHeader(concrete.description, concrete.origin, concrete.subSpec);

        auto tr = didir->getDataTree(dh, fi);
        if (!tr) {
          char* table;
          sprintf(table, "%s/%s/%" PRIu32, concrete.origin.str, concrete.description.str, concrete.subSpec);
          LOGP(ERROR, "Error while retrieving the tree for \"{}\"!", table);
          return;
        }

        auto o = Output(dh);
        auto& t2t = outputs.make<TreeToTable>(o);

        // add branches to read
        auto colnames = aod::datamodel::getColumnNames(dh);
        if (colnames.size() == 0) {
          t2t.addAllColumns(tr);
        } else {
          for (auto colname : colnames) {
            t2t.addColumn(colname.c_str());
          }
        }

        // fill the table
        t2t.fill(tr);
      }
    });
  })};

  return callback;
}

} // namespace o2::framework::readers
