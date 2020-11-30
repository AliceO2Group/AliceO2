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

#include <Monitoring/Monitoring.h>

#include <ROOT/RDataFrame.hxx>
#include <TGrid.h>
#include <TFile.h>

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
                                                 DeviceSpec const& spec,
                                                 Monitoring& monitoring) {
    monitoring.send(Metric{(uint64_t)0, "arrow-bytes-created"}.addTag(Key::Subsystem, monitoring::tags::Value::DPL));
    monitoring.send(Metric{(uint64_t)0, "arrow-messages-created"}.addTag(Key::Subsystem, monitoring::tags::Value::DPL));
    monitoring.send(Metric{(uint64_t)0, "arrow-bytes-destroyed"}.addTag(Key::Subsystem, monitoring::tags::Value::DPL));
    monitoring.send(Metric{(uint64_t)0, "arrow-messages-destroyed"}.addTag(Key::Subsystem, monitoring::tags::Value::DPL));
    monitoring.flushBuffer();

    if (!options.isSet("aod-file")) {
      LOGP(ERROR, "No input file defined!");
      throw std::runtime_error("Processing is stopped!");
    }

    auto filename = options.get<std::string>("aod-file");

    // create a DataInputDirector
    auto didir = std::make_shared<DataInputDirector>(filename);
    if (options.isSet("aod-reader-json")) {
      auto jsonFile = options.get<std::string>("aod-reader-json");
      if (!didir->readJson(jsonFile)) {
        LOGP(ERROR, "Check the JSON document! Can not be properly parsed!");
      }
    }

    // get the run time watchdog
    auto* watchdog = new RuntimeWatchdog(options.get<int64_t>("time-limit"));

    // selected the TFN input and
    // create list of requested tables
    header::DataHeader TFNumberHeader;
    std::vector<OutputRoute> requestedTables;
    std::vector<OutputRoute> routes(spec.outputs);
    for (auto route : routes) {
      if (DataSpecUtils::partialMatch(route.matcher, header::DataOrigin("TFN"))) {
        auto concrete = DataSpecUtils::asConcreteDataMatcher(route.matcher);
        TFNumberHeader = header::DataHeader(concrete.description, concrete.origin, concrete.subSpec);
      } else {
        requestedTables.emplace_back(route);
      }
    }

    auto fileCounter = std::make_shared<int>(0);
    auto numTF = std::make_shared<int>(-1);
    return adaptStateless([TFNumberHeader,
                           requestedTables,
                           fileCounter,
                           numTF,
                           watchdog,
                           didir](Monitoring& monitoring, DataAllocator& outputs, ControlService& control, DeviceSpec const& device) {
      // check if RuntimeLimit is reached
      if (!watchdog->update()) {
        LOGP(INFO, "Run time exceeds run time limit of {} seconds!", watchdog->runTimeLimit);
        LOGP(INFO, "Stopping reader {} after time frame {}.", device.inputTimesliceId, watchdog->numberTimeFrames - 1);
        monitoring.flushBuffer();
        didir->closeInputFiles();
        control.endOfStream();
        control.readyToQuit(QuitRequest::Me);
        return;
      }

      // Each parallel reader device.inputTimesliceId reads the files fileCounter*device.maxInputTimeslices+device.inputTimesliceId
      // the TF to read is numTF
      assert(device.inputTimesliceId < device.maxInputTimeslices);
      uint64_t timeFrameNumber = 0;
      int fcnt = (*fileCounter * device.maxInputTimeslices) + device.inputTimesliceId;
      int ntf = *numTF + 1;
      monitoring.send(Metric{(uint64_t)ntf, "tf-sent"}.addTag(Key::Subsystem, monitoring::tags::Value::DPL));

      // loop over requested tables
      TTree* tr = nullptr;
      bool first = true;
      static size_t totalSizeUncompressed = 0;
      static size_t totalSizeCompressed = 0;
      static size_t totalReadCalls = 0;

      for (auto route : requestedTables) {

        // create header
        auto concrete = DataSpecUtils::asConcreteDataMatcher(route.matcher);
        auto dh = header::DataHeader(concrete.description, concrete.origin, concrete.subSpec);

        // create a TreeToTable object
        auto info = didir->getFileFolder(dh, fcnt, ntf);
        size_t before = 0;
        if (info.file) {
          info.file->GetReadCalls();
        }
        tr = didir->getDataTree(dh, fcnt, ntf);
        if (!tr) {
          if (first) {
            // check if there is a next file to read
            fcnt += device.maxInputTimeslices;
            if (didir->atEnd(fcnt)) {
              LOGP(INFO, "No input files left to read for reader {}!", device.inputTimesliceId);
              didir->closeInputFiles();
              control.endOfStream();
              control.readyToQuit(QuitRequest::Me);
              return;
            }
            // get first folder of next file
            ntf = 0;
            tr = didir->getDataTree(dh, fcnt, ntf);
            if (!tr) {
              LOGP(FATAL, "Can not retrieve tree for table {}: fileCounter {}, timeFrame {}", concrete.origin, fcnt, ntf);
              throw std::runtime_error("Processing is stopped!");
            }
          } else {
            LOGP(FATAL, "Can not retrieve tree for table {}: fileCounter {}, timeFrame {}", concrete.origin, fcnt, ntf);
            throw std::runtime_error("Processing is stopped!");
          }
        }
        tr->SetCacheSize(0);
        if (first) {
          timeFrameNumber = didir->getTimeFrameNumber(dh, fcnt, ntf);
          auto o = Output(TFNumberHeader);
          outputs.make<uint64_t>(o) = timeFrameNumber;
        }

        // create table output
        auto o = Output(dh);
        auto& t2t = outputs.make<TreeToTable>(o);

        // add branches to read
        // fill the table

        auto colnames = aod::datamodel::getColumnNames(dh);
        if (colnames.size() == 0) {
          totalSizeCompressed += tr->GetZipBytes();
          totalSizeUncompressed += tr->GetTotBytes();
          t2t.addAllColumns(tr);
        } else {
          for (auto& colname : colnames) {
            TBranch* branch = tr->GetBranch(colname.c_str());
            totalSizeCompressed += branch->GetZipBytes("*");
            totalSizeUncompressed += branch->GetTotBytes("*");
            t2t.addColumn(colname.c_str());
          }
        }
        t2t.fill(tr);
        if (info.file) {
          totalReadCalls += info.file->GetReadCalls() - before;
        }
        delete tr;

        first = false;
      }
      monitoring.send(Metric{(uint64_t)totalSizeUncompressed, "aod-bytes-read-uncompressed"}.addTag(Key::Subsystem, monitoring::tags::Value::DPL));
      monitoring.send(Metric{(uint64_t)totalSizeCompressed, "aod-bytes-read-compressed"}.addTag(Key::Subsystem, monitoring::tags::Value::DPL));
      monitoring.send(Metric{(uint64_t)totalReadCalls, "aod-total-read-calls"}.addTag(Key::Subsystem, monitoring::tags::Value::DPL));

      // save file number and time frame
      *fileCounter = (fcnt - device.inputTimesliceId) / device.maxInputTimeslices;
      *numTF = ntf;
    });
  })};

  return callback;
}

} // namespace o2::framework::readers
