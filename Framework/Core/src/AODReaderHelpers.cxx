// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/TableTreeHelpers.h"
#include "Framework/AODReaderHelpers.h"
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
AlgorithmSpec AODReaderHelpers::aodSpawnerCallback(std::vector<InputSpec> requested)
{
  return AlgorithmSpec::InitCallback{[requested](InitContext& ic) {
    auto& callbacks = ic.services().get<CallbackService>();
    auto endofdatacb = [](EndOfStreamContext& eosc) {
      auto& control = eosc.services().get<ControlService>();
      control.endOfStream();
      control.readyToQuit(QuitRequest::Me);
    };
    callbacks.set(CallbackService::Id::EndOfStream, endofdatacb);

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
    LOGP(INFO, "Input filename {}",filename);
    LOGP(INFO, "time-limit {}",options.get<int64_t>("time-limit"));

    // create a DataInputDirector
    auto didir = std::make_shared<DataInputDirector>(filename);
    if (options.isSet("aodr-jsonfile")) {
      auto jsonFile = options.get<std::string>("aodr-jsonfile");
      if (!didir->readJson(jsonFile)) {
        LOGP(ERROR, "Check the JSON document! Can not be properly parsed!");
      }
    }
    didir->printOut();

    // get the run time watchdog
    auto* watchdog = new RuntimeWatchdog(options.get<int64_t>("time-limit"));

    // create list of requested tables
    std::vector<OutputRoute> requestedTables(spec.outputs);

    auto fileCounter = std::make_shared<int>(0);
    auto numTF = std::make_shared<int>(-1);
    return adaptStateless([requestedTables,
                           fileCounter,
                           numTF,
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

      // Each parallel reader device.inputTimesliceId reads the files fileCounter*device.maxInputTimeslices+device.inputTimesliceId
      // the TF to read is numTF
      assert(device.inputTimesliceId < device.maxInputTimeslices);
      //size_t fi = (watchdog->numberTimeFrames * device.maxInputTimeslices) + device.inputTimesliceId;

      // loop over requested tables
      TTree *tr = nullptr;
      int fcnt = (fileCounter.get()[0] * device.maxInputTimeslices) + device.inputTimesliceId;
      int ntf = numTF.get()[0] + 1;
      LOGP(INFO, "counters {} / {} / {} / {}", device.inputTimesliceId, fileCounter.get()[0], fcnt, ntf);
      bool first = true;
      for (auto route : requestedTables) {

        // create a TreeToTable object
        auto concrete = DataSpecUtils::asConcreteDataMatcher(route.matcher);
        auto dh = header::DataHeader(concrete.description, concrete.origin, concrete.subSpec);

        tr = didir->getDataTree(dh, fcnt, ntf);
        if (!tr) {
          if (first) {
            fcnt += device.maxInputTimeslices;
            if (didir->atEnd(fcnt)) {
              LOGP(INFO, "No input files left to read!");
              didir->closeInputFiles();
              control.endOfStream();
              control.readyToQuit(QuitRequest::Me);
              return;
            }
            ntf = 0;
            tr = didir->getDataTree(dh, fcnt, ntf);
            if (!tr) {
              // throw exception!
            }
          } else {
            // throw exception!
          }
        } else {
          first = false;
        }
        LOGP(INFO, "Reading {}",tr->GetName());
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
      
      // save file number and time frame
      fileCounter.get()[0] = (fcnt - device.inputTimesliceId) / device.maxInputTimeslices;
      numTF.get()[0] = ntf;

    });
  })};

  return callback;
}

} // namespace o2::framework::readers
