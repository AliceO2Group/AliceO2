// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "AODJAlienReaderHelpers.h"
#include "Framework/TableTreeHelpers.h"
#include "Framework/AnalysisHelpers.h"
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
#if __has_include(<TJAlienFile.h>)
#include <TJAlienFile.h>
#endif
#include <TGrid.h>
#include <TFile.h>
#include <TTreeCache.h>

#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>
#include <arrow/io/interfaces.h>
#include <arrow/table.h>
#include <arrow/util/key_value_metadata.h>

#include <thread>

using namespace o2;
using namespace o2::aod;

struct RuntimeWatchdog {
  int numberTimeFrames;
  uint64_t startTime;
  uint64_t lastTime;
  double runTime;
  uint64_t runTimeLimit;

  RuntimeWatchdog(Long64_t limit)
  {
    numberTimeFrames = -1;
    startTime = uv_hrtime();
    lastTime = startTime;
    runTime = 0.;
    runTimeLimit = limit;
  }

  bool update()
  {
    numberTimeFrames++;
    if (runTimeLimit <= 0) {
      return true;
    }

    auto nowTime = uv_hrtime();

    // time spent to process the time frame
    double time_spent = numberTimeFrames < 1 ? (double)(nowTime - lastTime) / 1.E9 : 0.;
    runTime += time_spent;
    lastTime = nowTime;

    return ((double)(lastTime - startTime) / 1.E9 + runTime / (numberTimeFrames + 1)) < runTimeLimit;
  }

  void printOut()
  {
    LOGP(INFO, "RuntimeWatchdog");
    LOGP(INFO, "  run time limit: {}", runTimeLimit);
    LOGP(INFO, "  number of time frames: {}", numberTimeFrames);
    LOGP(INFO, "  estimated run time per time frame: {}", (numberTimeFrames >= 0) ? runTime / (numberTimeFrames + 1) : 0.);
    LOGP(INFO, "  estimated total run time: {}", (double)(lastTime - startTime) / 1.E9 + ((numberTimeFrames >= 0) ? runTime / (numberTimeFrames + 1) : 0.));
  }
};

template <typename... C>
static constexpr auto columnNamesTrait(framework::pack<C...>)
{
  return std::vector<std::string>{C::columnLabel()...};
}

std::vector<std::string> getColumnNames(header::DataHeader dh)
{
  auto description = std::string(dh.dataDescription.str);
  auto origin = std::string(dh.dataOrigin.str);

  // get column names
  // AOD / RN2
  if (origin == "AOD") {
    if (description == "TRACK:PAR") {
      return columnNamesTrait(typename StoredTracksMetadata::table_t::persistent_columns_t{});
    } else if (description == "TRACK:PARCOV") {
      return columnNamesTrait(typename StoredTracksCovMetadata::table_t::persistent_columns_t{});
    } else if (description == "TRACK:EXTRA") {
      return columnNamesTrait(typename TracksExtraMetadata::table_t::persistent_columns_t{});
    }
  }

  // default: column names = {}
  return std::vector<std::string>({});
}

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

void AODJAlienReaderHelpers::dumpFileMetrics(Monitoring& monitoring, TFile* currentFile, uint64_t startedAt, int tfPerFile, int tfRead)
{
  if (currentFile == nullptr) {
    return;
  }
  std::string monitoringInfo(fmt::format("lfn={},size={},total_tf={},read_tf={},read_bytes={},read_calls={},run_time={:.1f}", currentFile->GetName(),
                                         currentFile->GetSize(), tfPerFile, tfRead, currentFile->GetBytesRead(), currentFile->GetReadCalls(), ((float)(uv_hrtime() - startedAt) / 1e9)));
#if __has_include(<TJAlienFile.h>)
  auto alienFile = dynamic_cast<TJAlienFile*>(currentFile);
  if (alienFile) {
    monitoringInfo += fmt::format(",se={},open_time={:.1f}", alienFile->GetSE(), alienFile->GetElapsed());
  }
#endif
  monitoring.send(Metric{monitoringInfo, "aod-file-read-info"}.addTag(Key::Subsystem, monitoring::tags::Value::DPL));
  LOGP(INFO, "Read info: {}", monitoringInfo);
}

AlgorithmSpec AODJAlienReaderHelpers::rootFileReaderCallback()
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
      LOGP(FATAL, "No input file defined!");
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
      // Each parallel reader device.inputTimesliceId reads the files fileCounter*device.maxInputTimeslices+device.inputTimesliceId
      // the TF to read is numTF
      assert(device.inputTimesliceId < device.maxInputTimeslices);
      uint64_t timeFrameNumber = 0;
      int fcnt = (*fileCounter * device.maxInputTimeslices) + device.inputTimesliceId;
      int ntf = *numTF + 1;
      static int currentFileCounter = -1;
      static int filesProcessed = 0;
      if (currentFileCounter != *fileCounter) {
        currentFileCounter = *fileCounter;
        monitoring.send(Metric{(uint64_t)++filesProcessed, "files-opened"}.addTag(Key::Subsystem, monitoring::tags::Value::DPL));
      }

      // loop over requested tables
      bool first = true;
      static size_t totalSizeUncompressed = 0;
      static size_t totalSizeCompressed = 0;
      static TFile* currentFile = nullptr;
      static int tfCurrentFile = -1;
      static auto currentFileStartedAt = uv_hrtime();

      // check if RuntimeLimit is reached
      if (!watchdog->update()) {
        LOGP(INFO, "Run time exceeds run time limit of {} seconds!", watchdog->runTimeLimit);
        LOGP(INFO, "Stopping reader {} after time frame {}.", device.inputTimesliceId, watchdog->numberTimeFrames - 1);
        dumpFileMetrics(monitoring, currentFile, currentFileStartedAt, tfCurrentFile, ntf);
        monitoring.flushBuffer();
        didir->closeInputFiles();
        control.endOfStream();
        control.readyToQuit(QuitRequest::Me);
        return;
      }

      for (auto route : requestedTables) {

        // create header
        auto concrete = DataSpecUtils::asConcreteDataMatcher(route.matcher);
        auto dh = header::DataHeader(concrete.description, concrete.origin, concrete.subSpec);

        // create a TreeToTable object
        TTree* tr = didir->getDataTree(dh, fcnt, ntf);
        if (!tr) {
          if (first) {
            // dump metrics of file which is done for reading
            dumpFileMetrics(monitoring, currentFile, currentFileStartedAt, tfCurrentFile, ntf);
            currentFile = nullptr;
            currentFileStartedAt = uv_hrtime();

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
        auto colnames = getColumnNames(dh);
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
        delete tr;

        // needed for metrics dumping (upon next file read, or terminate due to watchdog)
        if (currentFile == nullptr) {
          currentFile = didir->getFileFolder(dh, fcnt, ntf).file;
          tfCurrentFile = didir->getTimeFramesInFile(dh, fcnt);
        }

        first = false;
      }
      monitoring.send(Metric{(uint64_t)ntf, "tf-sent"}.addTag(Key::Subsystem, monitoring::tags::Value::DPL));
      monitoring.send(Metric{(uint64_t)totalSizeUncompressed / 1000, "aod-bytes-read-uncompressed"}.addTag(Key::Subsystem, monitoring::tags::Value::DPL));
      monitoring.send(Metric{(uint64_t)totalSizeCompressed / 1000, "aod-bytes-read-compressed"}.addTag(Key::Subsystem, monitoring::tags::Value::DPL));

      // save file number and time frame
      *fileCounter = (fcnt - device.inputTimesliceId) / device.maxInputTimeslices;
      *numTF = ntf;
    });
  })};

  return callback;
}

} // namespace o2::framework::readers
