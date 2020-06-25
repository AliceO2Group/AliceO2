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
#include "Framework/AnalysisDataModel.h"
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

#include <FairMQDevice.h>
#include <ROOT/RDataFrame.hxx>
#include <TFile.h>

#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>
#include <arrow/io/interfaces.h>
#include <arrow/table.h>
#include <arrow/util/key_value_metadata.h>

#include <thread>

namespace o2::framework::readers
{
enum AODTypeMask : uint64_t {
  None = 0,
  Track = 1 << 0,
  TrackCov = 1 << 1,
  TrackExtra = 1 << 2,
  Calo = 1 << 3,
  CaloTrigger = 1 << 4,
  Muon = 1 << 5,
  MuonCluster = 1 << 6,
  Zdc = 1 << 7,
  BC = 1 << 8,
  Collision = 1 << 9,
  FT0 = 1 << 10,
  FV0 = 1 << 11,
  FDD = 1 << 12,
  UnassignedTrack = 1 << 13,
  Run2V0 = 1 << 14,
  McCollision = 1 << 15,
  McTrackLabel = 1 << 16,
  McCaloLabel = 1 << 17,
  McCollisionLabel = 1 << 18,
  McParticle = 1 << 19,
  Unknown = 1 << 20
};

uint64_t getMask(header::DataDescription description)
{

  if (description == header::DataDescription{"TRACKPAR"}) {
    return AODTypeMask::Track;
  } else if (description == header::DataDescription{"TRACKPARCOV"}) {
    return AODTypeMask::TrackCov;
  } else if (description == header::DataDescription{"TRACKEXTRA"}) {
    return AODTypeMask::TrackExtra;
  } else if (description == header::DataDescription{"CALO"}) {
    return AODTypeMask::Calo;
  } else if (description == header::DataDescription{"CALOTRIGGER"}) {
    return AODTypeMask::CaloTrigger;
  } else if (description == header::DataDescription{"MUON"}) {
    return AODTypeMask::Muon;
  } else if (description == header::DataDescription{"MUONCLUSTER"}) {
    return AODTypeMask::MuonCluster;
  } else if (description == header::DataDescription{"ZDC"}) {
    return AODTypeMask::Zdc;
  } else if (description == header::DataDescription{"BC"}) {
    return AODTypeMask::BC;
  } else if (description == header::DataDescription{"COLLISION"}) {
    return AODTypeMask::Collision;
  } else if (description == header::DataDescription{"FT0"}) {
    return AODTypeMask::FT0;
  } else if (description == header::DataDescription{"FV0"}) {
    return AODTypeMask::FV0;
  } else if (description == header::DataDescription{"FDD"}) {
    return AODTypeMask::FDD;
  } else if (description == header::DataDescription{"UNASSIGNEDTRACK"}) {
    return AODTypeMask::UnassignedTrack;
  } else if (description == header::DataDescription{"RUN2V0"}) {
    return AODTypeMask::Run2V0;
  } else if (description == header::DataDescription{"MCCOLLISION"}) {
    return AODTypeMask::McCollision;
  } else if (description == header::DataDescription{"MCTRACKLABEL"}) {
    return AODTypeMask::McTrackLabel;
  } else if (description == header::DataDescription{"MCCALOLABEL"}) {
    return AODTypeMask::McCaloLabel;
  } else if (description == header::DataDescription{"MCCOLLISLABEL"}) {
    return AODTypeMask::McCollisionLabel;
  } else if (description == header::DataDescription{"MCPARTICLE"}) {
    return AODTypeMask::McParticle;
  } else {
    LOG(DEBUG) << "This is a tree of unknown type! " << description.str;
    return AODTypeMask::Unknown;
  }
}

uint64_t calculateReadMask(std::vector<OutputRoute> const& routes, header::DataOrigin const& origin)
{
  uint64_t readMask = None;
  for (auto& route : routes) {
    auto concrete = DataSpecUtils::asConcreteDataTypeMatcher(route.matcher);
    auto description = concrete.description;

    readMask |= getMask(description);
  }
  return readMask;
}

std::vector<OutputRoute> getListOfUnknown(std::vector<OutputRoute> const& routes)
{

  std::vector<OutputRoute> unknows;
  for (auto& route : routes) {
    auto concrete = DataSpecUtils::asConcreteDataTypeMatcher(route.matcher);

    if (getMask(concrete.description) == AODTypeMask::Unknown)
      unknows.push_back(route);
  }
  return unknows;
}

/// Expression-based column generator to materialize columns
template <typename... C>
auto spawner(framework::pack<C...> columns, arrow::Table* atable)
{
  arrow::TableBatchReader reader(*atable);
  std::shared_ptr<arrow::RecordBatch> batch;
  arrow::ArrayVector v;
  std::vector<arrow::ArrayVector> chunks(sizeof...(C));

  auto projectors = framework::expressions::createProjectors(columns, atable->schema());
  while (true) {
    auto s = reader.ReadNext(&batch);
    if (!s.ok()) {
      throw std::runtime_error(fmt::format("Cannot read batches from table {}", s.ToString()));
    }
    if (batch == nullptr) {
      break;
    }
    s = projectors->Evaluate(*batch, arrow::default_memory_pool(), &v);
    if (!s.ok()) {
      throw std::runtime_error(fmt::format("Cannot apply projector {}", s.ToString()));
    }
    for (auto i = 0u; i < sizeof...(C); ++i) {
      chunks[i].emplace_back(v.at(i));
    }
  }
  std::vector<std::shared_ptr<arrow::ChunkedArray>> results(sizeof...(C));
  for (auto i = 0u; i < sizeof...(C); ++i) {
    results[i] = std::make_shared<arrow::ChunkedArray>(chunks[i]);
  }
  return results;
}

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
          auto extra_schema = o2::soa::createSchemaFromColumns(expressions{});
          auto original_table = pc.inputs().get<TableConsumer>(input.binding)->asArrowTable();
          auto original_fields = original_table->schema()->fields();
          std::vector<std::shared_ptr<arrow::Field>> fields;
          auto arrays = spawner(expressions{}, original_table.get());
          std::vector<std::shared_ptr<arrow::ChunkedArray>> columns = original_table->columns();
          std::copy(original_fields.begin(), original_fields.end(), std::back_inserter(fields));
          for (auto i = 0u; i < framework::pack_size(expressions{}); ++i) {
            columns.push_back(arrays[i]);
            fields.emplace_back(extra_schema->field(i));
          }
          auto new_schema = std::make_shared<arrow::Schema>(fields);
          return arrow::Table::Make(new_schema, columns);
        };

        if (description == header::DataDescription{"TRACKPAR"}) {
          outputs.adopt(Output{origin, description}, maker(o2::aod::TracksMetadata{}));
        } else if (description == header::DataDescription{"TRACKPARCOV"}) {
          outputs.adopt(Output{origin, description}, maker(o2::aod::TracksCovMetadata{}));
        } else {
          throw std::runtime_error("Not an extended table");
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

    // analyze type of requested tables
    uint64_t readMask = calculateReadMask(spec.outputs, header::DataOrigin{"AOD"});
    std::vector<OutputRoute> unknowns;
    if (readMask & AODTypeMask::Unknown) {
      unknowns = getListOfUnknown(spec.outputs);
    }

    auto counter = std::make_shared<int>(0);
    return adaptStateless([readMask,
                           unknowns,
                           counter,
                           didir](DataAllocator& outputs, ControlService& control, DeviceSpec const& device) {
      // Each parallel reader reads the files whose index is associated to
      // their inputTimesliceId
      assert(device.inputTimesliceId < device.maxInputTimeslices);
      size_t fi = (*counter * device.maxInputTimeslices) + device.inputTimesliceId;
      *counter += 1;

      if (didir->atEnd(fi)) {
        LOGP(INFO, "All input files processed");
        didir->closeInputFiles();
        control.endOfStream();
        control.readyToQuit(QuitRequest::Me);
        return;
      }

      auto tableMaker = [&readMask, &outputs, fi, didir](auto metadata, AODTypeMask mask, char const* treeName) {
        if (readMask & mask) {

          auto dh = header::DataHeader(decltype(metadata)::description(), decltype(metadata)::origin(), 0);
          auto reader = didir->getTreeReader(dh, fi, treeName);

          using table_t = typename decltype(metadata)::table_t;
          if (!reader || (reader->IsInvalid())) {
            LOGP(ERROR, "Requested \"{}\" tree not found in input file \"{}\"", treeName, didir->getInputFilename(dh, fi));
          } else {
            auto& builder = outputs.make<TableBuilder>(Output{decltype(metadata)::origin(), decltype(metadata)::description()});
            RootTableBuilderHelpers::convertASoA<table_t>(builder, *reader);
          }
        }
      };
      tableMaker(o2::aod::CollisionsMetadata{}, AODTypeMask::Collision, "O2collision");
      tableMaker(o2::aod::StoredTracksMetadata{}, AODTypeMask::Track, "O2track");
      tableMaker(o2::aod::StoredTracksCovMetadata{}, AODTypeMask::TrackCov, "O2track");
      tableMaker(o2::aod::TracksExtraMetadata{}, AODTypeMask::TrackExtra, "O2track");
      tableMaker(o2::aod::CalosMetadata{}, AODTypeMask::Calo, "O2calo");
      tableMaker(o2::aod::CaloTriggersMetadata{}, AODTypeMask::Calo, "O2calotrigger");
      tableMaker(o2::aod::MuonsMetadata{}, AODTypeMask::Muon, "O2muon");
      tableMaker(o2::aod::MuonClustersMetadata{}, AODTypeMask::Muon, "O2muoncluster");
      tableMaker(o2::aod::ZdcsMetadata{}, AODTypeMask::Zdc, "O2zdc");
      tableMaker(o2::aod::BCsMetadata{}, AODTypeMask::BC, "O2bc");
      tableMaker(o2::aod::FT0sMetadata{}, AODTypeMask::FT0, "O2ft0");
      tableMaker(o2::aod::FV0sMetadata{}, AODTypeMask::FV0, "O2fv0");
      tableMaker(o2::aod::FDDsMetadata{}, AODTypeMask::FDD, "O2fdd");
      tableMaker(o2::aod::UnassignedTracksMetadata{}, AODTypeMask::UnassignedTrack, "O2unassignedtrack");
      tableMaker(o2::aod::Run2V0sMetadata{}, AODTypeMask::Run2V0, "Run2v0");
      tableMaker(o2::aod::McCollisionsMetadata{}, AODTypeMask::McCollision, "O2mccollision");
      tableMaker(o2::aod::McTrackLabelsMetadata{}, AODTypeMask::McTrackLabel, "O2mctracklabel");
      tableMaker(o2::aod::McCaloLabelsMetadata{}, AODTypeMask::McCaloLabel, "O2mccalolabel");
      tableMaker(o2::aod::McCollisionLabelsMetadata{}, AODTypeMask::McCollisionLabel, "O2mccollisionlabel");
      tableMaker(o2::aod::McParticlesMetadata{}, AODTypeMask::McParticle, "O2mcparticle");

      // tables not included in the DataModel
      if (readMask & AODTypeMask::Unknown) {

        // loop over unknowns
        for (auto route : unknowns) {

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
          auto& t2t = outputs.make<TreeToTable>(o, tr);

          // fill the table
          t2t.fill();
        }
      }
    });
  })};

  return callback;
}

} // namespace o2::framework::readers
