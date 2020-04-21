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
#include "Framework/AnalysisDataModel.h"
#include "DataProcessingHelpers.h"
#include "Framework/RootTableBuilderHelpers.h"
#include "Framework/AlgorithmSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
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
  Tracks = 1 << 0,
  TracksCov = 1 << 1,
  TracksExtra = 1 << 2,
  Calo = 1 << 3,
  Muon = 1 << 4,
  VZero = 1 << 5,
  Zdc = 1 << 6,
  Trigger = 1 << 7,
  Collisions = 1 << 8,
  Timeframe = 1 << 9,
  Unknown = 1 << 11
};

uint64_t getMask(header::DataDescription description)
{

  if (description == header::DataDescription{"TRACKPAR"}) {
    return AODTypeMask::Tracks;
  } else if (description == header::DataDescription{"TRACKPARCOV"}) {
    return AODTypeMask::TracksCov;
  } else if (description == header::DataDescription{"TRACKEXTRA"}) {
    return AODTypeMask::TracksExtra;
  } else if (description == header::DataDescription{"CALO"}) {
    return AODTypeMask::Calo;
  } else if (description == header::DataDescription{"MUON"}) {
    return AODTypeMask::Muon;
  } else if (description == header::DataDescription{"VZERO"}) {
    return AODTypeMask::VZero;
  } else if (description == header::DataDescription{"ZDC"}) {
    return AODTypeMask::Zdc;
  } else if (description == header::DataDescription{"TRIGGER"}) {
    return AODTypeMask::Trigger;
  } else if (description == header::DataDescription{"COLLISION"}) {
    return AODTypeMask::Collisions;
  } else if (description == header::DataDescription{"TIMEFRAME"}) {
    return AODTypeMask::Timeframe;
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
          if (!reader || (reader && reader->IsInvalid())) {
            LOGP(ERROR, "Requested \"{}\" tree not found in input file \"{}\"", treeName, didir->getInputFilename(dh, fi));
          } else {
            auto& builder = outputs.make<TableBuilder>(Output{decltype(metadata)::origin(), decltype(metadata)::description()});
            RootTableBuilderHelpers::convertASoA<table_t>(builder, *reader);
          }
        }
      };
      tableMaker(o2::aod::CollisionsMetadata{}, AODTypeMask::Collisions, "O2collisions");
      tableMaker(o2::aod::TracksMetadata{}, AODTypeMask::Tracks, "O2tracks");
      tableMaker(o2::aod::TracksCovMetadata{}, AODTypeMask::TracksCov, "O2tracks");
      tableMaker(o2::aod::TracksExtraMetadata{}, AODTypeMask::TracksExtra, "O2tracks");
      tableMaker(o2::aod::CalosMetadata{}, AODTypeMask::Calo, "O2calo");
      tableMaker(o2::aod::MuonsMetadata{}, AODTypeMask::Muon, "O2muon");
      tableMaker(o2::aod::VZerosMetadata{}, AODTypeMask::VZero, "O2vzero");
      tableMaker(o2::aod::ZdcsMetadata{}, AODTypeMask::Zdc, "O2zdc");
      tableMaker(o2::aod::TriggersMetadata{}, AODTypeMask::Trigger, "O2trigger");

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
