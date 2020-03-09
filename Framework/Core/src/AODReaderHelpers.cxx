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
namespace
{

// Input stream that just reads from stdin.
class FileStream : public arrow::io::InputStream
{
 public:
  FileStream(FILE* stream) : mStream(stream)
  {
    set_mode(arrow::io::FileMode::READ);
  }
  ~FileStream() override = default;

  // FIXME: handle return code
  arrow::Status Close() override { return arrow::Status::OK(); }
  bool closed() const override { return false; }

  arrow::Status Tell(int64_t* position) const override
  {
    *position = mPos;
    return arrow::Status::OK();
  }

  arrow::Status Read(int64_t nbytes, int64_t* bytes_read, void* out) override
  {
    auto count = fread(out, nbytes, 1, mStream);
    if (ferror(mStream) == 0) {
      *bytes_read = nbytes;
      mPos += nbytes;
    } else {
      *bytes_read = 0;
    }
    return arrow::Status::OK();
  }

  arrow::Status Read(int64_t nbytes, std::shared_ptr<arrow::Buffer>* out) override
  {
    std::shared_ptr<arrow::ResizableBuffer> buffer;
    ARROW_RETURN_NOT_OK(AllocateResizableBuffer(nbytes, &buffer));
    int64_t bytes_read;
    ARROW_RETURN_NOT_OK(Read(nbytes, &bytes_read, buffer->mutable_data()));
    ARROW_RETURN_NOT_OK(buffer->Resize(bytes_read, false));
    buffer->ZeroPadding();
    *out = buffer;
    return arrow::Status::OK();
  }

 private:
  FILE* mStream = nullptr;
  int64_t mPos = 0;
};

} // anonymous namespace

enum AODTypeMask : uint64_t {
  None = 0,
  Tracks = 1 << 0,
  TracksCov = 1 << 1,
  TracksExtra = 1 << 2,
  Calo = 1 << 3,
  Muon = 1 << 4,
  VZero = 1 << 5,
  Collisions = 1 << 6,
  Timeframe = 1 << 7,
  DZeroFlagged = 1 << 8,
  Unknown = 1 << 9
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
  } else if (description == header::DataDescription{"COLLISION"}) {
    return AODTypeMask::Collisions;
  } else if (description == header::DataDescription{"TIMEFRAME"}) {
    return AODTypeMask::Timeframe;
  } else if (description == header::DataDescription{"DZEROFLAGGED"}) {
    return AODTypeMask::DZeroFlagged;
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

AlgorithmSpec AODReaderHelpers::run2ESDConverterCallback()
{
  auto callback = AlgorithmSpec{adaptStateful([](ConfigParamRegistry const& options,
                                                 ControlService& control,
                                                 DeviceSpec const& spec) {
    std::vector<std::string> filenames;
    auto filename = options.get<std::string>("esd-file");
    auto nEvents = options.get<int>("events");

    if (filename.empty()) {
      LOG(error) << "Option --esd-file did not provide a filename";
      control.readyToQuit(QuitRequest::All);
      return adaptStateless([](RawDeviceService& service) {
        service.device()->WaitFor(std::chrono::milliseconds(1000));
      });
    }

    // If option starts with a @, we consider the file as text which contains a list of
    // files.
    if (filename.size() && filename[0] == '@') {
      try {
        filename.erase(0, 1);
        std::ifstream filelist(filename);
        while (std::getline(filelist, filename)) {
          filenames.push_back(filename);
        }
      } catch (...) {
        LOG(error) << "Unable to process file list: " << filename;
      }
    } else {
      filenames.push_back(filename);
    }

    uint64_t readMask = calculateReadMask(spec.outputs, header::DataOrigin{"AOD"});
    auto counter = std::make_shared<unsigned int>(0);
    return adaptStateless([readMask,
                           counter,
                           filenames,
                           spec, nEvents](DataAllocator& outputs, ControlService& ctrl, RawDeviceService& service) {
      if (*counter >= filenames.size()) {
        LOG(info) << "All input files processed";
        ctrl.endOfStream();
        ctrl.readyToQuit(QuitRequest::Me);
        return;
      }
      auto f = filenames[*counter];
      setenv("O2RUN2CONVERTER", "run2ESD2Run3AOD", 0);
      auto command = std::string(getenv("O2RUN2CONVERTER"));
      if (nEvents > 0) {
        command += fmt::format(" -n {} ", nEvents);
      }
      FILE* pipe = popen((command + " " + f).c_str(), "r");
      if (pipe == nullptr) {
        LOG(ERROR) << "Unable to run converter: " << (command + " " + f).c_str() << f;
        ctrl.endOfStream();
        ctrl.readyToQuit(QuitRequest::All);
        return;
      }
      *counter += 1;

      /// We keep reading until the popen is not empty.
      while ((feof(pipe) == false) && (ferror(pipe) == 0)) {
        // Skip extra 0-padding...
        int c = fgetc(pipe);
        if (c == 0 || c == EOF) {
          continue;
        }
        ungetc(c, pipe);

        std::shared_ptr<arrow::RecordBatchReader> reader;
        auto input = std::make_shared<FileStream>(pipe);
        auto readerStatus = arrow::ipc::RecordBatchStreamReader::Open(input, &reader);
        if (readerStatus.ok() == false) {
          LOG(ERROR) << "Reader status not ok: " << readerStatus.message();
          break;
        }

        while (true) {
          std::shared_ptr<arrow::RecordBatch> batch;
          auto status = reader->ReadNext(&batch);
          if (batch.get() == nullptr) {
            break;
          }
          std::unordered_map<std::string, std::string> meta;
          batch->schema()->metadata()->ToUnorderedMap(&meta);
          std::shared_ptr<arrow::ipc::RecordBatchWriter> writer;
          if (meta["description"] == "TRACKPAR" && (readMask & AODTypeMask::Tracks)) {
            writer = outputs.make<arrow::ipc::RecordBatchWriter>(Output{"AOD", "TRACKPAR"}, batch->schema());
          } else if (meta["description"] == "TRACKPARCOV" && (readMask & AODTypeMask::TracksCov)) {
            writer = outputs.make<arrow::ipc::RecordBatchWriter>(Output{"AOD", "TRACKPARCOV"}, batch->schema());
          } else if (meta["description"] == "TRACKEXTRA" && (readMask & AODTypeMask::TracksExtra)) {
            writer = outputs.make<arrow::ipc::RecordBatchWriter>(Output{"AOD", "TRACKEXTRA"}, batch->schema());
          } else if (meta["description"] == "CALO" && (readMask & AODTypeMask::Calo)) {
            writer = outputs.make<arrow::ipc::RecordBatchWriter>(Output{"AOD", "CALO"}, batch->schema());
          } else if (meta["description"] == "MUON" && (readMask & AODTypeMask::Muon)) {
            writer = outputs.make<arrow::ipc::RecordBatchWriter>(Output{"AOD", "MUON"}, batch->schema());
          } else if (meta["description"] == "VZERO" && (readMask & AODTypeMask::VZero)) {
            writer = outputs.make<arrow::ipc::RecordBatchWriter>(Output{"AOD", "VZERO"}, batch->schema());
          } else if (meta["description"] == "COLLISION" && (readMask & AODTypeMask::Collisions)) {
            writer = outputs.make<arrow::ipc::RecordBatchWriter>(Output{"AOD", "COLLISION"}, batch->schema());
          } else if (meta["description"] == "TIMEFRAME" && (readMask & AODTypeMask::Timeframe)) {
            writer = outputs.make<arrow::ipc::RecordBatchWriter>(Output{"AOD", "TIMEFRAME"}, batch->schema());
          } else {
            continue;
          }
          auto writeStatus = writer->WriteRecordBatch(*batch);
          if (writeStatus.ok() == false) {
            throw std::runtime_error("Error while writing record");
          }
        }
      }
      if (ferror(pipe)) {
        LOG(ERROR) << "Error while reading from PIPE";
      }
      pclose(pipe);
    });
  })};

  return callback;
}

AlgorithmSpec AODReaderHelpers::rootFileReaderCallback()
{
  auto callback = AlgorithmSpec{adaptStateful([](ConfigParamRegistry const& options,
                                                 DeviceSpec const& spec) {
    std::vector<std::string> filenames;
    auto filename = options.get<std::string>("aod-file");

    // If option starts with a @, we consider the file as text which contains a list of
    // files.
    if (filename.size() && filename[0] == '@') {
      try {
        filename.erase(0, 1);
        std::ifstream filelist(filename);
        while (std::getline(filelist, filename)) {
          filenames.push_back(filename);
        }
      } catch (...) {
        LOG(error) << "Unable to process file list: " << filename;
      }
    } else {
      filenames.push_back(filename);
    }

    // analyze type of requested tables
    uint64_t readMask = calculateReadMask(spec.outputs, header::DataOrigin{"AOD"});
    std::vector<OutputRoute> unknowns;
    if (readMask & AODTypeMask::Unknown)
      unknowns = getListOfUnknown(spec.outputs);

    auto counter = std::make_shared<int>(0);
    return adaptStateless([readMask,
                           unknowns,
                           counter,
                           filenames](DataAllocator& outputs, ControlService& control, DeviceSpec const& device) {
      // Each parallel reader reads the files whose index is associated to
      // their inputTimesliceId
      assert(device.inputTimesliceId < device.maxInputTimeslices);
      size_t fi = (*counter * device.maxInputTimeslices) + device.inputTimesliceId;
      if (fi >= filenames.size()) {
        LOG(info) << "All input files processed";
        control.endOfStream();
        control.readyToQuit(QuitRequest::Me);
        return;
      }
      auto f = filenames[fi];
      LOG(INFO) << "Processing " << f;
      auto infile = std::make_unique<TFile>(f.c_str());
      *counter += 1;
      if (infile.get() == nullptr || infile->IsOpen() == false) {
        LOG(ERROR) << "File not found: " + f;
        return;
      }

      /// FIXME: Substitute here the actual data you want to convert for the AODReader
      if (readMask & AODTypeMask::Collisions) {
        std::unique_ptr<TTreeReader> reader = std::make_unique<TTreeReader>("O2events", infile.get());
        auto& collisionBuilder = outputs.make<TableBuilder>(Output{"AOD", "COLLISION"});
        RootTableBuilderHelpers::convertASoA<o2::aod::Collisions>(collisionBuilder, *reader);
      }

      if (readMask & AODTypeMask::Tracks) {
        std::unique_ptr<TTreeReader> reader = std::make_unique<TTreeReader>("O2tracks", infile.get());
        auto& trackParBuilder = outputs.make<TableBuilder>(Output{"AOD", "TRACKPAR"});
        RootTableBuilderHelpers::convertASoA<o2::aod::Tracks>(trackParBuilder, *reader);
      }

      if (readMask & AODTypeMask::TracksCov) {
        std::unique_ptr<TTreeReader> covReader = std::make_unique<TTreeReader>("O2tracks", infile.get());
        auto& trackParCovBuilder = outputs.make<TableBuilder>(Output{"AOD", "TRACKPARCOV"});
        RootTableBuilderHelpers::convertASoA<o2::aod::TracksCov>(trackParCovBuilder, *covReader);
      }

      if (readMask & AODTypeMask::TracksExtra) {
        std::unique_ptr<TTreeReader> extraReader = std::make_unique<TTreeReader>("O2tracks", infile.get());
        auto& extraBuilder = outputs.make<TableBuilder>(Output{"AOD", "TRACKEXTRA"});
        RootTableBuilderHelpers::convertASoA<o2::aod::TracksExtra>(extraBuilder, *extraReader);
      }

      if (readMask & AODTypeMask::Calo) {
        std::unique_ptr<TTreeReader> extraReader = std::make_unique<TTreeReader>("O2calo", infile.get());
        auto& extraBuilder = outputs.make<TableBuilder>(Output{"AOD", "CALO"});
        RootTableBuilderHelpers::convertASoA<o2::aod::Calos>(extraBuilder, *extraReader);
      }

      if (readMask & AODTypeMask::Muon) {
        std::unique_ptr<TTreeReader> muReader = std::make_unique<TTreeReader>("O2muon", infile.get());
        auto& muBuilder = outputs.make<TableBuilder>(Output{"AOD", "MUON"});
        RootTableBuilderHelpers::convertASoA<o2::aod::Muons>(muBuilder, *muReader);
      }

      if (readMask & AODTypeMask::VZero) {
        std::unique_ptr<TTreeReader> vzReader = std::make_unique<TTreeReader>("O2vzero", infile.get());
        auto& vzBuilder = outputs.make<TableBuilder>(Output{"AOD", "VZERO"});
        RootTableBuilderHelpers::convertASoA<o2::aod::Muons>(vzBuilder, *vzReader);
      }

      // Candidates as described by Gianmichele example
      if (readMask & AODTypeMask::DZeroFlagged) {
        std::unique_ptr<TTreeReader> dzReader = std::make_unique<TTreeReader>("fTreeDzeroFlagged", infile.get());

        TTreeReaderValue<float> c0(*dzReader, "d_len_ML");
        TTreeReaderValue<int> c1(*dzReader, "cand_type_ML");
        TTreeReaderValue<float> c2(*dzReader, "cos_p_ML");
        TTreeReaderValue<float> c3(*dzReader, "cos_p_xy_ML");
        TTreeReaderValue<float> c4(*dzReader, "d_len_xy_ML");
        TTreeReaderValue<float> c5(*dzReader, "eta_prong0_ML");
        TTreeReaderValue<float> c6(*dzReader, "eta_prong1_ML");
        TTreeReaderValue<float> c7(*dzReader, "imp_par_prong0_ML");
        TTreeReaderValue<float> c8(*dzReader, "imp_par_prong1_ML");
        TTreeReaderValue<float> c9(*dzReader, "imp_par_xy_ML");
        TTreeReaderValue<float> c10(*dzReader, "inv_mass_ML");
        TTreeReaderValue<float> c11(*dzReader, "max_norm_d0d0exp_ML");
        TTreeReaderValue<float> c12(*dzReader, "norm_dl_xy_ML");
        TTreeReaderValue<float> c13(*dzReader, "pt_cand_ML");
        TTreeReaderValue<float> c14(*dzReader, "pt_prong0_ML");
        TTreeReaderValue<float> c15(*dzReader, "pt_prong1_ML");
        TTreeReaderValue<float> c16(*dzReader, "y_cand_ML");
        TTreeReaderValue<float> c17(*dzReader, "phi_cand_ML");
        TTreeReaderValue<float> c18(*dzReader, "eta_cand_ML");
        TTreeReaderValue<int> c19(*dzReader, "cand_evtID_ML");
        TTreeReaderValue<int> c20(*dzReader, "cand_fileID_ML");

        auto& dzBuilder = outputs.make<TableBuilder>(Output{"AOD", "DZEROFLAGGED"});
        RootTableBuilderHelpers::convertTTree(dzBuilder, *dzReader,
                                              c0, c1, c2, c3, c4, c5, c6, c7,
                                              c8, c9, c10, c11, c12, c13, c14,
                                              c15, c16, c17, c18, c19, c20);
      }

      // tables not included in the DataModel
      if (readMask & AODTypeMask::Unknown) {

        // loop over unknowns
        for (auto route : unknowns) {
          auto concrete = DataSpecUtils::asConcreteDataMatcher(route.matcher);

          // get the tree from infile
          auto trname = concrete.description.str;
          auto tr = (TTree*)infile.get()->Get(trname);
          if (!tr) {
            LOG(ERROR) << "Tree " << trname << "is not contained in file " << f;
            return;
          }

          // create a TreeToTable object
          auto h = header::DataHeader(concrete.description, concrete.origin, concrete.subSpec);
          auto o = Output(h);
          auto& t2t = outputs.make<TreeToTable>(o, tr);

          // fill the table
          t2t.Fill();
        }
      }
    });
  })};

  return callback;
}

} // namespace o2::framework::readers
