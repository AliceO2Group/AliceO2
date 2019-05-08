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
#include "Framework/RootTableBuilderHelpers.h"
#include "Framework/AlgorithmSpec.h"
#include "Framework/ControlService.h"
#include "Framework/DeviceSpec.h"
#include "Framework/RawDeviceService.h"
#include <FairMQDevice.h>
#include <ROOT/RDataFrame.hxx>
#include <TFile.h>

#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>
#include <arrow/io/interfaces.h>
#include <arrow/table.h>

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
      LOG(INFO) << mPos;
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

AlgorithmSpec AODReaderHelpers::run2ESDConverterCallback()
{
  auto callback = AlgorithmSpec{ adaptStateful([](ConfigParamRegistry const& options,
                                                  ControlService& control,
                                                  DeviceSpec const& spec) {
    std::vector<std::string> filenames;
    auto filename = options.get<std::string>("esd-file");

    if (filename.empty()) {
      LOG(error) << "Option --esd-file did not provide a filename";
      control.readyToQuit(true);
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

    bool readTracks = false;
    bool readTracksCov = false;
    bool readTracksExtra = false;
    bool readCalo = false;
    bool readMuon = false;
    bool readVZ = false;

    // FIXME: bruteforce but effective.
    for (auto& route : spec.outputs) {
      if (route.matcher.origin != header::DataOrigin{ "RN2" }) {
        continue;
      }
      auto description = route.matcher.description;
      if (description == header::DataDescription{ "TRACKPAR" }) {
        readTracks = true;
      } else if (description == header::DataDescription{ "TRACKPARCOV" }) {
        readTracksCov = true;
      } else if (description == header::DataDescription{ "TRACKEXTRA" }) {
        readTracksExtra = true;
      } else if (description == header::DataDescription{ "CALO" }) {
        readCalo = true;
      } else if (description == header::DataDescription{ "MUON" }) {
        readMuon = true;
      } else if (description == header::DataDescription{ "VZERO" }) {
        readVZ = true;
      } else {
        throw std::runtime_error(std::string("Unknown AOD type: ") + route.matcher.description.str);
      }
    }

    auto counter = std::make_shared<int>(0);
    return adaptStateless([readTracks,
                           readTracksCov,
                           readTracksExtra,
                           readCalo,
                           readMuon,
                           readVZ,
                           counter,
                           filenames](DataAllocator& outputs, ControlService& ctrl, RawDeviceService& service) {
      if (*counter >= filenames.size()) {
        LOG(info) << "All input files processed";
        ctrl.readyToQuit(false);
        service.device()->WaitFor(std::chrono::milliseconds(1000));
        return;
      }
      auto f = filenames[*counter];
      setenv("O2RUN2CONVERTER", "run2ESD2Run3AOD", 0);
      auto command = std::string(getenv("O2RUN2CONVERTER"));
      FILE* pipe = popen((command + " " + f).c_str(), "r");
      if (pipe == nullptr) {
        LOG(ERROR) << "Unable to run converter: " << (command + " " + f).c_str() << f;
        ctrl.readyToQuit(true);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        return;
      }
      *counter += 1;
      auto input = std::make_shared<FileStream>(pipe);

      /// We keep reading until the popen is not empty.
      while ((feof(pipe) == false) && (ferror(pipe) == 0)) {
        std::shared_ptr<arrow::RecordBatchReader> reader;
        auto readerStatus = arrow::ipc::RecordBatchStreamReader::Open(input.get(), &reader);
        if (readerStatus.ok() == false) {
          break;
        }

        while (true) {
          std::shared_ptr<arrow::RecordBatch> batch;
          auto status = reader->ReadNext(&batch);
          if (batch.get() == nullptr) {
            LOG(INFO) << "End of batches reached";
            break;
          }
          std::unordered_map<std::string, std::string> meta;
          batch->schema()->metadata()->ToUnorderedMap(&meta);
          LOG(INFO) << "description:" << meta["description"];
          std::shared_ptr<arrow::ipc::RecordBatchWriter> writer;
          if (meta["description"] == "TRACKPAR" && readTracks) {
            writer = outputs.make<arrow::ipc::RecordBatchWriter>(Output{ "RN2", "TRACKPAR" }, batch->schema());
          } else if (meta["description"] == "TRACKPARCOV" && readTracksCov) {
            writer = outputs.make<arrow::ipc::RecordBatchWriter>(Output{ "RN2", "TRACKPARCOV" }, batch->schema());
          } else if (meta["description"] == "TRACKEXTRA" && readTracksExtra) {
            writer = outputs.make<arrow::ipc::RecordBatchWriter>(Output{ "RN2", "TRACKEXTRA" }, batch->schema());
          } else if (meta["description"] == "CALO" && readCalo) {
            writer = outputs.make<arrow::ipc::RecordBatchWriter>(Output{ "RN2", "CALO" }, batch->schema());
          } else if (meta["description"] == "MUON" && readMuon) {
            writer = outputs.make<arrow::ipc::RecordBatchWriter>(Output{ "RN2", "MUON" }, batch->schema());
          } else if (meta["description"] == "VZERO" && readVZ) {
            writer = outputs.make<arrow::ipc::RecordBatchWriter>(Output{ "RN2", "VZERO" }, batch->schema());
          } else {
            continue;
          }
          auto writeStatus = writer->WriteRecordBatch(*batch);
          if (writeStatus.ok() == false) {
            throw std::runtime_error("Error while writing record");
          }
        }
      }
      pclose(pipe);
    });
  }) };

  return callback;
}

AlgorithmSpec AODReaderHelpers::rootFileReaderCallback()
{
  auto callback = AlgorithmSpec{ adaptStateful([](ConfigParamRegistry const& options,
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

    bool readTracks = false;
    bool readTracksCov = false;
    bool readTracksExtra = false;
    bool readCalo = false;
    bool readMuon = false;
    bool readDZeroFlagged = false;
    bool readVZ = false;

    // FIXME: bruteforce but effective.
    for (auto& route : spec.outputs) {
      if (route.matcher.origin != header::DataOrigin{ "AOD" }) {
        continue;
      }
      auto description = route.matcher.description;
      if (description == header::DataDescription{ "TRACKPAR" }) {
        readTracks = true;
      } else if (description == header::DataDescription{ "TRACKPARCOV" }) {
        readTracksCov = true;
      } else if (description == header::DataDescription{ "TRACKEXTRA" }) {
        readTracksExtra = true;
      } else if (description == header::DataDescription{ "CALO" }) {
        readCalo = true;
      } else if (description == header::DataDescription{ "MUON" }) {
        readMuon = true;
      } else if (description == header::DataDescription{ "DZEROFLAGGED" }) {
        readDZeroFlagged = true;
      } else if (description == header::DataDescription{ "VZERO" }) {
        readVZ = true;
      } else {
        throw std::runtime_error(std::string("Unknown AOD type: ") + route.matcher.description.str);
      }
    }

    auto counter = std::make_shared<int>(0);
    return adaptStateless([readTracks,
                           readTracksCov,
                           readTracksExtra,
                           readCalo,
                           readMuon,
                           readDZeroFlagged,
                           readVZ,
                           counter,
                           filenames](DataAllocator& outputs, ControlService& control) {
      if (*counter >= filenames.size()) {
        LOG(info) << "All input files processed";
        control.readyToQuit(false);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        return;
      }
      auto f = filenames[*counter];
      auto infile = std::make_unique<TFile>(f.c_str());
      *counter += 1;
      if (infile.get() == nullptr || infile->IsOpen() == false) {
        LOG(ERROR) << "File not found: " + f;
        return;
      }

      /// FIXME: Substitute here the actual data you want to convert for the AODReader
      if (readTracks) {
        std::unique_ptr<TTreeReader> reader = std::make_unique<TTreeReader>("O2tracks", infile.get());
        auto& trackParBuilder = outputs.make<TableBuilder>(Output{ "AOD", "TRACKPAR" });
        TTreeReaderValue<int> c0(*reader, "fID4Tracks");
        TTreeReaderValue<float> c1(*reader, "fX");
        TTreeReaderValue<float> c2(*reader, "fAlpha");
        TTreeReaderValue<float> c3(*reader, "fY");
        TTreeReaderValue<float> c4(*reader, "fZ");
        TTreeReaderValue<float> c5(*reader, "fSnp");
        TTreeReaderValue<float> c6(*reader, "fTgl");
        TTreeReaderValue<float> c7(*reader, "fSigned1Pt");
        RootTableBuilderHelpers::convertTTree(trackParBuilder, *reader,
                                              c0, c1, c2, c3, c4, c5, c6, c7);
      }

      if (readTracksCov) {
        std::unique_ptr<TTreeReader> covReader = std::make_unique<TTreeReader>("O2tracks", infile.get());
        TTreeReaderValue<float> c0(*covReader, "fCYY");
        TTreeReaderValue<float> c1(*covReader, "fCZY");
        TTreeReaderValue<float> c2(*covReader, "fCZZ");
        TTreeReaderValue<float> c3(*covReader, "fCSnpY");
        TTreeReaderValue<float> c4(*covReader, "fCSnpZ");
        TTreeReaderValue<float> c5(*covReader, "fCSnpSnp");
        TTreeReaderValue<float> c6(*covReader, "fCTglSnp");
        TTreeReaderValue<float> c7(*covReader, "fCTglTgl");
        TTreeReaderValue<float> c8(*covReader, "fC1PtY");
        TTreeReaderValue<float> c9(*covReader, "fC1PtZ");
        TTreeReaderValue<float> c10(*covReader, "fC1PtSnp");
        TTreeReaderValue<float> c11(*covReader, "fC1PtTgl");
        TTreeReaderValue<float> c12(*covReader, "fC1Pt21Pt2");
        auto& trackParCovBuilder = outputs.make<TableBuilder>(Output{ "AOD", "TRACKPARCOV" });
        RootTableBuilderHelpers::convertTTree(trackParCovBuilder, *covReader,
                                              c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12);
      }

      if (readTracksExtra) {
        std::unique_ptr<TTreeReader> extraReader = std::make_unique<TTreeReader>("O2tracks", infile.get());
        TTreeReaderValue<float> c0(*extraReader, "fTPCinnerP");
        TTreeReaderValue<uint64_t> c1(*extraReader, "fFlags");
        TTreeReaderValue<unsigned char> c2(*extraReader, "fITSClusterMap");
        TTreeReaderValue<unsigned short> c3(*extraReader, "fTPCncls");
        TTreeReaderValue<unsigned char> c4(*extraReader, "fTRDntracklets");
        TTreeReaderValue<float> c5(*extraReader, "fITSchi2Ncl");
        TTreeReaderValue<float> c6(*extraReader, "fTPCchi2Ncl");
        TTreeReaderValue<float> c7(*extraReader, "fTRDchi2");
        TTreeReaderValue<float> c8(*extraReader, "fTOFchi2");
        TTreeReaderValue<float> c9(*extraReader, "fTPCsignal");
        TTreeReaderValue<float> c10(*extraReader, "fTRDsignal");
        TTreeReaderValue<float> c11(*extraReader, "fTOFsignal");
        TTreeReaderValue<float> c12(*extraReader, "fLength");
        auto& extraBuilder = outputs.make<TableBuilder>(Output{ "AOD", "TRACKEXTRA" });
        RootTableBuilderHelpers::convertTTree(extraBuilder, *extraReader,
                                              c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11);
      }

      if (readCalo) {
        std::unique_ptr<TTreeReader> extraReader = std::make_unique<TTreeReader>("O2calo", infile.get());
        TTreeReaderValue<int> c0(*extraReader, "fID4Calo");
        TTreeReaderValue<short> c1(*extraReader, "fCellNumber");
        TTreeReaderValue<float> c2(*extraReader, "fAmplitude");
        TTreeReaderValue<float> c3(*extraReader, "fTime");
        TTreeReaderValue<int8_t> c4(*extraReader, "fType");
        auto& extraBuilder = outputs.make<TableBuilder>(Output{ "AOD", "CALO" });
        RootTableBuilderHelpers::convertTTree(extraBuilder, *extraReader,
                                              c0, c1, c2, c3, c4);
      }

      if (readMuon) {
        std::unique_ptr<TTreeReader> muReader = std::make_unique<TTreeReader>("O2mu", infile.get());
        TTreeReaderValue<int> c0(*muReader, "fID4mu");
        TTreeReaderValue<float> c1(*muReader, "fInverseBendingMomentum");
        TTreeReaderValue<float> c2(*muReader, "fThetaX");
        TTreeReaderValue<float> c3(*muReader, "fThetaY");
        TTreeReaderValue<float> c4(*muReader, "fZmu");
        TTreeReaderValue<float> c5(*muReader, "fBendingCoor");
        TTreeReaderValue<float> c6(*muReader, "fNonBendingCoor");
        TTreeReaderArray<float> c7(*muReader, "fCovariances");
        TTreeReaderValue<float> c8(*muReader, "fChi2");
        TTreeReaderValue<float> c9(*muReader, "fChi2MatchTrigger");
        TTreeReaderValue<int> c10(*muReader, "fID4vz");
        auto& muBuilder = outputs.make<TableBuilder>(Output{ "AOD", "MUON" });
        RootTableBuilderHelpers::convertTTree(muBuilder, *muReader,
                                              c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10);
      }

      if (readVZ) {
        std::unique_ptr<TTreeReader> vzReader = std::make_unique<TTreeReader>("O2vz", infile.get());
        TTreeReaderArray<float> c0(*vzReader, "fAdcVZ"); // FIXME: we do not support arrays for now
        TTreeReaderArray<float> c1(*vzReader, "fTimeVZ");
        TTreeReaderArray<float> c2(*vzReader, "fWidthVZ");
        auto& vzBuilder = outputs.make<TableBuilder>(Output{ "AOD", "VZERO" });
        RootTableBuilderHelpers::convertTTree(vzBuilder, *vzReader,
                                              c0, c1, c2);
      }

      // Candidates as described by Gianmichele example
      if (readDZeroFlagged) {
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

        auto& dzBuilder = outputs.make<TableBuilder>(Output{ "AOD", "DZEROFLAGGED" });
        RootTableBuilderHelpers::convertTTree(dzBuilder, *dzReader,
                                              c0, c1, c2, c3, c4, c5, c6, c7,
                                              c8, c9, c10, c11, c12, c13, c14,
                                              c15, c16, c17, c18, c19, c20);
      }
    });
  }) };

  return callback;
}

} // namespace o2::framework::readers
