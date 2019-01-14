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
#include "Framework/DeviceSpec.h"
#include <ROOT/RDataFrame.hxx>
#include <TFile.h>

namespace o2::framework::readers
{

AlgorithmSpec AODReaderHelpers::rootFileReaderCallback()
{
  auto callback = AlgorithmSpec{ adaptStateful([](ConfigParamRegistry const& options,
                                                  DeviceSpec const& spec) {
    std::shared_ptr<TFile> infile;
    try {
      auto filename = options.get<std::string>("aod-file");
      infile = std::make_shared<TFile>(filename.c_str());
    } catch (...) {
      LOG(ERROR) << "Unable to open file";
    }

    bool readTracks = false;
    bool readTracksCov = false;
    bool readTracksExtra = false;
    bool readCalo = false;
    bool readMuon = false;
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
      } else if (description == header::DataDescription{ "VZERO" }) {
        readVZ = true;
      } else {
        throw std::runtime_error(std::string("Unknown AOD type: ") + route.matcher.description.str);
      }
    }

    return adaptStateless([readTracks,
                           readTracksCov,
                           readTracksExtra,
                           readCalo,
                           readMuon,
                           readVZ,
                           infile](DataAllocator& outputs) {
      if (infile.get() == nullptr || infile->IsOpen() == false) {
        LOG(ERROR) << "File not found: aod.root";
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
    });
  }) };

  return callback;
}

} // namespace o2::framework::readers
