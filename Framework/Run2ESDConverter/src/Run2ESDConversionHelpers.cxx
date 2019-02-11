// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Run2ESDConversionHelpers.h"
#include "Framework/TableBuilder.h"
#include "Framework/AlgorithmSpec.h"
#include "Framework/ControlService.h"

#include "AliESDEvent.h"
#include "AliESDtrack.h"
#include "AliESDVZERO.h"
#include "AliESDCaloCells.h"
#include "AliESDMuonTrack.h"
#include "AliExternalTrackParam.h"

#include <Monitoring/Monitoring.h>

#include <TFile.h>
#include <memory>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

namespace o2
{
namespace framework
{
namespace run2
{

AlgorithmSpec Run2ESDConversionHelpers::getESDConverter()
{
  return AlgorithmSpec{
    adaptStateful([](ConfigParamRegistry const& options) {
      // File can be obtained from the
      std::shared_ptr<TFile> infile;
      std::string filename;
      filename = options.get<std::string>("esd-file");
      std::vector<std::string> filenames;
      // If option starts with a @, we consider the file as text which contains a list of
      // files.
      if (filename.size() && filename[0] == '@') {
        filename.erase(0, 1);
        std::ifstream filelist(filename);
        while (std::getline(filelist, filename)) {
          filenames.push_back(filename);
        }
      } else {
        filenames.push_back(filename);
      }

      auto counter = std::make_shared<int>(0);

      return adaptStateless([filenames, counter](DataAllocator& outputs,
                                                 ControlService& control,
                                                 monitoring::Monitoring& monitoring) {
        // We have read all the files.
        if (*counter >= filenames.size()) {
          control.readyToQuit(false);
          return;
        }
        std::string f = filenames[*counter];
        *counter += 1;
        std::unique_ptr<TFile> infile;
        try {
          infile = std::make_unique<TFile>(f.c_str());
        } catch (...) {
          LOG(ERROR) << "Unable to open file: " + f;
          return;
        }

        if (infile->IsOpen() == false) {
          LOG(ERROR) << "Unable to open file: " << f;
          return;
        }

        auto& trackParBuilder = outputs.make<TableBuilder>(Output{ "ESD", "TRACKPAR" });
        auto& trackParCovBuilder = outputs.make<TableBuilder>(Output{ "ESD", "TRACKPARCOV" });
        auto& trackExtraBuilder = outputs.make<TableBuilder>(Output{ "ESD", "TRACKEXTRA" });
        auto& caloBuilder = outputs.make<TableBuilder>(Output{ "ESD", "CALO" });
        auto& muonBuilder = outputs.make<TableBuilder>(Output{ "ESD", "MUON" });
        auto& v0Builder = outputs.make<TableBuilder>(Output{ "ESD", "VZERO" });

        auto trackFiller = trackParBuilder.persist<
          int,
          float,
          float,
          float,
          float,
          float,
          float,
          float>({ "fID4Tracks",
                   "fX",
                   "fAlpha",
                   "fY",
                   "fZ",
                   "fSnp",
                   "fTgl",
                   "fSigned1Pt" });

        auto sigmaFiller = trackParCovBuilder.persist<
          float,
          float,
          float,
          float,
          float,
          float,
          float,
          float,
          float,
          float,
          float,
          float,
          float,
          float,
          float>({ "fCYY",
                   "fCZY",
                   "fCZZ",
                   "fCSnpY",
                   "fCSnpZ",
                   "fCSnpSnp",
                   "fCTglY",
                   "fCTglZ",
                   "fCTglSnp",
                   "fCTglTgl",
                   "fC1PtY",
                   "fC1PtZ",
                   "fC1PtSnp",
                   "fC1PtTgl",
                   "fC1Pt21Pt2" });

        auto extraFiller = trackExtraBuilder.persist<
          float,
          uint64_t,
          uint8_t,
          uint16_t,
          uint8_t,
          float,
          float,
          float,
          float,
          float,
          float,
          float,
          float>({ "fTPCinnerP",
                   "fFlags",
                   "fITSClusterMap",
                   "fTPCncls",
                   "fTRDntracklets",
                   "fITSchi2Ncl",
                   "fTPCchi2Ncl",
                   "fTRDchi2",
                   "fTOFchi2",
                   "fTPCsignal",
                   "fTRDsignal",
                   "fTOFsignal",
                   "fLength" });

        auto caloFiller = caloBuilder.persist<
          int32_t,
          int64_t,
          float,
          float,
          int8_t>(
          {
            "fID4Calo",
            "fCellNumber",
            "fAmplitude",
            "fTime",
            "fType",
          });

        auto muonFiller = muonBuilder.persist<
          int,
          float,
          float,
          float,
          float,
          float,
          float,
          //float, // fixme... we need to support arrays...
          float,
          float>({
          "fID4mu",
          "fInverseBendingMomentum",
          "fThetaX",
          "fThetaY",
          "fZmu",
          "fBendingCoor",
          "fNonBendingCoor",
          // "fCovariances",
          "fChi2",
          "fChi2MatchTrigger",
        });

        auto vzeroFiller = v0Builder.persist<
          int>({ "fID4vz" });

        TFile* fEsd = infile.get();
        TTree* tEsd = (TTree*)fEsd->Get("esdTree");
        AliESDEvent* esd = new AliESDEvent();
        esd->ReadFromTree(tEsd);
        size_t nev = tEsd->GetEntries();
        for (size_t iev = 0; iev < nev; ++iev) {
          esd->Reset();
          tEsd->GetEntry(iev);
          esd->ConnectTracks();

          // Tracks information
          size_t ntrk = esd->GetNumberOfTracks();
          for (size_t itrk = 0; itrk < ntrk; ++itrk) {
            AliESDtrack* track = esd->GetTrack(itrk);
            track->SetESDEvent(esd);
            trackFiller(
              0,
              iev,
              track->GetX(),
              track->GetAlpha(),
              track->GetY(),
              track->GetZ(),
              track->GetSnp(),
              track->GetTgl(),
              track->GetSigned1Pt());

            sigmaFiller(
              0,
              track->GetSigmaY2(),
              track->GetSigmaZY(),
              track->GetSigmaZ2(),
              track->GetSigmaSnpY(),
              track->GetSigmaSnpZ(),
              track->GetSigmaSnp2(),
              track->GetSigmaTglY(),
              track->GetSigmaTglZ(),
              track->GetSigmaTglSnp(),
              track->GetSigmaTgl2(),
              track->GetSigma1PtY(),
              track->GetSigma1PtZ(),
              track->GetSigma1PtSnp(),
              track->GetSigma1PtTgl(),
              track->GetSigma1Pt2());
            const AliExternalTrackParam* intp = track->GetTPCInnerParam();

            extraFiller(
              0,
              (intp ? intp->GetP() : 0), // Set the momentum to 0 if the track did not reach TPC      //
              track->GetStatus(),
              //
              track->GetITSClusterMap(),
              track->GetTPCNcls(),
              track->GetTRDntracklets(),
              //
              (track->GetITSNcls() ? track->GetITSchi2() / track->GetITSNcls() : 0),
              (track->GetTPCNcls() ? track->GetTPCchi2() / track->GetTPCNcls() : 0),
              track->GetTRDchi2(),
              track->GetTOFchi2(),
              //
              track->GetTPCsignal(),
              track->GetTRDsignal(),
              track->GetTOFsignal(),
              track->GetIntegratedLength());
          } // End loop on tracks

          // Calorimeters:
          // EMCAL
          AliESDCaloCells* cells = esd->GetEMCALCells();
          size_t nCells = cells->GetNumberOfCells();
          auto cellType = cells->GetType();
          for (size_t ice = 0; ice < nCells; ++ice) {
            Short_t cellNumber;
            Double_t amplitude;
            Double_t time;
            Int_t mclabel;
            Double_t efrac;

            cells->GetCell(ice, cellNumber, amplitude, time, mclabel, efrac);
            caloFiller(0, iev, cellNumber, amplitude, time, cellType);
          }

          // PHOS
          cells = esd->GetPHOSCells();
          nCells = cells->GetNumberOfCells();
          cellType = cells->GetType();
          for (size_t icp = 0; icp < nCells; ++icp) {
            Short_t cellNumber;
            Double_t amplitude;
            Double_t time;
            Int_t mclabel;
            Double_t efrac;

            cells->GetCell(icp, cellNumber, amplitude, time, mclabel, efrac);
            caloFiller(0, iev, cellNumber, amplitude, time, cellType);
          }

          // Muon Tracks
          size_t nmu = esd->GetNumberOfMuonTracks();
          for (size_t imu = 0; imu < nmu; ++imu) {
            AliESDMuonTrack* mutrk = esd->GetMuonTrack(imu);
            //
            //      TMatrixD cov;
            //      mutrk->GetCovariances(cov);
            //      for (Int_t i = 0; i < 5; i++)
            //        for (Int_t j = 0; j <= i; j++)
            //          fCovariances[i*(i+1)/2 + j] = cov(i,j);
            //      //
            //
            //
            muonFiller(
              0,
              iev,
              mutrk->GetInverseBendingMomentum(),
              mutrk->GetThetaX(),
              mutrk->GetThetaY(),
              mutrk->GetZ(),
              mutrk->GetBendingCoor(),
              mutrk->GetNonBendingCoor(),
              // covariance matrix goes here...
              mutrk->GetChi2(),
              mutrk->GetChi2MatchTrigger());
          }

          // VZERO
          AliESDVZERO* vz = esd->GetVZEROData();
          //for (Int_t ich = 0; ich < 64; ++ich) {
          //  fAdcVZ[ich] = vz->GetAdc(ich);
          //  fTimeVZ[ich] = vz->GetTime(ich);
          //  fWidthVZ[ich] = vz->GetWidth(ich);
          //}
          vzeroFiller(0, iev);
        } // Loop on events

        control.readyToQuit(false);
      }             // stateless callback
      );            // adaptStateless
    }               // stateful lambda
                  ) // adaptStatefull
  };                // algorithmSpec
}

} // namespace run2
} // namespace framework
} // namespace o2
