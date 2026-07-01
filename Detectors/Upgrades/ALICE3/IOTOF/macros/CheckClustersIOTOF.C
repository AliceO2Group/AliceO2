// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CheckClustersIOTOF.C
/// \brief Simple macro to create clusters from TF3 digits

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TCanvas.h>
#include <TFile.h>
#include <TH1F.h>
#include <TNtuple.h>
#include <TTree.h>
#include <TStyle.h>

#include "IOTOFSimulation/Segmentation.h"
#include "IOTOFBase/IOTOFBaseParam.h"
#include "IOTOFBase/GeometryTGeo.h"
#include "DataFormatsIOTOF/Digit.h"
#include "DataFormatsIOTOF/Cluster.h"
#include "MathUtils/Utils.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/IOMCTruthContainerView.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DetectorsBase/GeometryManager.h"

#include "DataFormatsITSMFT/ROFRecord.h"

#endif

#define ENABLE_UPGRADES

void CheckClustersIOTOF(std::string digiFilePath = "tf3digits.root", std::string clsFilePath = "tf3clusters.root", std::string inputGeomPath = "o2sim_geometry.root")
{
  gStyle->SetPalette(55);

  using namespace o2::base;
  using namespace o2::iotof;

  using o2::iotof::Cluster;
  using o2::iotof::Digit;

  o2::conf::ConfigurableParam::updateFromString("IOTOFBase.segmentedInnerTOF=true;IOTOFBase.segmentedOuterTOF=true;IOTOFBase.enableForwardTOF=false;IOTOFBase.enableBackwardTOF=false");

  auto segGeom = o2::iotof::Segmentation::Instance();

  // Geometry
  o2::base::GeometryManager::loadGeometry(inputGeomPath);
  auto* tofGeo = o2::iotof::GeometryTGeo::Instance();
  tofGeo->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G));

  // Digits
  TFile* digiFile = TFile::Open(digiFilePath.data());
  TTree* digiTree = (TTree*)digiFile->Get("o2sim");
  std::vector<o2::iotof::Digit>* digitsArray{nullptr};
  digiTree->SetBranchAddress("TF3Digit", &digitsArray);
  std::vector<o2::itsmft::ROFRecord>* digiRofRecordsArr{nullptr};
  digiTree->SetBranchAddress("TF3DigitROF", &digiRofRecordsArr);
  auto& digiRofArr = *digiRofRecordsArr;
  o2::dataformats::IOMCTruthContainerView* digiLabelsArr{nullptr};
  digiTree->SetBranchAddress("TF3DigitMCTruth", &digiLabelsArr);
  digiTree->GetEntry(0);
  o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel> digiLabels;
  digiLabelsArr->copyandflatten(digiLabels);

  // Clusters
  TFile* clsFile = TFile::Open(clsFilePath.data());
  TTree* clsTree = (TTree*)clsFile->Get("o2sim");
  std::vector<o2::iotof::Cluster>* clsArray{nullptr};
  clsTree->SetBranchAddress("TF3ClusterComp", &clsArray);
  std::vector<o2::itsmft::ROFRecord>* clsRofRecordsArr{nullptr};
  clsTree->SetBranchAddress("TF3ClusterROF", &clsRofRecordsArr);
  auto& clsRofArr = *clsRofRecordsArr;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clsLabels{nullptr};
  clsTree->SetBranchAddress("TF3ClusterMCTruth", &clsLabels);
  clsTree->GetEntry(0);

  // Summary of entries in all branches
  std::cout << std::endl;
  std::cout << "---> Number of digits: " << digitsArray->size() << std::endl;
  std::cout << "---> Number of digit ROFs: " << digiRofArr.size() << std::endl;
  std::cout << "---> Number of clusters: " << clsArray->size() << std::endl;
  std::cout << "---> Number of cluster ROFs: " << clsRofArr.size() << std::endl;
  std::cout << "---> Number of digits with MC label: " << digiLabels.getNElements() << std::endl;
  std::cout << "---> Number of digits with MC label: " << digiLabels.getIndexedSize() << std::endl;
  std::cout << "---> Number of clusters with MC label: " << clsLabels->getNElements() << std::endl;
  std::cout << "---> Number of clusters with MC label: " << clsLabels->getIndexedSize() << std::endl;
  std::cout << std::endl;

  auto clsTuple = new TNtuple("clsTuple", "clsTuple", "chip_id:x:y:z:row:col:time");
  clsTuple->SetDirectory(nullptr);

  TH1F* histXCoordCls = new TH1F("histXCoordCls", "histXCoordCls", 8000, -100, 100);
  TH1F* histYCoordCls = new TH1F("histYCoordCls", "histYCoordCls", 8000, -100, 100);
  TH1F* histZCoordCls = new TH1F("histZCoordCls", "histZCoordCls", 28000, -400, 400);
  TH1F* histXCoordDigit = new TH1F("histXCoordDigit", "histXCoordDigit", 8000, -100, 100);
  TH1F* histYCoordDigit = new TH1F("histYCoordDigit", "histYCoordDigit", 8000, -100, 100);
  TH1F* histZCoordDigit = new TH1F("histZCoordDigit", "histZCoordDigit", 28000, -400, 400);
  TH1F* histXCoordRes = new TH1F("histXCoordRes", "histXCoordRes", 100, -0.05, 0.05);
  TH1F* histYCoordRes = new TH1F("histYCoordRes", "histYCoordRes", 100, -0.05, 0.05);
  TH1F* histZCoordRes = new TH1F("histZCoordRes", "histZCoordRes", 100, -0.05, 0.05);
  TH1F* histTimeRes = new TH1F("histTimeRes", "histTimeRes", 100, -0.05, 0.05);

  // Load all digits upfront and build a lookup map
  int nDigits = digiTree->GetEntries();
  std::unordered_map<o2::MCCompLabel, int> digitsLabels;
  for (int iDigit = 0; iDigit < digitsArray->size(); ++iDigit) {
    auto label = digiLabels.getLabels(iDigit)[0];
    if (!label.isValid()) {
      continue;
    }
    digitsLabels.emplace(label, iDigit);
  }

  // LOOP on : ROFRecord array
  for (unsigned int iROF = 0; iROF < clsRofArr.size(); ++iROF) {

    const unsigned int rofIndex = clsRofArr[iROF].getFirstEntry();
    const unsigned int rofNEntries = clsRofArr[iROF].getNEntries();

    // LOOP on : digits array
    std::cout << "\n\n ----> Starting loop on digits for ROF " << iROF << " with index " << rofIndex << " and nEntries " << rofNEntries << std::endl;
    for (unsigned int iDigit = rofIndex; iDigit < rofIndex + rofNEntries; iDigit++) {
      if (iDigit % 10000 == 0) {
        std::cout << "Reading digit " << iDigit << " / " << digitsArray->size() << std::endl;
      }

      Int_t iRow = (*digitsArray)[iDigit].getRow();
      Int_t iCol = (*digitsArray)[iDigit].getColumn();
      Int_t iDetID = (*digitsArray)[iDigit].getChipIndex();
      Int_t chipID = (*digitsArray)[iDigit].getChipIndex();
      Int_t subDetID = tofGeo->getIOTOFLayer(iDetID);

      Float_t x{0.f}, y{0.f}, z{0.f};
      if (subDetID >= 0) {
        segGeom->detectorToLocal(iRow, iCol, x, z, subDetID);
      }

      o2::math_utils::Point3D<float> localDigitCoord(x, y, z); // local Digit

      const auto globalDigitCoord = tofGeo->getMatrixL2G(chipID)(localDigitCoord); // convert to global
      histXCoordDigit->Fill(globalDigitCoord.X());
      histYCoordDigit->Fill(globalDigitCoord.Y());
      histZCoordDigit->Fill(globalDigitCoord.Z());
    } // end loop on digits array

    // LOOP on : clusters array
    std::cout << "\n\n ----> Starting loop on clusters for ROF " << iROF << " with index " << rofIndex << " and nEntries " << rofNEntries << std::endl;
    for (unsigned int iCls = rofIndex; iCls < rofIndex + rofNEntries; iCls++) {
      if (iCls % 10000 == 0) {
        std::cout << "Reading cluster " << iCls << " / " << clsArray->size() << std::endl;
      }

      Int_t iRow = (*clsArray)[iCls].row;
      Int_t iCol = (*clsArray)[iCls].col;
      Int_t chipID = (*clsArray)[iCls].chipID;
      Int_t subDetID = tofGeo->getIOTOFLayer(chipID);
      Float_t time = (*clsArray)[iCls].time;

      Float_t x = 0.f, y = 0.f, z = 0.f;
      if (subDetID >= 0) {
        segGeom->detectorToLocal(iRow, iCol, x, z, subDetID);
      }

      o2::math_utils::Point3D<float> localClsCoords(x, y, z);                    // local Digit
      const auto globalClsCoords = tofGeo->getMatrixL2G(chipID)(localClsCoords); // convert to global
      clsTuple->Fill((*clsArray)[iCls].chipID,
                     globalClsCoords.x(),
                     globalClsCoords.y(),
                     globalClsCoords.z(),
                     (*clsArray)[iCls].row,
                     (*clsArray)[iCls].col,
                     (*clsArray)[iCls].time);
      histXCoordCls->Fill(globalClsCoords.x());
      histYCoordCls->Fill(globalClsCoords.y());
      histZCoordCls->Fill(globalClsCoords.z());

      // Match to digit
      auto digitLabelFromCls = (clsLabels->getLabels(iCls))[0];
      auto digitEntry = digitsLabels.find(digitLabelFromCls);

      if (digitEntry == digitsLabels.end()) {
        LOG(error) << "No matching digit for cluster " << iCls << " with label " << digitLabelFromCls.getRawValue();
        continue;
      }

      int iDigit = digitEntry->second;
      Int_t iRowFromDigit = (*digitsArray)[iDigit].getRow();
      Int_t iColFromDigit = (*digitsArray)[iDigit].getColumn();
      Int_t iChipIDFromDigit = (*digitsArray)[iDigit].getChipIndex();
      Int_t iSubDetIDFromDigit = tofGeo->getIOTOFLayer(iChipIDFromDigit);
      Float_t timeFromDigit = (*digitsArray)[iDigit].getTime();

      float xFromDigit = 0.f, yFromDigit = 0.f, zFromDigit = 0.f;
      if (iSubDetIDFromDigit >= 0) {
        segGeom->detectorToLocal(iRowFromDigit, iColFromDigit, xFromDigit, zFromDigit, iSubDetIDFromDigit);
      }

      o2::math_utils::Point3D<float> localDigitCoordFromDigit(xFromDigit, yFromDigit, zFromDigit);             // local Digit
      const auto globalDigitCoordFromDigit = tofGeo->getMatrixL2G(iChipIDFromDigit)(localDigitCoordFromDigit); // convert to global
      histXCoordRes->Fill(globalClsCoords.x() - globalDigitCoordFromDigit.X());
      histYCoordRes->Fill(globalClsCoords.y() - globalDigitCoordFromDigit.Y());
      histZCoordRes->Fill(globalClsCoords.z() - globalDigitCoordFromDigit.Z());
      histTimeRes->Fill(time - timeFromDigit);
    } // end loop on clusters array
  } // end loop on ROFRecords

  std::cout << "Cluster array size: " << clsTuple->GetEntries() << std::endl;

  // cluster maps in the xy and yz planes
  auto canvXY = new TCanvas("canvXY", "", 1600, 800);
  canvXY->Divide(2, 1);
  canvXY->cd(1);
  clsTuple->Draw("y:x>>h_y_vs_x_IOTOF(1000, -100, 100, 1000, -100, 100)", "", "colz");
  canvXY->cd(2);
  clsTuple->Draw("y:z>>h_y_vs_z_IOTOF(1000, -400, 400, 1000, -100, 100)", "", "colz");
  canvXY->SaveAs("clusters_digits_y_vs_x_vs_z.pdf");

  // z distributions
  auto canvZ = new TCanvas("canvZ", "", 800, 800);
  canvZ->cd();
  clsTuple->Draw("z>>h_z_IOTOF(500, -70, 70)", "");
  canvZ->SaveAs("clusters_digits_z.pdf");

  TFile* outFile = new TFile("CheckClusters.root", "RECREATE");
  // Save all columns of the tuple as hists
  clsTuple->Write();
  histXCoordCls->Write();
  histYCoordCls->Write();
  histZCoordCls->Write();
  histXCoordDigit->Write();
  histYCoordDigit->Write();
  histZCoordDigit->Write();
  histXCoordRes->Write();
  histYCoordRes->Write();
  histZCoordRes->Write();
  histTimeRes->Write();
  outFile->Write();
  outFile->Close();
}
