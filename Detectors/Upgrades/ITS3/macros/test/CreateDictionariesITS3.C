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

/// \file CreateDictionaries.C
/// \brief Macros to test the generation of a dictionary of topologies. Three dictionaries are generated: one with signal-cluster only, one with noise-clusters only and one with all the clusters.

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <string>
#include <unordered_map>

#include <TAxis.h>
#include <TCanvas.h>
#include <TSystem.h>
#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TNtuple.h>
#include <TString.h>
#include <TStyle.h>
#include <TTree.h>
#include <TStopwatch.h>
#include <cmath>

#define ENABLE_UPGRADES
#include "DetectorsCommonDataFormats/DetID.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITS3Base/SpecsV2.h"
#include "ITSMFTBase/SegmentationAlpide.h"
#include "ITS3Base/SegmentationMosaix.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ClusterTopology.h"
#include "ITS3Reconstruction/TopologyDictionary.h"
#include "ITS3Reconstruction/BuildTopologyDictionary.h"
#include "DataFormatsITSMFT/ROFRecord.h"

#include "ITSMFTSimulation/Hit.h"
#include "MathUtils/Cartesian.h"
#include "MathUtils/Utils.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#include "SimulationDataFormat/DigitizationContext.h"
#include "Framework/Logger.h"

#endif

void CreateDictionariesITS3(bool saveDeltas = true,
                            float probThreshold = 1e-6,
                            std::string clusDictFile = "",
                            std::string clusfile = "o2clus_its.root",
                            std::string hitfile = "o2sim_HitsIT3.root",
                            std::string collContextfile = "collisioncontext.root",
                            std::string inputGeom = "",
                            float checkOutliers = 2., // reject outliers (MC dX or dZ exceeds row/col span by a factor above the threshold)
                            float minPtMC = 0.1)      // account only MC hits with pT above threshold
{
  const int QEDSourceID = 99; // Clusters from this MC source correspond to QED electrons

  using namespace o2::base;
  using namespace o2::its;

  using Segmentation = o2::itsmft::SegmentationAlpide;
  using o2::its3::BuildTopologyDictionary;
  using o2::itsmft::ClusterTopology;
  using o2::itsmft::CompCluster;
  using o2::itsmft::CompClusterExt;
  using o2::itsmft::Hit;
  using ROFRec = o2::itsmft::ROFRecord;
  using MC2ROF = o2::itsmft::MC2ROFRecord;
  using HitVec = std::vector<Hit>;
  using MC2HITS_map = std::unordered_map<uint64_t, int>; // maps (track_ID<<16 + chip_ID) to entry in the hit vector
  std::unordered_map<int, int> hadronicMCMap;            // mapping from MC event entry to hadronic event ID
  std::vector<HitVec*> hitVecPool;
  std::vector<MC2HITS_map> mc2hitVec;
  o2::its3::TopologyDictionary clusDictOld;
  std::array<o2::its3::SegmentationMosaix, 3> mMosaixSegmentations{0, 1, 2};
  if (!clusDictFile.empty()) {
    clusDictOld.readFromFile(clusDictFile);
    LOGP(info, "Loaded external cluster dictionary with {} IB/{} OBentries from {}", clusDictOld.getSize(true), clusDictOld.getSize(false), clusDictFile);
  }

  ULong_t cOkIB{0}, cOutliersIB{0}, cFailedMCIB{0};
  ULong_t cOkOB{0}, cOutliersOB{0}, cFailedMCOB{0};

  TFile* fout = nullptr;
  TNtuple* nt = nullptr;
  if (saveDeltas) {
    fout = TFile::Open("CreateDictionaries.root", "recreate");
    nt = new TNtuple("nt", "hashes ntuple", "hash:layer:chipID:xhf:zhf:xcf:zcf:dx:dz:outlimDx:outlimDz:clusterSize:eta");
  }

  const o2::steer::DigitizationContext* digContext = nullptr;
  TStopwatch sw;
  sw.Start();
  float minPtMC2 = minPtMC > 0 ? minPtMC * minPtMC : -1;
  // Geometry
  o2::base::GeometryManager::loadGeometry(inputGeom);
  auto gman = o2::its::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot,
                                                 o2::math_utils::TransformType::L2G)); // request cached transforms

  // Hits
  TFile* fileH = nullptr;
  TTree* hitTree = nullptr;

  if (!hitfile.empty() && !collContextfile.empty() && !gSystem->AccessPathName(hitfile.c_str()) && !gSystem->AccessPathName(collContextfile.c_str())) {
    LOGP(info, "Loading MC information");
    fileH = TFile::Open(hitfile.data());
    hitTree = (TTree*)fileH->Get("o2sim");
    mc2hitVec.resize(hitTree->GetEntries());
    hitVecPool.resize(hitTree->GetEntries(), nullptr);
    digContext = o2::steer::DigitizationContext::loadFromFile(collContextfile);

    auto& intGlo = digContext->getEventParts(digContext->isQEDProvided());
    int hadrID = -1, nGlo = intGlo.size(), nHadro = 0;
    for (int iglo = 0; iglo < nGlo; iglo++) {
      const auto& parts = intGlo[iglo];
      bool found = false;
      for (auto& part : parts) {
        if (part.sourceID == 0) { // we use underlying background
          hadronicMCMap[iglo] = part.entryID;
          found = true;
          nHadro++;
          break;
        }
      }
      if (!found) {
        hadronicMCMap[iglo] = -1;
      }
    }
    if (nHadro < hitTree->GetEntries()) {
      LOG(fatal) << "N=" << nHadro << " hadronic events < "
                 << " N=" << hitTree->GetEntries() << " Hit enties.";
    }
  }

  // Clusters
  TFile* fileCl = TFile::Open(clusfile.data());
  TTree* clusTree = (TTree*)fileCl->Get("o2sim");
  std::vector<CompClusterExt>* clusArr = nullptr;
  clusTree->SetBranchAddress("ITSClusterComp", &clusArr);
  std::vector<unsigned char>* patternsPtr = nullptr;
  auto pattBranch = clusTree->GetBranch("ITSClusterPatt");
  if (pattBranch != nullptr) {
    pattBranch->SetAddress(&patternsPtr);
  }

  // ROFrecords
  std::vector<ROFRec> rofRecVec, *rofRecVecP = &rofRecVec;
  clusTree->SetBranchAddress("ITSClustersROF", &rofRecVecP);

  // Cluster MC labels
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clusLabArr = nullptr;
  std::vector<MC2ROF> mc2rofVec, *mc2rofVecP = &mc2rofVec;
  if (hitTree && clusTree->GetBranch("ITSClusterMCTruth")) {
    clusTree->SetBranchAddress("ITSClusterMCTruth", &clusLabArr);
    clusTree->SetBranchAddress("ITSClustersMC2ROF", &mc2rofVecP);
  }
  clusTree->GetEntry(0);
  if (clusTree->GetEntries() > 1 && !hitfile.empty()) {
    LOGP(error, "Hits are provided but the cluster tree containes {} entries, looks like real data", clusTree->GetEntries());
    return;
  }

  // Topologies dictionaries: 1) all clusters 2) signal clusters only 3) noise clusters only
  BuildTopologyDictionary completeDictionary;
  BuildTopologyDictionary signalDictionary;
  BuildTopologyDictionary noiseDictionary;

  int nROFRec = (int)rofRecVec.size();
  std::vector<int> mcEvMin, mcEvMax;

  if (clusLabArr) { // >> build min and max MC events used by each ROF
    mcEvMin.resize(nROFRec, hitTree->GetEntries());
    mcEvMax.resize(nROFRec, -1);
    for (int imc = mc2rofVec.size(); imc--;) {
      int hadrID = hadronicMCMap[imc];
      if (hadrID < 0) {
        continue;
      }
      const auto& mc2rof = mc2rofVec[imc];
      if (mc2rof.rofRecordID < 0) {
        continue; // this MC event did not contribute to any ROF
      }
      for (int irfd = mc2rof.maxROF - mc2rof.minROF + 1; irfd--;) {
        int irof = mc2rof.rofRecordID + irfd;
        if (mcEvMin[irof] > hadrID) {
          mcEvMin[irof] = hadrID;
        }
        if (mcEvMax[irof] < hadrID) {
          mcEvMax[irof] = hadrID;
        }
      }
    }
  } // << build min and max MC events used by each ROF

  for (Long64_t ient = 0; ient < clusTree->GetEntries(); ient++) {
    clusTree->GetEntry(ient);
    nROFRec = (int)rofRecVec.size();
    LOGP(info, "Processing TF {} with {} ROFs and {} clusters", ient, nROFRec, clusArr->size());
    auto pattIdx = patternsPtr->cbegin();
    for (int irof = 0; irof < nROFRec; irof++) {
      const auto& rofRec = rofRecVec[irof];

      /* rofRec.print(); */

      if (clusLabArr) { // >> read and map MC events contributing to this ROF
        for (int im = mcEvMin[irof]; im <= mcEvMax[irof]; im++) {
          if (!hitVecPool[im]) {
            hitTree->SetBranchAddress("IT3Hit", &hitVecPool[im]);
            hitTree->GetEntry(im);
            auto& mc2hit = mc2hitVec[im];
            const auto* hitArray = hitVecPool[im];
            for (int ih = hitArray->size(); ih--;) {
              const auto& hit = (*hitArray)[ih];
              uint64_t key = (uint64_t(hit.GetTrackID()) << 32) + hit.GetDetectorID();
              mc2hit.emplace(key, ih);
            }
          }
        }
      } // << cache MC events contributing to this ROF

      for (int icl = 0; icl < rofRec.getNEntries(); icl++) {
        int clEntry = rofRec.getFirstEntry() + icl; // entry of icl-th cluster of this ROF in the vector of clusters
        // do we read MC data?

        const auto& cluster = (*clusArr)[clEntry];
        o2::itsmft::ClusterPattern pattern;
        bool ib = o2::its3::constants::detID::isDetITS3(cluster.getChipID());

        if (cluster.getPatternID() != CompCluster::InvalidPatternID) {
          if (clusDictOld.getSize(ib) == 0) {
            LOG(error) << "Encountered patternID = " << cluster.getPatternID() << " != " << CompCluster::InvalidPatternID;
            LOG(error) << "Clusters have already been generated with a dictionary which was not provided";
            return;
          }
          if (clusDictOld.isGroup(cluster.getPatternID(), ib)) {
            pattern.acquirePattern(pattIdx);
          } else {
            pattern = clusDictOld.getPattern(cluster.getPatternID(), ib);
          }
        } else {
          pattern.acquirePattern(pattIdx);
        }
        ClusterTopology topology;
        topology.setPattern(pattern);

        float dX = BuildTopologyDictionary::IgnoreVal, dZ = BuildTopologyDictionary::IgnoreVal;
        if (clusLabArr != nullptr) {
          const auto& lab = (clusLabArr->getLabels(clEntry))[0];
          auto srcID = lab.getSourceID();
          if (lab.isValid()) { // use MC truth info only for non-QED and non-noise clusters
            auto trID = lab.getTrackID();
            const auto& mc2hit = mc2hitVec[lab.getEventID()];
            const auto* hitArray = hitVecPool[lab.getEventID()];
            Int_t chipID = cluster.getSensorID();
            uint64_t key = (uint64_t(trID) << 32) + chipID;
            auto hitEntry = mc2hit.find(key);
            if (hitEntry != mc2hit.end()) {
              const auto& hit = (*hitArray)[hitEntry->second];
              if (minPtMC < 0.f || hit.GetMomentum().Perp2() > minPtMC2) {
                auto xyzLocE = gman->getMatrixL2G(chipID) ^ (hit.GetPos()); // inverse conversion from global to local
                auto xyzLocS = gman->getMatrixL2G(chipID) ^ (hit.GetPosStart());
                o2::math_utils::Vector3D<float> xyzLocM;
                auto locC = o2::its3::TopologyDictionary::getClusterCoordinates(cluster, pattern, false);
                int layer = gman->getLayer(chipID);
                float x0, y0, z0, dltx, dlty, dltz, r;
                if (ib) {
                  float xFlat{0.}, yFlat{0.};
                  mMosaixSegmentations[layer].curvedToFlat(locC.X(), locC.Y(), xFlat, yFlat);
                  locC.SetCoordinates(xFlat, yFlat, locC.Z());
                  mMosaixSegmentations[layer].curvedToFlat(xyzLocE.X(), xyzLocE.Y(), xFlat, yFlat);
                  xyzLocE.SetCoordinates(xFlat, yFlat, xyzLocE.Z());
                  mMosaixSegmentations[layer].curvedToFlat(xyzLocS.X(), xyzLocS.Y(), xFlat, yFlat);
                  xyzLocS.SetCoordinates(xFlat, yFlat, xyzLocS.Z());
                  x0 = xyzLocS.X();
                  dltx = xyzLocE.X() - x0;
                  y0 = xyzLocS.Y();
                  dlty = xyzLocE.Y() - y0;
                  z0 = xyzLocS.Z();
                  dltz = xyzLocE.Z() - z0;
                  r = (o2::its3::constants::pixelarray::pixels::apts::responseYShift - y0) / dlty;
                } else {
                  x0 = xyzLocS.X();
                  dltx = xyzLocE.X() - x0;
                  y0 = xyzLocS.Y();
                  dlty = xyzLocE.Y() - y0;
                  z0 = xyzLocS.Z();
                  dltz = xyzLocE.Z() - z0;
                  r = (0.5 * (Segmentation::SensorLayerThickness - Segmentation::SensorLayerThicknessEff) - y0) / dlty;
                }
                xyzLocM.SetXYZ(x0 + r * dltx, y0 + r * dlty, z0 + r * dltz);

                auto pitchX = (ib) ? o2::its3::SegmentationMosaix::PitchRow : o2::itsmft::SegmentationAlpide::PitchRow;
                auto pitchZ = (ib) ? o2::its3::SegmentationMosaix::PitchCol : o2::itsmft::SegmentationAlpide::PitchCol;
                dX = xyzLocM.X() - locC.X();
                dZ = xyzLocM.Z() - locC.Z();

                float outLimitDx{-1}, outLimitDz{-1};
                if (checkOutliers > 0.) {
                  outLimitDx = topology.getRowSpan() * checkOutliers * pitchX;
                  outLimitDz = topology.getColumnSpan() * checkOutliers * pitchZ;
                  bool isOutDx = std::abs(dX) > outLimitDx;
                  bool isOutDz = std::abs(dZ) > outLimitDz;
                  if (isOutDx || isOutDz) { // ignore outlier
                    (ib) ? ++cOutliersIB : ++cOutliersOB;
                    LOGP(debug, "Ignored Value dX={} > {} * {} -> {}", dX, topology.getRowSpan(), checkOutliers, isOutDx);
                    LOGP(debug, "Ignored Value dZ={} > {} * {} -> {}", dZ, topology.getColumnSpan(), checkOutliers, isOutDz);
                    dX = dZ = BuildTopologyDictionary::IgnoreVal;
                  } else {
                    (ib) ? ++cOkIB : ++cOkOB;
                  }
                }
                if (saveDeltas) {
                  auto vectDiff = xyzLocE - xyzLocS;
                  auto theta = std::acos(vectDiff.Z() / std::hypot(vectDiff.X(), vectDiff.Y(), vectDiff.Z()));
                  auto eta = -std::log(std::tan(theta / 2));
                  if (ib) {
                    LOGP(info, "Yhit flat start: {}, end: {}, middle: {}", xyzLocS.Y(), xyzLocE.Y(), xyzLocM.Y());
                  }
                  nt->Fill(topology.getHash(), layer, chipID, xyzLocM.X(), xyzLocM.Z(), locC.X(), locC.Z(), dX, dZ, outLimitDx, outLimitDz, pattern.getNPixels(), eta);
                }
              }
            } else {
              /* LOGP(info, "  Failed to find MC hit entry for Tr: {} chipID: {}", trID, chipID); */
              /* lab.print(); */
              (ib) ? ++cFailedMCIB : ++cFailedMCOB;
            }
            signalDictionary.accountTopology(topology, ib, dX, dZ);
          } else {
            noiseDictionary.accountTopology(topology, ib, dX, dZ);
          }
        }
        completeDictionary.accountTopology(topology, ib, dX, dZ);
      }

      // clean MC cache for events which are not needed anymore
      if (clusLabArr) {
        int irfNext = irof;
        int limMC = irfNext == nROFRec ? hitVecPool.size() : mcEvMin[irfNext]; // can clean events up to this
        for (int imc = mcEvMin[irof]; imc < limMC; imc++) {
          delete hitVecPool[imc];
          hitVecPool[imc] = nullptr;
          mc2hitVec[imc].clear();
        }
      }
    }
  }

  LOGP(info, "IB Clusters: {} okay (failed MCHit2Clus {}); outliers {}", cOkIB, cFailedMCIB, cOutliersIB);
  LOGP(info, "OB Clusters: {} okay (failed MCHit2Clus {}); outliers {}", cOkOB, cFailedMCOB, cOutliersOB);

  auto dID = o2::detectors::DetID::IT3;

  LOGP(info, "Complete Dictionary:");
  completeDictionary.setThreshold(probThreshold, true);
  completeDictionary.setThreshold(probThreshold, false);
  completeDictionary.groupRareTopologies();
  completeDictionary.printDictionaryBinary(o2::base::DetectorNameConf::getAlpideClusterDictionaryFileName(dID, ""));
  completeDictionary.printDictionary(o2::base::DetectorNameConf::getAlpideClusterDictionaryFileName(dID, "", "txt"));
  completeDictionary.saveDictionaryRoot(o2::base::DetectorNameConf::getAlpideClusterDictionaryFileName(dID, "", "root"));

  TFile histogramOutput("histograms.root", "recreate");
  TCanvas* cComplete = new TCanvas("cComplete", "Distribution of all the topologies");
  cComplete->Divide(2, 1);
  cComplete->cd(1);
  TH1F* hCompleteIB = completeDictionary.getDictionary().getTopologyDistribution("hCompleteInnerBarrel", true);
  hCompleteIB->SetDirectory(nullptr);
  hCompleteIB->Draw("hist");
  gPad->SetLogy();
  cComplete->cd(2);
  TH1F* hCompleteOB = completeDictionary.getDictionary().getTopologyDistribution("hCompleteOuterBarrel", false);
  hCompleteOB->SetDirectory(nullptr);
  hCompleteOB->Draw("hist");
  gPad->SetLogy();
  histogramOutput.cd();
  hCompleteIB->Write();
  hCompleteOB->Write();
  cComplete->Write();

  if (clusLabArr) {
    LOGP(info, "Noise Dictionary:");
    noiseDictionary.setThreshold(0.0001, true);
    noiseDictionary.setThreshold(0.0001, false);
    noiseDictionary.groupRareTopologies();
    noiseDictionary.printDictionaryBinary(o2::base::DetectorNameConf::getAlpideClusterDictionaryFileName(dID, "noiseClusTopo"));
    noiseDictionary.printDictionary(o2::base::DetectorNameConf::getAlpideClusterDictionaryFileName(dID, "noiseClusTopo", "txt"));
    noiseDictionary.saveDictionaryRoot(o2::base::DetectorNameConf::getAlpideClusterDictionaryFileName(dID, "noiseClusTopo", "root"));

    LOGP(info, "Signal Dictionary:");
    signalDictionary.setThreshold(0.0001, true);
    signalDictionary.setThreshold(0.0001, false);
    signalDictionary.groupRareTopologies();
    signalDictionary.printDictionaryBinary(o2::base::DetectorNameConf::getAlpideClusterDictionaryFileName(dID, "signal"));
    signalDictionary.printDictionary(o2::base::DetectorNameConf::getAlpideClusterDictionaryFileName(dID, "signal", "txt"));
    signalDictionary.saveDictionaryRoot(o2::base::DetectorNameConf::getAlpideClusterDictionaryFileName(dID, "signal", "root"));

    LOGP(info, "Plotting Channels");
    auto cNoise = new TCanvas("cNoise", "Distribution of noise topologies");
    cNoise->Divide(2, 1);
    cNoise->cd(1);
    auto hNoiseIB = noiseDictionary.getDictionary().getTopologyDistribution("hNoiseInnerBarrel", true);
    hNoiseIB->SetDirectory(nullptr);
    hNoiseIB->Draw("hist");
    gPad->SetLogy();
    cNoise->cd(2);
    auto hNoiseOB = noiseDictionary.getDictionary().getTopologyDistribution("hNoiseOuterBarrel", false);
    hNoiseOB->SetDirectory(nullptr);
    hNoiseOB->Draw("hist");
    gPad->SetLogy();
    histogramOutput.cd();
    hNoiseIB->Write();
    hNoiseOB->Write();
    cNoise->Write();

    auto cSignal = new TCanvas("cSignal", "cSignal");
    cSignal->Divide(2, 1);
    cSignal->cd(1);
    auto hSignalIB = signalDictionary.getDictionary().getTopologyDistribution("hSignalInnerBarrel", true);
    hSignalIB->SetDirectory(nullptr);
    hSignalIB->Draw("hist");
    gPad->SetLogy();
    cSignal->cd(2);
    cSignal->SetLogy();
    auto hSignalOB = signalDictionary.getDictionary().getTopologyDistribution("hSignalOuterBarrel", false);
    hSignalOB->SetDirectory(nullptr);
    hSignalOB->Draw("hist");
    gPad->SetLogy();
    histogramOutput.cd();
    hSignalIB->Write();
    hSignalOB->Write();
    cSignal->Write();
  }
  sw.Stop();
  sw.Print();
  if (saveDeltas) {
    fout->cd();
    nt->Write();
  }
}
