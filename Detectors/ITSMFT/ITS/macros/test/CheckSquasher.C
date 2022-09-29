#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <DataFormatsITSMFT/ROFRecord.h>
#include <DataFormatsITSMFT/CompCluster.h>
#include <ITSBase/GeometryTGeo.h>
#include <Framework/Logger.h>
#include <DataFormatsITSMFT/TopologyDictionary.h>
#include <DetectorsCommonDataFormats/DetectorNameConf.h>
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/MCTruthContainer.h>

#include <TFile.h>
#include <TTree.h>
#include <TH2D.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TColor.h>

#include <vector>
#include <gsl/gsl>
#endif

void CheckSquasher(const uint chipId = 0, const uint startingROF = 0, const unsigned int nRofs = 3)
{
  TColor::InvertPalette();
  gStyle->SetOptStat(0);
  // Geometry
  o2::base::GeometryManager::loadGeometry("o2sim_geometry.root");
  auto gman = o2::its::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));
  // Topology dictionary
  o2::itsmft::TopologyDictionary mdict;
  mdict.readFromFile(o2::base::DetectorNameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::ITS, ""));

  auto fITSclus = TFile::Open("o2clus_its.root", "r");
  auto treeITSclus = (TTree*)fITSclus->Get("o2sim");

  std::vector<o2::itsmft::CompClusterExt>* ITSclus = nullptr;
  std::vector<o2::itsmft::ROFRecord>* ITSrof = nullptr;
  std::vector<unsigned char>* ITSpatt = nullptr;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clusLabArr = nullptr;

  treeITSclus->SetBranchAddress("ITSClusterComp", &ITSclus);
  treeITSclus->SetBranchAddress("ITSClustersROF", &ITSrof);
  treeITSclus->SetBranchAddress("ITSClusterPatt", &ITSpatt);
  treeITSclus->SetBranchAddress("ITSClusterMCTruth", &clusLabArr);

  auto clSpan = gsl::span(ITSclus->data(), ITSclus->size());
  std::vector<TH2D*> hHitMapsVsFrame(nRofs);

  treeITSclus->GetEvent(0);
  LOGP(info, "there are {} rofs in this TF", ITSrof->size());
  for (unsigned int iR{0}; iR < nRofs; iR++) {
    LOGP(info, "Processing rof {}", iR + startingROF);
    hHitMapsVsFrame[iR] = new TH2D(Form("chip%i_rof%i", chipId, startingROF + iR), Form("chip %i rof%i; ; ; Counts", chipId, startingROF + iR), 1024, 0, 1024, 512, 0, 512);

    // work on data
    const auto& rof = (*ITSrof)[startingROF + iR];
    auto clustersInFrame = rof.getROFData(*ITSclus);
    auto pattIt = ITSpatt->cbegin();

    for (unsigned int clusInd{0}; clusInd < clustersInFrame.size(); clusInd++) {
      const auto& clus = clustersInFrame[clusInd];
      o2::itsmft::ClusterPattern patt;
      auto sID = clus.getSensorID();
      if (sID == chipId) {
        auto pattID = clus.getPatternID();
        int npix;
        if (pattID == o2::itsmft::CompCluster::InvalidPatternID || mdict.isGroup(pattID)) {
          patt.acquirePattern(pattIt);
          npix = patt.getNPixels();
        } else {
          npix = mdict.getNpixels(pattID);
          patt = mdict.getPattern(pattID);
        }
        auto labels = clusLabArr->getLabels(clusInd);
        // extract pattern info
        auto col = clus.getCol();
        auto row = clus.getRow();

        LOGP(info, "row: {} col: {} cluster size is: {}, labels are:", row, col, npix);
        std::cout << "\t\t\t\t\t";
        for (auto l : labels) {
          std::cout << l << "\t";
        }

        std::cout << std::endl;
        int ic = 0, ir = 0;

        auto colSpan = patt.getColumnSpan();
        auto rowSpan = patt.getRowSpan();
        auto nBits = rowSpan * colSpan;

        for (int i = 2; i < patt.getUsedBytes() + 2; i++) {
          unsigned char tempChar = patt.getByte(i);
          int s = 128; // 0b10000000
          while (s > 0) {
            if ((tempChar & s) != 0) // checking active pixels
            {
              hHitMapsVsFrame[iR]->Fill(col + ic, row + rowSpan - ir);
            }
            ic++;
            s >>= 1;
            if ((ir + 1) * ic == nBits) {
              break;
            }
            if (ic == colSpan) {
              ic = 0;
              ir++;
            }
            if ((ir + 1) * ic == nBits) {
              break;
            }
          }
        }
      }
    }
  }
  auto canvas = new TCanvas(Form("chip%d", chipId), Form("chip%d", chipId), nRofs * 1000, 600);

  canvas->Divide(nRofs, 1);
  for (unsigned int i{0}; i < nRofs; ++i) {
    canvas->cd(i + 1);
    hHitMapsVsFrame[i]->Draw("colz");
  }
}