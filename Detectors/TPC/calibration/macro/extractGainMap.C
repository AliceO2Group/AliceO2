// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/ClusterNativeHelper.h"
#include "TPCCalibration/CalibPadGainTracks.h"
//MC includes
#include "SimulationDataFormat/MCTrack.h"
//ROOT
#include "TF1.h"
#include "TLine.h"
#include "TStyle.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TPCBase/Sector.h"
#include "TPCBase/CalDet.h"
#include "TPCBase/PadPos.h"
#include "TPCBase/ROC.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/PadRegionInfo.h"
#include "DataFormatsTPC/Defs.h"
#endif

// this mapper instance is just a dummy for the possiblity of interpreting the macro
const auto& gmapper = o2::tpc::Mapper::instance();

/// this function creates the gainmap and make plots for each sector and a nice polar plot of the pad by pad gain map. tpctracks.root and tpc-native-clusters.root must exist
void processPadGainCalib(const float momMin = 0, const float momMax = 10);

/// run CalibPadGainTracks method with "tpctracks.root" and "tpc-native-clusters.root"
void calibGainMacroTrk(const float momMin = 0, const float momMax = 100, const std::string_view fileNameTracks = "tpctracks.root", const std::string_view fileNameCluster = "tpc-native-clusters.root");

// Drawing functions
/// this functions compares the extracted map with some reference gainmap
/// \param folder output folder for plots
/// \param refMap name of reference gainmap
/// \param outFileType file type of output
void compareMaps(std::string gainmapfile = "GainMap.root", std::string refMap = "RefGainMap.root", std::string gainmapMapName = "GainMap", std::string refMapName = "Gain", std::string outFileType = "pdf");

/// this functions plots the reference map
/// \param folder output folder for plots
/// \param outFileType file type of output
void plotExtractedGainmap(std::string gainmapfile = "GainMap.root", std::string outFileType = "pdf");

//string folder, std::string refMap, std::string outFileType
void makePolarPlot(std::string gainmapfile = "GainMap.root", std::string outFileType = "jpg");

// set logarithmic color palette
void setLogColorPalette();

/// draw the gainmap in polarcoordinates
void drawTPCPolar(TH2F& tpc_Side, o2::tpc::Side side, o2::tpc::CalPad& gainMap);

void drawTicks();

o2::tpc::CalPad loadGainMap(std::string gainmapfile = "GainMap.root", std::string gainMapName = "GainMap");

// this function is just for plotting
void vhcolorTH1(TH1& hist, TString xtitle = "xaxis", TString ytitle = "yaxis", int isize = 14, float xOff = 1, float yOff = 1);
void vhcolorTH2(TH2& hist, TString xtitle = "xaxis", TString ytitle = "yaxis", int isize = 14, float xOff = 1, float yOff = 1);
void setgpad(float tM = 0.03f, float rM = 0.02f, float bM = 0.12f, float lM = 0.12f);
void setleg(TLegend& leg, int isize = 34);
std::array<std::array<std::vector<double>, 10>, 2> getBinning(); // get the correct binning for IROC and OROC

void processPadGainCalib(const float momMin, const float momMax)
{
  // run CalibPadGainTracks method with "tpctracks.root" and "tpc-native-clusters.root"
  calibGainMacroTrk(momMin, momMax);
  // create sector by sector plots of the extracted gain map
  plotExtractedGainmap();
  // make a nice polar plot of the whole A side and C side
  makePolarPlot();
}

void calibGainMacroTrk(const float momMin, const float momMax, const std::string_view fileNameTracks, const std::string_view fileNameCluster)
{
  using namespace o2::tpc;
  // this track object is just a dummy for the possiblity of interpreting the macro
  TrackTPC tdummy;
  TFile file(fileNameTracks.data());
  if (file.IsZombie()) {
    std::cout << "Error getting file\n";
    return;
  }
  auto tree = (TTree*)file.Get("tpcrec");
  if (tree == nullptr) {
    std::cout << "Error getting tree\n";
    return;
  }
  std::vector<TrackTPC>* tpcTracks{nullptr};
  std::vector<TPCClRefElem>* tpcTrackClIdxVecInput{nullptr};
  tree->SetBranchAddress("TPCTracks", &tpcTracks);
  tree->SetBranchAddress("ClusRefs", &tpcTrackClIdxVecInput);

  ClusterNativeHelper::Reader tpcClusterReader{};
  tpcClusterReader.init(fileNameCluster.data());

  ClusterNativeAccess clusterIndex{};
  std::unique_ptr<ClusterNative[]> clusterBuffer{};
  o2::tpc::ClusterNativeHelper::ConstMCLabelContainerViewWithBuffer clusterMCBuffer;
  memset(&clusterIndex, 0, sizeof(clusterIndex));

  CalibPadGainTracks cGain{};
  cGain.init(20, 0, 3, 1, 1); // use 20 bins, minimum x=0, maximum x=10, use underflow and overflow bin
  const long long maxEvent = tree->GetEntriesFast();
  for (unsigned int iEvent = 0; iEvent < maxEvent; ++iEvent) {
    std::cout << "Event " << iEvent << " of " << maxEvent << "\n";
    tree->GetEntry(iEvent);
    tpcClusterReader.read(iEvent);
    tpcClusterReader.fillIndex(clusterIndex, clusterBuffer, clusterMCBuffer);
    cGain.setMembers(tpcTracks, tpcTrackClIdxVecInput, clusterIndex);
    cGain.processTracks(true, momMin, momMax);
  }
  cGain.fillgainMap();
  cGain.dumpGainMap();
}

void plotExtractedGainmap(std::string gainmapfile, std::string outFileType)
{
  using namespace o2::tpc;
  // this function plots the extracted gain map
  CalPad gainMap = loadGainMap(gainmapfile);

  // get the correct binning for each roc
  std::array<std::array<std::vector<double>, 10>, 2> binPads = getBinning(); // 10 different pad sizes (width and length)
  std::array<std::array<TH2F, binPads[0].size()>, 36> h_Region_Gain{};       // 36 sectors

  for (unsigned int iSec = 0; iSec < h_Region_Gain.size(); ++iSec) {
    for (unsigned int iReg = 0; iReg < binPads[0].size(); iReg++) {
      h_Region_Gain[iSec][iReg].SetBins(static_cast<int>(binPads[1][iReg].size() - 1), binPads[1][iReg].data(), static_cast<int>(binPads[0][iReg].size() - 1), binPads[0][iReg].data());
    }
  }

  //fill the histograms
  for (size_t iSec = 0; iSec < 72; iSec++) {
    ROC rocTemp(static_cast<unsigned char>(iSec)); // only this region!!! 0-72
    ROC rocIROC(0);

    const auto& mapper = Mapper::instance();
    const int nrows = mapper.getNumberOfRowsROC(rocTemp);
    const int nrowsIROC = mapper.getNumberOfRowsROC(rocIROC);
    for (int irow = 0; irow < nrows; ++irow) {
      int currRow = nrowsIROC + irow;
      if (iSec < 36) {
        currRow = irow;
      }

      const int npads = mapper.getNumberOfPadsInRowROC(rocTemp, irow);
      for (int ipad = 0; ipad < npads; ++ipad) {
        const auto valNew = gainMap.getValue(rocTemp, static_cast<unsigned int>(irow), static_cast<unsigned int>(ipad));

        unsigned int iCurrentRegion = 0;
        int iPadRowRegion = 0;
        //get the padlength and padwidth for current region
        for (unsigned char iRegion = 0; iRegion < 10; iRegion++) {
          const PadRegionInfo& region = mapper.getPadRegionInfo(iRegion);
          iPadRowRegion += (int)region.getNumberOfPadRows();
          if ((currRow) < iPadRowRegion)
            break;
          iCurrentRegion++;
        }

        const float xPosition = mapper.getPadCentre(PadPos(static_cast<unsigned char>(currRow), static_cast<unsigned char>(ipad))).X();
        const float yPosition = mapper.getPadCentre(PadPos(static_cast<unsigned char>(currRow), static_cast<unsigned char>(ipad))).Y();

        if (iSec >= 36) {
          h_Region_Gain[iSec - 36][iCurrentRegion].Fill(xPosition, yPosition, valNew);
        } else {
          h_Region_Gain[iSec][iCurrentRegion].Fill(xPosition, yPosition, valNew);
        }
      }
    }
  }
  //===============================Drawing ================================
  TCanvas canv("canv", "canv", 900, 600);
  canv.cd();
  setgpad(0.03f, 0.2f, 0.14f, 0.14f);
  std::string sMap = "Extracted";
  for (unsigned int iSec = 0; iSec < h_Region_Gain.size(); ++iSec) {
    TH2F hDUMMY_Gain("hDUMMY_Gain", "hDUMMY_Gain", 153, 80, 250, 90, -45, 45);
    vhcolorTH2(hDUMMY_Gain, "#it{x} (cm)", "#it{y} (cm)", 36, .8f, .8f);
    hDUMMY_Gain.GetZaxis()->SetRangeUser(0., 1.3);
    hDUMMY_Gain.GetYaxis()->SetNdivisions(5, 5, 0);
    hDUMMY_Gain.GetXaxis()->SetNdivisions(5, 5, 0);
    hDUMMY_Gain.Draw("");
    for (unsigned int iROC = 0; iROC < 10; iROC++) {

      h_Region_Gain[iSec][iROC].SetMaximum(1.4);
      if (iROC == 0) {
        vhcolorTH2(h_Region_Gain[iSec][iROC], "", "", 36, .8f, .8f);
        const std::string zTitle = "rel. gain extracted";
        h_Region_Gain[iSec][iROC].GetZaxis()->SetTitle(zTitle.data());
        h_Region_Gain[iSec][iROC].Draw("colzsame");
      } else {
        h_Region_Gain[iSec][iROC].Draw("colsame");
      }
    }
    canv.SaveAs(Form("Gainmap_Sector_%s_%d.%s", sMap.data(), iSec, outFileType.data()));
  }
}

void makePolarPlot(std::string gainmapfile, std::string outFileType)
{
  using namespace o2::tpc;
  CalPad gainMap = loadGainMap(gainmapfile);

  TCanvas canv("canv", "canv", 900, 600);
  canv.cd();
  setgpad(0.03f, 0.2f, 0.14f, 0.16f);
  std::array<TH2F, 2> htpc_Side;
  TH1F hDummy("dummy", "dummy", 10, -300, 300);
  hDummy.SetMinimum(-300);
  hDummy.SetMaximum(300);
  vhcolorTH1(hDummy, "#it{x} (cm)", "#it{y} (cm)", 36, 1, 1.2f);
  hDummy.SetLineWidth(0);
  hDummy.GetXaxis()->SetNdivisions(5, 5, 0);
  hDummy.GetYaxis()->SetNdivisions(5, 5, 0);
  setLogColorPalette();

  for (unsigned int iSide = 0; iSide < 2; ++iSide) {
    const std::array<Side, 2> side = {Side::A, Side::C};
    const std::array<char, 2> side_name = {'A', 'C'};

    htpc_Side[iSide].SetBins(800, -M_PI, M_PI, 120, 80, 260);
    vhcolorTH2(htpc_Side[iSide], "#it{x} (cm)", "#it{y} (cm)", 36, 1, 1);
    htpc_Side[iSide].GetZaxis()->SetTitle("rel. gain extracted");

    drawTPCPolar(htpc_Side[iSide], side[iSide], gainMap);

    gStyle->SetLineWidth(0);
    hDummy.Draw();
    htpc_Side[iSide].GetZaxis()->SetRangeUser(0, 1.4);
    htpc_Side[iSide].DrawCopy("same colz pol");
    drawTicks();
    canv.SaveAs(Form("TPC_%s_Map_Side_%c.%s", "Extracted", side_name[iSide], outFileType.data()));
  }

  gStyle->SetLineWidth(1);
  TColor::SetPalette(kBird, 0, 1);
}

void compareMaps(std::string gainmapfile, std::string refMap, std::string gainmapMapName, std::string refMapName, std::string outFileType)
{
  using namespace o2::tpc;
  // this function compares the extracted gain map with the reference gain map
  CalPad gainMap = loadGainMap(gainmapfile, gainmapMapName); // created gainmap
  CalPad gainMapRef = loadGainMap(refMap, refMapName);       // reference gainmap

  std::array<TH1F, 3> h_Sector{};
  std::array<std::string, 3> title = {"gain from reference map", "gain from created map", "gain ref / gain new"};
  for (unsigned int it = 0; it < h_Sector.size(); it++) {
    h_Sector[it].SetBins(300, 0.1, 2);
    vhcolorTH1(h_Sector[it], title[it], "counts", 36, .9f, 1);
  }

  // get the correct binning for each roc
  std::array<std::array<std::vector<double>, 10>, 2> binPads = getBinning();          // 10 different pad sizes (width and length)
  std::array<std::array<std::array<TH2F, binPads[0].size()>, 36>, 3> h_Region_Gain{}; // 36 sectors and 3 map typs

  for (unsigned int iMap = 0; iMap < h_Region_Gain.size(); ++iMap) {
    for (unsigned int iSec = 0; iSec < h_Region_Gain[0].size(); ++iSec) {
      for (unsigned int iReg = 0; iReg < binPads[0].size(); iReg++) {
        h_Region_Gain[iMap][iSec][iReg].SetBins(static_cast<int>(binPads[1][iReg].size() - 1), binPads[1][iReg].data(), static_cast<int>(binPads[0][iReg].size() - 1), binPads[0][iReg].data());
      }
    }
  }

  //fill the histograms
  for (size_t iSec = 0; iSec < 72; iSec++) {
    ROC rocTemp(static_cast<unsigned char>(iSec)); // only this region!!! 0-72
    ROC rocIROC(0);

    const auto& mapper = Mapper::instance();
    const int nrows = mapper.getNumberOfRowsROC(rocTemp);
    const int nrowsIROC = mapper.getNumberOfRowsROC(rocIROC);
    for (int irow = 0; irow < nrows; ++irow) {
      int currRow = nrowsIROC + irow;
      if (iSec < 36) {
        currRow = irow;
      }

      const int npads = mapper.getNumberOfPadsInRowROC(rocTemp, irow);
      for (int ipad = 0; ipad < npads; ++ipad) {
        const auto valRef = gainMapRef.getValue(rocTemp, static_cast<unsigned int>(irow), static_cast<unsigned int>(ipad));
        const auto valNew = gainMap.getValue(rocTemp, static_cast<unsigned int>(irow), static_cast<unsigned int>(ipad));

        unsigned int iCurrentRegion = 0;
        int iPadRowRegion = 0;
        //get the padlength and padwidth for current region
        for (unsigned char iRegion = 0; iRegion < 10; iRegion++) {
          const PadRegionInfo& region = mapper.getPadRegionInfo(iRegion);
          iPadRowRegion += (int)region.getNumberOfPadRows();
          if ((currRow) < iPadRowRegion) {
            break;
          }
          iCurrentRegion++;
        }

        const float xPosition = mapper.getPadCentre(PadPos(static_cast<unsigned char>(currRow), static_cast<unsigned char>(ipad))).X();
        const float yPosition = mapper.getPadCentre(PadPos(static_cast<unsigned char>(currRow), static_cast<unsigned char>(ipad))).Y();

        h_Sector[0].Fill(valRef);
        h_Sector[1].Fill(valNew);
        h_Sector[2].Fill(valRef / valNew);
        if (iSec >= 36) {
          h_Region_Gain[0][iSec - 36][iCurrentRegion].Fill(xPosition, yPosition, valNew);
          h_Region_Gain[1][iSec - 36][iCurrentRegion].Fill(xPosition, yPosition, valRef);
          h_Region_Gain[2][iSec - 36][iCurrentRegion].Fill(xPosition, yPosition, valRef / valNew);
        } else {
          h_Region_Gain[0][iSec][iCurrentRegion].Fill(xPosition, yPosition, valNew);
          h_Region_Gain[1][iSec][iCurrentRegion].Fill(xPosition, yPosition, valRef);
          h_Region_Gain[2][iSec][iCurrentRegion].Fill(xPosition, yPosition, valRef / valNew);
        }
      }
    }
  }
  //===============================Drawing ================================
  TCanvas canv("canv", "canv", 900, 600);
  canv.cd();
  setgpad(0.03f, 0.2f, 0.14f, 0.14f);
  for (unsigned int iMap = 0; iMap < h_Region_Gain.size(); iMap++) {
    std::array<std::string, 3> sMap = {"Extracted", "Reference", "Divided"};
    for (unsigned int iSec = 0; iSec < h_Region_Gain[iMap].size(); ++iSec) {
      TH2F hDUMMY_Gain("hDUMMY_Gain", "hDUMMY_Gain", 153, 80, 250, 90, -45, 45);
      vhcolorTH2(hDUMMY_Gain, "#it{x} (cm)", "#it{y} (cm)", 36, .8f, .8f);
      hDUMMY_Gain.GetZaxis()->SetRangeUser(0., 1.3);
      hDUMMY_Gain.GetYaxis()->SetNdivisions(5, 5, 0);
      hDUMMY_Gain.GetXaxis()->SetNdivisions(5, 5, 0);
      hDUMMY_Gain.Draw("");
      for (unsigned int iROC = 0; iROC < 10; iROC++) {

        h_Region_Gain[iMap][iSec][iROC].SetMaximum(1.4);
        if (iROC == 0) {
          vhcolorTH2(h_Region_Gain[iMap][iSec][iROC], "", "", 36, .8f, .8f);
          const std::array<std::string, 3> zTitle = {"rel. gain extracted", "rel. gain reference", "gain reference/gain extracted"};
          h_Region_Gain[iMap][iSec][iROC].GetZaxis()->SetTitle(zTitle[iMap].data());
          h_Region_Gain[iMap][iSec][iROC].Draw("colzsame");
        } else {
          h_Region_Gain[iMap][iSec][iROC].Draw("colsame");
        }
      }
      canv.SaveAs(Form("Gainmap_Sector_%s_%d.%s", sMap[iMap].data(), iSec, outFileType.data()));
    }
  }

  setgpad(0.03f, 0.03f, 0.14f, 0.14f);
  TLegend leg(0.65, 0.7, 0.95, 0.9);
  setleg(leg);
  leg.SetFillColorAlpha(0, 0);
  TF1 fGaus("gaus", "gaus", 0.5, 1.5);
  std::array<std::string, 3> sName = {"OROC_REF", "OROC_EXT", "OROC_DIV"};
  for (unsigned int it = 0; it < h_Sector.size(); ++it) {
    h_Sector[it].Draw();
    h_Sector[it].Fit(&fGaus, "QMR0");
    fGaus.DrawCopy("SAME");
    leg.SetHeader("Fit Parameter");
    leg.AddEntry(&fGaus, Form("mean: %.3f", fGaus.GetParameter(1)), "l");
    leg.AddEntry(&fGaus, Form("sigma: %.3f", fGaus.GetParameter(2)), "l");
    leg.Draw("SAME");
    canv.SaveAs(Form("%s.%s", sName[it].data(), outFileType.data()));
    leg.Clear();
  }
}

void drawTPCPolar(TH2F& tpc_Side, o2::tpc::Side side, o2::tpc::CalPad& gainMap)
{
  using namespace o2::tpc;
  for (ROC roc; !roc.looped(); ++roc) {
    if (roc.side() != side) {
      continue;
    }
    const auto& mapper = Mapper::instance();
    const int nrows = mapper.getNumberOfRowsROC(roc);
    for (int irow = 0; irow < nrows; ++irow) {
      const int npads = mapper.getNumberOfPadsInRowROC(roc, irow);
      for (int ipad = 0; ipad < npads; ++ipad) {
        const auto val = gainMap.getValue(roc, static_cast<unsigned int>(irow), static_cast<unsigned int>(ipad));
        GlobalPosition2D pos = mapper.getPadCentre(PadROCPos(roc, irow, ipad));
        const float x = pos.X();
        const float y = pos.Y();
        const float r = std::sqrt(x * x + y * y);
        const float theta = std::atan2(y, x);
        const int bin = tpc_Side.FindBin(theta, r);
        if (!static_cast<bool>(tpc_Side.GetBinContent(bin))) {
          tpc_Side.SetBinContent(bin, val);
        }
      }
    }
  }
}

std::array<std::array<std::vector<double>, 10>, 2> getBinning()
{
  using namespace o2::tpc;
  const auto& mapper = Mapper::instance();
  const int totalRegions = 10;
  std::array<std::vector<double>, totalRegions> binningPadWidthX{};
  std::array<std::vector<double>, totalRegions> binningPadLengthX{};

  for (unsigned char iRegion = 0; iRegion < totalRegions; iRegion++) {
    const PadRegionInfo& region = mapper.getPadRegionInfo(iRegion);
    const int iPadRows = (int)region.getNumberOfPadRows();
    std::vector<double> vPadLength;
    vPadLength.reserve(static_cast<unsigned int>(iPadRows));

    const float padWidth = region.getPadWidth();
    const float padLength = region.getPadHeight();

    for (int iRow = 0; iRow < iPadRows; iRow++) {
      const int offs = mapper.getGlobalRowOffsetRegion(iRegion);
      const float xPosition = mapper.getPadCentre(PadPos(static_cast<unsigned char>(iRow + offs), 0)).X();
      vPadLength.emplace_back(xPosition - 0.5f * padLength);
    }

    const int offs = mapper.getGlobalRowOffsetRegion(iRegion);
    const int nPads = mapper.getNumberOfPadsInRowSector(offs + iPadRows - 1);
    std::vector<double> vPadwidth;
    vPadLength.reserve(static_cast<unsigned int>(nPads));
    for (int iPad = nPads - 1; iPad >= 0; --iPad) {
      const float yPosition = mapper.getPadCentre(PadPos(static_cast<unsigned char>(offs + iPadRows - 1), static_cast<unsigned char>(iPad))).Y();
      vPadwidth.emplace_back(yPosition - 0.5f * padWidth);
    }

    vPadwidth.push_back(vPadwidth.back() + padWidth);
    vPadLength.push_back(vPadLength.back() + padLength);
    binningPadLengthX[iRegion] = std::move(vPadLength);
    binningPadWidthX[iRegion] = std::move(vPadwidth);
  }

  std::array<std::array<std::vector<double>, totalRegions>, 2> binningPads{binningPadWidthX, binningPadLengthX};
  return binningPads;
}

o2::tpc::CalPad loadGainMap(std::string gainmapfile, std::string gainMapName)
{
  TFile f(gainmapfile.data(), "READ");
  o2::tpc::CalPad* gainMap = nullptr;
  f.GetObject(gainMapName.data(), gainMap);

  if (!gainMap) {
    std::cout << gainmapfile.data() << " NOT FOUND! RETURNING! " << std::endl;
  }

  f.Close();
  return *gainMap;
}

void drawTicks()
{
  TLine line;
  line.SetLineWidth(1);
  int x_Pos = -280;
  for (int i = 0; i < 30; ++i) {
    int y_min = -300;
    int y_max = -280.;
    if (x_Pos % 100)
      y_max = -292.;
    line.DrawLine(x_Pos, y_min, x_Pos, y_max); //draw ticks on x axis
    line.DrawLine(y_min, x_Pos, y_max, x_Pos); //draw ticks on x axis
    x_Pos += 20;
  }
}

void setLogColorPalette()
{
  // use the colors from the standard kBird color palette and set the stop points logarithmic
  const float alpha = 1.;
  double stops[9] = {0.0000, 0.1250, 0.2500, 0.3750, 0.5000, 0.6250, 0.7500, 0.8750, 1.0000};
  double red[9] = {0.2082, 0.0592, 0.0780, 0.0232, 0.1802, 0.5301, 0.8186, 0.9956, 0.9764};
  double green[9] = {0.1664, 0.3599, 0.5041, 0.6419, 0.7178, 0.7492, 0.7328, 0.7862, 0.9832};
  double blue[9] = {0.5293, 0.8684, 0.8385, 0.7914, 0.6425, 0.4662, 0.3499, 0.1968, 0.0539};

  for (int i = 0; i < 9; i++) {
    stops[i] = std::expm1(stops[i] * 1) / std::expm1(stops[8] * 1);
  }

  double stopsNew[9] = {0};
  int invCount = 8;
  for (int i = 0; i < 9; i++) {
    stopsNew[i] = 1 - stops[invCount];
    --invCount;
  }
  TColor::CreateGradientColorTable(9, stopsNew, red, green, blue, 255, alpha);
}

void setgpad(float tM, float rM, float bM, float lM)
{
  gPad->SetTopMargin(tM);
  gPad->SetRightMargin(rM);
  gPad->SetLeftMargin(lM);
  gPad->SetBottomMargin(bM);
}

void setleg(TLegend& leg, int isize)
{
  leg.SetFillColor(10);
  leg.SetBorderSize(0);
  leg.SetTextFont(63);
  leg.SetTextSize(isize);
}

void vhcolorTH1(TH1& hist, TString xtitle, TString ytitle, int isize, float xOff, float yOff)
{
  hist.SetLineWidth(2);
  hist.SetTitle("");
  hist.SetStats(0);
  hist.GetXaxis()->SetTitle(xtitle.Data());
  hist.GetYaxis()->SetTitle(ytitle.Data());
  hist.GetXaxis()->SetTitleOffset(xOff);
  hist.GetYaxis()->SetTitleOffset(yOff);

  const int iStdFont = 63;
  hist.GetXaxis()->SetTitleFont(iStdFont);
  hist.GetXaxis()->SetLabelFont(iStdFont);
  hist.GetYaxis()->SetTitleFont(iStdFont);
  hist.GetYaxis()->SetLabelFont(iStdFont);
  hist.SetTitleFont(iStdFont, "t");

  hist.GetXaxis()->SetTitleSize(isize + 4);
  hist.GetXaxis()->SetLabelSize(isize);
  hist.GetYaxis()->SetTitleSize(isize + 4);
  hist.GetYaxis()->SetLabelSize(isize);
  hist.SetTitleSize(isize, "t");
}

void vhcolorTH2(TH2& hist, TString xtitle, TString ytitle, int isize, float xOff, float yOff)
{
  hist.SetTitle("");
  hist.SetStats(0);
  hist.GetXaxis()->SetTitle(xtitle.Data());
  hist.GetYaxis()->SetTitle(ytitle.Data());
  hist.GetXaxis()->SetTitleOffset(xOff);
  hist.GetYaxis()->SetTitleOffset(yOff);

  const int iStdFont = 63;
  hist.GetXaxis()->SetTitleFont(iStdFont);
  hist.GetXaxis()->SetLabelFont(iStdFont);
  hist.GetYaxis()->SetTitleFont(iStdFont);
  hist.GetYaxis()->SetLabelFont(iStdFont);
  hist.GetZaxis()->SetTitleFont(iStdFont);
  hist.GetZaxis()->SetLabelFont(iStdFont);
  hist.SetTitleFont(iStdFont, "t");

  hist.GetXaxis()->SetTitleSize(isize + 4);
  hist.GetXaxis()->SetLabelSize(isize);
  hist.GetYaxis()->SetTitleSize(isize + 4);
  hist.GetYaxis()->SetLabelSize(isize);
  hist.GetZaxis()->SetTitleSize(isize + 4);
  hist.GetZaxis()->SetLabelSize(isize);
  hist.SetTitleSize(isize, "t");
}
