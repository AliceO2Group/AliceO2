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

#include "TPCCalibration/IDCCCDBHelper.h"
#include "TPCCalibration/IDCDrawHelper.h"
#include "TPCCalibration/IDCGroupHelperSector.h"
#include "TPCCalibration/IDCContainer.h"
#include "TPCBase/CalDet.h"
#include "TPCBase/Mapper.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "TPCBase/Painter.h"
#include "TCanvas.h"
#include "TH2Poly.h"
#include "TPCCalibration/IDCFactorization.h"
#include <map>

#include "TProfile.h"

template <typename DataT>
unsigned int o2::tpc::IDCCCDBHelper<DataT>::getNIntegrationIntervalsIDCDelta(const o2::tpc::Side side) const
{
  return (mIDCDelta[side] && mHelperSector[side]) ? mIDCDelta[side]->getNIDCs() / (mHelperSector[side]->getNIDCsPerSector() * SECTORSPERSIDE) : 0;
}

template <typename DataT>
unsigned int o2::tpc::IDCCCDBHelper<DataT>::getNIntegrationIntervalsIDCOne(const o2::tpc::Side side) const
{
  return mIDCOne[side] ? mIDCOne[side]->getNIDCs() : 0;
}

template <typename DataT>
float o2::tpc::IDCCCDBHelper<DataT>::getIDCZeroVal(const unsigned int sector, const unsigned int region, unsigned int urow, unsigned int upad) const
{
  /// if the number of pads of the IDC0 corresponds to the number of pads of one TPC side, then no grouping was applied
  const auto side = Sector(sector).side();
  return !mIDCZero[side] ? -1 : (mIDCZero[side]->getNIDC0() == Mapper::getNumberOfPadsPerSide()) ? mIDCZero[side]->getValueIDCZero(getUngroupedIndexGlobal(sector, region, urow, upad, 0))
                                                                                                 : mIDCZero[side]->getValueIDCZero(mHelperSector[side]->getIndexUngrouped(sector, region, urow, upad, 0));
}

template <typename DataT>
float o2::tpc::IDCCCDBHelper<DataT>::getIDCDeltaVal(const unsigned int sector, const unsigned int region, unsigned int urow, unsigned int upad, unsigned int integrationInterval) const
{
  const auto side = Sector(sector).side();
  return (!mIDCDelta[side] || !mHelperSector[side]) ? -1 : mIDCDelta[side]->getValue(mHelperSector[side]->getIndexUngrouped(sector, region, urow, upad, integrationInterval));
}

template <typename DataT>
float o2::tpc::IDCCCDBHelper<DataT>::getIDCOneVal(const o2::tpc::Side side, const unsigned int integrationInterval) const
{
  return !mIDCOne[side] ? -1 : mIDCOne[side]->getValueIDCOne(integrationInterval);
}

template <typename DataT>
float o2::tpc::IDCCCDBHelper<DataT>::getIDCVal(const unsigned int sector, const unsigned int region, unsigned int urow, unsigned int upad, unsigned int integrationInterval) const
{
  return (getIDCDeltaVal(sector, region, urow, upad, integrationInterval) + 1.f) * getIDCZeroVal(sector, region, urow, upad) * getIDCOneVal(Sector(sector).side(), integrationInterval);
}

template <typename DataT>
void o2::tpc::IDCCCDBHelper<DataT>::drawIDCZeroHelper(const bool type, const o2::tpc::Sector sector, const std::string filename, const float minZ, const float maxZ) const
{
  std::function<float(const unsigned int, const unsigned int, const unsigned int, const unsigned int)> idcFunc = [this](const unsigned int sector, const unsigned int region, const unsigned int irow, const unsigned int pad) {
    return this->getIDCZeroVal(sector, region, irow, pad);
  };

  IDCDrawHelper::IDCDraw drawFun;
  drawFun.mIDCFunc = idcFunc;
  const std::string zAxisTitle = IDCDrawHelper::getZAxisTitle(IDCType::IDCZero);
  type ? IDCDrawHelper::drawSide(drawFun, sector.side(), zAxisTitle, filename, minZ, maxZ) : IDCDrawHelper::drawSector(drawFun, 0, Mapper::NREGIONS, sector, zAxisTitle, filename, minZ, maxZ);
}

template <typename DataT>
void o2::tpc::IDCCCDBHelper<DataT>::drawIDCDeltaHelper(const bool type, const Sector sector, const unsigned int integrationInterval, const std::string filename, const float minZ, const float maxZ) const
{
  std::function<float(const unsigned int, const unsigned int, const unsigned int, const unsigned int)> idcFunc = [this, integrationInterval](const unsigned int sector, const unsigned int region, const unsigned int irow, const unsigned int pad) {
    return this->getIDCDeltaVal(sector, region, irow, pad, integrationInterval);
  };

  IDCDrawHelper::IDCDraw drawFun;
  drawFun.mIDCFunc = idcFunc;
  const std::string zAxisTitle = IDCDrawHelper::getZAxisTitle(IDCType::IDCDelta);
  type ? IDCDrawHelper::drawSide(drawFun, sector.side(), zAxisTitle, filename, minZ, maxZ) : IDCDrawHelper::drawSector(drawFun, 0, Mapper::NREGIONS, sector, zAxisTitle, filename, minZ, maxZ);
}

template <typename DataT>
void o2::tpc::IDCCCDBHelper<DataT>::drawIDCHelper(const bool type, const Sector sector, const unsigned int integrationInterval, const std::string filename, const float minZ, const float maxZ) const
{
  std::function<float(const unsigned int, const unsigned int, const unsigned int, const unsigned int)> idcFunc = [this, integrationInterval](const unsigned int sector, const unsigned int region, const unsigned int irow, const unsigned int pad) {
    return this->getIDCVal(sector, region, irow, pad, integrationInterval);
  };

  IDCDrawHelper::IDCDraw drawFun;
  drawFun.mIDCFunc = idcFunc;
  const std::string zAxisTitle = IDCDrawHelper::getZAxisTitle(IDCType::IDC);
  type ? IDCDrawHelper::drawSide(drawFun, sector.side(), zAxisTitle, filename, minZ, maxZ) : IDCDrawHelper::drawSector(drawFun, 0, Mapper::NREGIONS, sector, zAxisTitle, filename, minZ, maxZ);
}

template <typename DataT>
void o2::tpc::IDCCCDBHelper<DataT>::drawPadFlagMap(const bool type, const Sector sector, const std::string filename, const PadFlags flag) const
{
  if (!mPadFlagsMap) {
    LOGP(info, "Status map not set returning");
    return;
  }

  std::function<float(const unsigned int, const unsigned int, const unsigned int, const unsigned int)> idcFunc = [this, flag](const unsigned int sector, const unsigned int region, const unsigned int row, const unsigned int pad) {
    const unsigned int padInRegion = Mapper::OFFSETCRULOCAL[region][row] + pad;
    const auto flagDraw = mPadFlagsMap->getCalArray(region + sector * Mapper::NREGIONS).getValue(padInRegion);
    if ((flagDraw & flag) == flag) {
      return 1;
    } else {
      return 0;
    }
  };

  IDCDrawHelper::IDCDraw drawFun;
  drawFun.mIDCFunc = idcFunc;
  const std::string zAxisTitle = "status flag";
  type ? IDCDrawHelper::drawSide(drawFun, sector.side(), zAxisTitle, filename) : IDCDrawHelper::drawSector(drawFun, 0, Mapper::NREGIONS, sector, zAxisTitle, filename);
}

template <typename DataT>
unsigned int o2::tpc::IDCCCDBHelper<DataT>::getUngroupedIndexGlobal(const unsigned int sector, const unsigned int region, unsigned int urow, unsigned int upad, unsigned int integrationInterval) const
{
  return IDCGroupHelperSector::getUngroupedIndexGlobal(sector, region, urow, upad, integrationInterval);
}

template <typename DataT>
TCanvas* o2::tpc::IDCCCDBHelper<DataT>::drawIDCZeroCanvas(TCanvas* outputCanvas, std::string_view type, int nbins1D, float xMin1D, float xMax1D, int integrationInterval) const
{
  TCanvas* canv = nullptr;

  std::function<float(const unsigned int, const unsigned int, const unsigned int, const unsigned int)> idcFunc;

  if (type == "IDC0") {
    idcFunc = [this](const unsigned int sector, const unsigned int region, const unsigned int irow, const unsigned int pad) {
      return this->getIDCZeroVal(sector, region, irow, pad);
    };
  } else if (type == "IDCDelta") {
    idcFunc = [this, integrationInterval](const unsigned int sector, const unsigned int region, const unsigned int irow, const unsigned int pad) {
      return this->getIDCDeltaVal(sector, region, irow, pad, integrationInterval);
    };
    if (integrationInterval <= 0) {
      integrationInterval = std::min(getNIntegrationIntervalsIDCDelta(Side::A), getNIntegrationIntervalsIDCDelta(Side::C));
    }
  } else if (type == "IDC") {
    idcFunc = [this, integrationInterval](const unsigned int sector, const unsigned int region, const unsigned int irow, const unsigned int pad) {
      return this->getIDCVal(sector, region, irow, pad, integrationInterval);
    };
    if (integrationInterval <= 0) {
      integrationInterval = std::min(getNIntegrationIntervalsIDCDelta(Side::A), getNIntegrationIntervalsIDCDelta(Side::C));
    }
  } else {
    LOGP(error, "Please provide a valid IDC data type. 'IDC0', 'IDCDelta', or 'IDC'.");
    return canv;
  }

  if (outputCanvas) {
    canv = outputCanvas;
  } else {
    canv = new TCanvas(fmt::format("c_sides_{}", type.data()).data(), fmt::format("{}", type.data()).data(), 1000, 1000);
  }

  IDCDrawHelper::IDCDraw drawFun;
  drawFun.mIDCFunc = idcFunc;
  const std::string zAxisTitle = IDCDrawHelper::getZAxisTitle(IDCType::IDCZero);

  auto hAside2D = IDCDrawHelper::drawSide(drawFun, Side::A, zAxisTitle);
  auto hCside2D = IDCDrawHelper::drawSide(drawFun, Side::C, zAxisTitle);
  hAside2D->SetTitle(fmt::format("{} (A-Side)", type.data()).data());
  hCside2D->SetTitle(fmt::format("{} (C-Side)", type.data()).data());
  hAside2D->SetMinimum(xMin1D);
  hAside2D->SetMaximum(xMax1D);
  hCside2D->SetMinimum(xMin1D);
  hCside2D->SetMaximum(xMax1D);

  auto hAside1D = IDCDrawHelper::drawSide(drawFun, fmt::format("{}", type.data()).data(), Side::A, nbins1D, xMin1D, xMax1D);
  auto hCside1D = IDCDrawHelper::drawSide(drawFun, fmt::format("{}", type.data()).data(), Side::C, nbins1D, xMin1D, xMax1D);

  canv->Divide(2, 2);
  canv->cd(1);
  hAside2D->Draw("colz");
  hAside2D->SetStats(0);
  hAside2D->SetTitleOffset(1.05, "XY");
  hAside2D->SetTitleSize(0.05, "XY");
  o2::tpc::painter::drawSectorsXY(Side::A);

  canv->cd(2);
  hCside2D->Draw("colz");
  hCside2D->SetStats(0);
  hCside2D->SetTitleOffset(1.05, "XY");
  hCside2D->SetTitleSize(0.05, "XY");
  o2::tpc::painter::drawSectorsXY(Side::C);

  canv->cd(3);
  hAside1D->Draw();

  canv->cd(4);
  hCside1D->Draw();

  hAside2D->SetBit(TObject::kCanDelete);
  hCside2D->SetBit(TObject::kCanDelete);
  hAside1D->SetBit(TObject::kCanDelete);
  hCside1D->SetBit(TObject::kCanDelete);

  return canv;
}

template <typename DataT>
TCanvas* o2::tpc::IDCCCDBHelper<DataT>::drawIDCZeroRadialProfile(TCanvas* outputCanvas, int nbinsY, float yMin, float yMax) const
{
  std::function<float(const unsigned int, const unsigned int, const unsigned int, const unsigned int)> idcFunc = [this](const unsigned int sector, const unsigned int region, const unsigned int irow, const unsigned int pad) {
    return this->getIDCZeroVal(sector, region, irow, pad);
  };

  TCanvas* canv = nullptr;

  if (outputCanvas) {
    canv = outputCanvas;
  } else {
    canv = new TCanvas("c_sides_IDC0_radialProfile", "IDC0", 1000, 1000);
  }

  IDCDrawHelper::IDCDraw drawFun;
  drawFun.mIDCFunc = idcFunc;

  const auto radialBinning = o2::tpc::painter::getRowBinningCM();

  auto hAside2D = new TH2F("h_IDC0_radialProfile_Aside", "IDC0: Radial profile (A-Side)", radialBinning.size() - 1, radialBinning.data(), nbinsY, yMin, yMax);
  hAside2D->GetXaxis()->SetTitle("x (cm)");
  hAside2D->GetYaxis()->SetTitle("IDC0 (ADC)");
  hAside2D->SetTitleOffset(1.05, "XY");
  hAside2D->SetTitleSize(0.05, "XY");
  hAside2D->SetStats(0);

  auto hCside2D = new TH2F("h_IDC0_radialProfile_Cside", "IDC0: Radial profile (C-Side)", radialBinning.size() - 1, radialBinning.data(), nbinsY, yMin, yMax);
  hCside2D->GetXaxis()->SetTitle("x (cm)");
  hCside2D->GetYaxis()->SetTitle("IDC0 (ADC)");
  hCside2D->SetTitleOffset(1.05, "XY");
  hCside2D->SetTitleSize(0.05, "XY");
  hCside2D->SetStats(0);

  IDCDrawHelper::drawRadialProfile(drawFun, *hAside2D, Side::A);
  IDCDrawHelper::drawRadialProfile(drawFun, *hCside2D, Side::C);

  // auto profA = hAside2D->ProfileX();
  // auto profC = hCside2D->ProfileX();

  canv->Divide(1, 2);
  canv->cd(1);
  hAside2D->Draw("colz");
  hAside2D->ProfileX("profile_ASide", 1, -1, "d,same");
  // profA->Draw("d,same");
  hAside2D->SetStats(0);

  canv->cd(2);
  hCside2D->Draw("colz");
  hCside2D->ProfileX("profile_CSide", 1, -1, "d,same");
  // profC->Draw("d,same");
  hAside2D->SetStats(0);

  hAside2D->SetBit(TObject::kCanDelete);
  hCside2D->SetBit(TObject::kCanDelete);

  return canv;
}

template <typename DataT>
TCanvas* o2::tpc::IDCCCDBHelper<DataT>::drawIDCZeroStackCanvas(TCanvas* outputCanvas, Side side, std::string_view type, int nbins1D, float xMin1D, float xMax1D, int integrationInterval) const
{
  TCanvas* canv = nullptr;

  std::function<float(const unsigned int, const unsigned int, const unsigned int, const unsigned int)> idcFunc;

  if (type == "IDC0") {
    idcFunc = [this](const unsigned int sector, const unsigned int region, const unsigned int irow, const unsigned int pad) {
      return this->getIDCZeroVal(sector, region, irow, pad);
    };
  } else if (type == "IDCDelta") {
    idcFunc = [this, integrationInterval](const unsigned int sector, const unsigned int region, const unsigned int irow, const unsigned int pad) {
      return this->getIDCDeltaVal(sector, region, irow, pad, integrationInterval);
    };
    if (integrationInterval <= 0) {
      integrationInterval = std::min(getNIntegrationIntervalsIDCDelta(Side::A), getNIntegrationIntervalsIDCDelta(Side::C));
    }
  } else {
    LOGP(error, "Please provide a valid IDC data type. 'IDC0' or 'IDCDelta'.");
    return canv;
  }

  if (outputCanvas) {
    canv = outputCanvas;
  } else {
    canv = new TCanvas(fmt::format("c_GEMStacks_{}_1D_{}Side", type.data(), (side == Side::A) ? "A" : "C").data(), fmt::format("{} value 1D distribution for each GEM stack {}-Side", type.data(), (side == Side::A) ? "A" : "C").data(), 1000, 1000);
  }

  IDCDrawHelper::IDCDraw drawFun;
  drawFun.mIDCFunc = idcFunc;

  IDCDrawHelper::drawIDCZeroStackCanvas(drawFun, side, type, nbins1D, xMin1D, xMax1D, *canv, integrationInterval);

  return canv;
}

template <typename DataT>
TCanvas* o2::tpc::IDCCCDBHelper<DataT>::drawIDCOneCanvas(TCanvas* outputCanvas, int nbins1D, float xMin1D, float xMax1D, int integrationIntervals) const
{
  TCanvas* canv = nullptr;

  if (outputCanvas) {
    canv = outputCanvas;
  } else {
    canv = new TCanvas("c_sides_IDC1_1D", "IDC1 1D distribution for each side", 1000, 1000);
  }

  auto hAside1D = new TH1F("h_IDC1_1D_ASide", "IDC1 distribution over integration intervals A-Side", nbins1D, xMin1D, xMax1D);
  auto hCside1D = new TH1F("h_IDC1_1D_CSide", "IDC1 distribution over integration intervals C-Side", nbins1D, xMin1D, xMax1D);

  hAside1D->GetXaxis()->SetTitle("IDC1");
  hAside1D->SetTitleOffset(1.05, "XY");
  hAside1D->SetTitleSize(0.05, "XY");
  hCside1D->GetXaxis()->SetTitle("IDC1");
  hCside1D->SetTitleOffset(1.05, "XY");
  hCside1D->SetTitleSize(0.05, "XY");

  if (integrationIntervals <= 0) {
    integrationIntervals = std::min(mIDCOne[Side::A]->getNIDCs(), mIDCOne[Side::C]->getNIDCs());
  }

  for (unsigned int integrationInterval = 0; integrationInterval < integrationIntervals; ++integrationInterval) {
    hAside1D->Fill(getIDCOneVal(Side::A, integrationInterval));
    hCside1D->Fill(getIDCOneVal(Side::C, integrationInterval));
  }

  canv->Divide(1, 2);
  canv->cd(1);
  hAside1D->Draw();
  canv->cd(2);
  hCside1D->Draw();

  hAside1D->SetBit(TObject::kCanDelete);
  hCside1D->SetBit(TObject::kCanDelete);

  return canv;
}

template <typename DataT>
TCanvas* o2::tpc::IDCCCDBHelper<DataT>::drawFourierCoeff(TCanvas* outputCanvas, Side side, int nbins1D, float xMin1D, float xMax1D) const
{
  TCanvas* canv = nullptr;

  if (outputCanvas) {
    canv = outputCanvas;
  } else {
    canv = new TCanvas(fmt::format("c_FourierCoefficients_1D_{}Side", (side == Side::A) ? "A" : "C").data(), fmt::format("1D distributions of Fourier Coefficients ({}-Side)", (side == Side::A) ? "A" : "C").data(), 1000, 1000);
  }

  std::vector<TH1F*> histos;

  for (int i = 0; i < mFourierCoeff[side]->getNCoefficientsPerTF(); i++) {
    histos.emplace_back(new TH1F(fmt::format("h_FourierCoeff{}_{}Side", i, (side == Side::A) ? "A" : "C").data(), fmt::format("1D distribution of Fourier Coefficient {} ({}-Side)", i, (side == Side::A) ? "A" : "C").data(), nbins1D, xMin1D, xMax1D));
    histos.back()->GetXaxis()->SetTitle(fmt::format("Fourier Coefficient {}", i).data());
    histos.back()->SetBit(TObject::kCanDelete);
  }

  const auto& coeffs = mFourierCoeff[side]->getFourierCoefficients();
  const auto nCoeffPerTF = mFourierCoeff[side]->getNCoefficientsPerTF();

  for (int i = 0; i < mFourierCoeff[side]->getNCoefficients(); i++) {
    histos.at(i % nCoeffPerTF)->Fill(coeffs.at(i));
  }

  canv->DivideSquare(mFourierCoeff[side]->getNCoefficientsPerTF());

  size_t pad = 1;

  for (const auto& hist : histos) {
    canv->cd(pad);
    hist->SetTitleOffset(1.05, "XY");
    hist->SetTitleSize(0.05, "XY");
    hist->Draw();
    pad++;
  }

  return canv;
}

template <typename DataT>
void o2::tpc::IDCCCDBHelper<DataT>::dumpToTree(const char* outFileName) const
{
  const Mapper& mapper = Mapper::instance();
  o2::utils::TreeStreamRedirector pcstream(outFileName, "RECREATE");
  pcstream.GetFile()->cd();

  const int integrationInterval = 0; // std::min(mIDCOne->getNIDCs(Side::A), mIDCOne->getNIDCs(Side::C));

  const unsigned int nIDCsSector = Mapper::getPadsInSector() * Mapper::NSECTORS;
  std::vector<int> vRow(nIDCsSector);
  std::vector<int> vPad(nIDCsSector);
  std::vector<float> vXPos(nIDCsSector);
  std::vector<float> vYPos(nIDCsSector);
  std::vector<float> vGlobalXPos(nIDCsSector);
  std::vector<float> vGlobalYPos(nIDCsSector);
  std::vector<float> idcs(nIDCsSector);
  std::vector<float> idcsZero(nIDCsSector);
  std::vector<float> idcsDelta(nIDCsSector);
  std::vector<unsigned int> sectorv(nIDCsSector);

  unsigned int index = 0;
  for (unsigned int sector = 0; sector < Mapper::NSECTORS; ++sector) {
    for (unsigned int region = 0; region < Mapper::NREGIONS; ++region) {
      for (unsigned int irow = 0; irow < Mapper::ROWSPERREGION[region]; ++irow) {
        for (unsigned int ipad = 0; ipad < Mapper::PADSPERROW[region][irow]; ++ipad) {
          const auto padNum = Mapper::getGlobalPadNumber(irow, ipad, region);
          const auto padTmp = (sector < SECTORSPERSIDE) ? ipad : (Mapper::PADSPERROW[region][irow] - ipad - 1); // C-Side is mirrored
          const auto& padPosLocal = mapper.padPos(padNum);
          vRow[index] = padPosLocal.getRow();
          vPad[index] = padPosLocal.getPad();
          vXPos[index] = mapper.getPadCentre(padPosLocal).X();
          vYPos[index] = mapper.getPadCentre(padPosLocal).Y();
          const GlobalPosition2D globalPos = mapper.LocalToGlobal(LocalPosition2D(vXPos[index], vYPos[index]), Sector(sector));
          vGlobalXPos[index] = globalPos.X();
          vGlobalYPos[index] = globalPos.Y();
          idcs[index] = getIDCVal(sector, region, irow, padTmp, integrationInterval);
          idcsZero[index] = getIDCZeroVal(sector, region, irow, padTmp);
          idcsDelta[index] = getIDCDeltaVal(sector, region, irow, padTmp, integrationInterval);
          sectorv[index++] = sector;
        }
      }
    }
  }
  std::vector<float> idcOneA = !mIDCOne[Side::A] ? std::vector<float>() : mIDCOne[Side::A]->mIDCOne;
  std::vector<float> idcOneC = !mIDCOne[Side::C] ? std::vector<float>() : mIDCOne[Side::C]->mIDCOne;

  pcstream << "tree"
           << "IDC.=" << idcs
           << "IDC0.=" << idcsZero
           << "IDC1A=" << idcOneA
           << "IDC1C=" << idcOneC
           << "IDCDelta.=" << idcsDelta
           << "pad.=" << vPad
           << "row.=" << vRow
           << "lx.=" << vXPos
           << "ly.=" << vYPos
           << "gx.=" << vGlobalXPos
           << "gy.=" << vGlobalYPos
           << "sector.=" << sectorv
           << "\n";
  pcstream.Close();
}

template <typename DataT>
void o2::tpc::IDCCCDBHelper<DataT>::dumpToTreeIDCDelta(const char* outFileName) const
{
  const Mapper& mapper = Mapper::instance();
  o2::utils::TreeStreamRedirector pcstream(outFileName, "RECREATE");
  pcstream.GetFile()->cd();
  const unsigned int nIDCsSector = Mapper::getPadsInSector() * Mapper::NSECTORS;
  std::vector<float> idcsDelta(nIDCsSector);
  for (int integrationInterval = 0; integrationInterval < std::min(getNIntegrationIntervalsIDCDelta(Side::A), getNIntegrationIntervalsIDCDelta(Side::C)); ++integrationInterval) {
    unsigned int index = 0;
    for (unsigned int sector = 0; sector < Mapper::NSECTORS; ++sector) {
      for (unsigned int region = 0; region < Mapper::NREGIONS; ++region) {
        for (unsigned int irow = 0; irow < Mapper::ROWSPERREGION[region]; ++irow) {
          for (unsigned int ipad = 0; ipad < Mapper::PADSPERROW[region][irow]; ++ipad) {
            const auto padNum = Mapper::getGlobalPadNumber(irow, ipad, region);
            const auto padTmp = (sector < SECTORSPERSIDE) ? ipad : (Mapper::PADSPERROW[region][irow] - ipad - 1); // C-Side is mirrored
            const auto& padPosLocal = mapper.padPos(padNum);
            idcsDelta[index++] = getIDCDeltaVal(sector, region, irow, padTmp, integrationInterval);
          }
        }
      }
    }
    float idcOneA = getIDCOneVal(o2::tpc::Side::A, integrationInterval);
    float idcOneC = getIDCOneVal(o2::tpc::Side::C, integrationInterval);

    pcstream << "tree"
             << "IDC1A=" << idcOneA
             << "IDC1C=" << idcOneC
             << "IDCDelta.=" << idcsDelta
             << "\n";
  }
  pcstream.Close();
}

template <typename DataT>
void o2::tpc::IDCCCDBHelper<DataT>::dumpToFourierCoeffToTree(const char* outFileName) const
{
  o2::utils::TreeStreamRedirector pcstream(outFileName, "RECREATE");
  pcstream.GetFile()->cd();

  for (int iside = 0; iside < SIDES; ++iside) {
    const Side side = (iside == 0) ? Side::A : Side::C;

    if (!mFourierCoeff[side]) {
      continue;
    }

    const int nTFs = mFourierCoeff[side]->getNCoefficients() / mFourierCoeff[side]->getNCoefficientsPerTF();
    for (int iTF = 0; iTF < nTFs; ++iTF) {
      std::vector<float> coeff;
      std::vector<int> ind;
      int coeffPerTF = mFourierCoeff[side]->getNCoefficientsPerTF();
      for (int i = 0; i < coeffPerTF; ++i) {
        const int index = mFourierCoeff[side]->getIndex(iTF, i);
        coeff.emplace_back((*mFourierCoeff[side])(index));
        ind.emplace_back(i);
      }

      pcstream << "tree"
               << "iTF=" << iTF
               << "index=" << ind
               << "coeffPerTF=" << coeffPerTF
               << "coeff.=" << coeff
               << "side=" << iside
               << "\n";
    }
  }
  pcstream.Close();
}

template <typename DataT>
o2::tpc::CalDet<float> o2::tpc::IDCCCDBHelper<DataT>::getIDCZeroCalDet() const
{
  CalDet<float> calIDC0("IDC0");
  for (unsigned int cru = 0; cru < CRU::MaxCRU; ++cru) {
    const o2::tpc::CRU cruTmp(cru);
    const Side side = cruTmp.side();
    const int region = cruTmp.region();
    const int sector = cruTmp.sector();
    for (unsigned int lrow = 0; lrow < Mapper::ROWSPERREGION[region]; ++lrow) {
      const unsigned int integrationInterval = 0;
      for (unsigned int pad = 0; pad < Mapper::PADSPERROW[region][lrow]; ++pad) {
        const auto idcZero = getIDCZeroVal(sector, region, lrow, pad);
        calIDC0.setValue(sector, Mapper::ROWOFFSET[region] + lrow, pad, idcZero);
      }
    }
  }
  return calIDC0;
}

template <typename DataT>
std::vector<o2::tpc::CalDet<float>> o2::tpc::IDCCCDBHelper<DataT>::getIDCDeltaCalDet() const
{
  const unsigned int nIntervalsA = getNIntegrationIntervalsIDCDelta(Side::A);
  const unsigned int nIntervalsC = getNIntegrationIntervalsIDCDelta(Side::C);
  if (nIntervalsA != nIntervalsC) {
    LOGP(info, "Number of integration interval for A Side {} unequal for C side {}", nIntervalsA, nIntervalsC);
  }

  std::vector<CalPad> calIDCDelta(std::max(nIntervalsA, nIntervalsC));
  for (auto& calpadIDC : calIDCDelta) {
    calpadIDC = CalPad("IDCDelta", PadSubset::ROC);
  }

  for (unsigned int cru = 0; cru < CRU::MaxCRU; ++cru) {
    const o2::tpc::CRU cruTmp(cru);
    const Side side = cruTmp.side();
    const int region = cruTmp.region();
    const int sector = cruTmp.sector();

    const unsigned int nIntervals = getNIntegrationIntervalsIDCDelta(side);
    for (unsigned int integrationInterval = 0; integrationInterval < nIntervals; ++integrationInterval) {
      for (unsigned int lrow = 0; lrow < Mapper::ROWSPERREGION[region]; ++lrow) {
        for (unsigned int pad = 0; pad < Mapper::PADSPERROW[region][lrow]; ++pad) {
          const auto idcdelta = getIDCDeltaVal(sector, region, lrow, pad, integrationInterval);
          calIDCDelta[integrationInterval].setValue(sector, Mapper::ROWOFFSET[region] + lrow, pad, idcdelta);
        }
      }
    }
  }
  return calIDCDelta;
}

template <typename DataT>
void o2::tpc::IDCCCDBHelper<DataT>::createOutlierMap()
{
  if (!mIDCZero[Side::A] && !mIDCZero[Side::C]) {
    LOGP(info, "IDC0 not set. returning");
  }

  int nCRUS = 0;
  if (mIDCZero[Side::A]) {
    nCRUS += CRU::MaxCRU / 2;
  }

  if (mIDCZero[Side::C]) {
    nCRUS += CRU::MaxCRU / 2;
  }

  std::vector<uint32_t> crus(nCRUS);
  std::iota(crus.begin(), crus.end(), mIDCZero[Side::A] ? 0 : (CRU::MaxCRU / 2));

  IDCFactorization idc(1, 1, crus);
  if (mIDCZero[Side::A]) {
    idc.setIDCZero(Side::A, *mIDCZero[Side::A]);
  }
  if (mIDCZero[Side::C]) {
    idc.setIDCZero(Side::C, *mIDCZero[Side::C]);
  }
  idc.createStatusMap();
  mPadFlagsMap = idc.getPadStatusMap();
}

template <typename DataT>
float o2::tpc::IDCCCDBHelper<DataT>::scaleIDC0(const Side side, const bool rejectOutlier)
{
  if (!mIDCZero[side]) {
    LOGP(info, "IDC0 not set. returning");
    return -1;
  }

  // calculate sum of IDCs over working channels
  double idc0Sum = 0;
  unsigned int idcChannelsCount = 0;
  const unsigned int firstCRU = (side == Side::A) ? 0 : CRU::MaxCRU / 2;
  for (unsigned int cru = firstCRU; cru < firstCRU + CRU::MaxCRU / 2; ++cru) {
    const o2::tpc::CRU cruTmp(cru);
    const int region = cruTmp.region();
    const int sector = cruTmp.sector();
    for (unsigned int lrow = 0; lrow < Mapper::ROWSPERREGION[region]; ++lrow) {
      const unsigned int integrationInterval = 0;
      for (unsigned int pad = 0; pad < Mapper::PADSPERROW[region][lrow]; ++pad) {
        const unsigned int padInRegion = Mapper::OFFSETCRULOCAL[region][lrow] + pad;
        const o2::tpc::PadFlags flag = (mPadFlagsMap && rejectOutlier) ? mPadFlagsMap->getCalArray(cru).getValue(padInRegion) : o2::tpc::PadFlags::flagGoodPad;
        const auto index = getUngroupedIndexGlobal(sector, region, lrow, pad, 0);
        if ((flag & PadFlags::flagSkip) == PadFlags::flagSkip) {
          continue;
        }
        ++idcChannelsCount;
        idc0Sum += getIDCZeroVal(sector, region, lrow, pad);
      }
    }
  }

  if (idcChannelsCount == 0) {
    return idc0Sum;
  }

  idc0Sum /= idcChannelsCount;
  *mIDCZero[side] /= idc0Sum;
  return idc0Sum;
}

template class o2::tpc::IDCCCDBHelper<float>;
template class o2::tpc::IDCCCDBHelper<unsigned short>;
template class o2::tpc::IDCCCDBHelper<unsigned char>;
