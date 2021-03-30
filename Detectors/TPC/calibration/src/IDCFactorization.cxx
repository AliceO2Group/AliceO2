// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TPCCalibration/IDCFactorization.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "Framework/Logger.h"
#include "TPCBase/Painter.h"
#include "TH2Poly.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TLatex.h"
#include <functional>

o2::tpc::IDCFactorization::IDCFactorization(const std::array<unsigned char, Mapper::NREGIONS>& groupPads, const std::array<unsigned char, Mapper::NREGIONS>& groupRows, const std::array<unsigned char, Mapper::NREGIONS>& groupLastRowsThreshold, const std::array<unsigned char, Mapper::NREGIONS>& groupLastPadsThreshold, const unsigned int timeFrames, const unsigned int timeframesDeltaIDC)
  : IDCGroupHelperSector{groupPads, groupRows, groupLastRowsThreshold, groupLastPadsThreshold}, mTimeFrames{timeFrames}, mTimeFramesDeltaIDC{timeframesDeltaIDC}, mIDCDelta{timeFrames / timeframesDeltaIDC + (timeFrames % timeframesDeltaIDC != 0)}
{
  for (auto& idc : mIDCs) {
    idc.resize(mTimeFrames);
  }
}

void o2::tpc::IDCFactorization::drawSector(const IDCType type, const unsigned int sector, const unsigned int integrationInterval, const std::string filename, const IDCDeltaCompression compression) const
{
  const auto coords = o2::tpc::painter::getPadCoordinatesSector();
  TH2Poly* poly = o2::tpc::painter::makeSectorHist("hSector", "Sector;local #it{x} (cm);local #it{y} (cm); #it{IDC}");
  poly->SetContour(255);
  poly->SetTitle(nullptr);
  poly->GetYaxis()->SetTickSize(0.002f);
  poly->GetYaxis()->SetTitleOffset(0.7f);
  poly->GetZaxis()->SetTitleOffset(1.3f);
  poly->SetStats(0);
  poly->GetZaxis()->SetTitle(getZAxisTitle(type, compression).data());

  TCanvas* can = new TCanvas("can", "can", 2000, 1400);
  can->SetRightMargin(0.14f);
  can->SetLeftMargin(0.06f);
  can->SetTopMargin(0.04f);

  TLatex lat;
  lat.SetTextFont(63);
  lat.SetTextSize(2);
  poly->Draw("colz");

  unsigned int chunk = 0;
  unsigned int localintegrationInterval = 0;
  getLocalIntegrationInterval(0, integrationInterval, chunk, localintegrationInterval);
  for (unsigned int region = 0; region < Mapper::NREGIONS; ++region) {
    for (unsigned int irow = 0; irow < Mapper::ROWSPERREGION[region]; ++irow) {
      for (unsigned int ipad = 0; ipad < Mapper::PADSPERROW[region][irow]; ++ipad) {
        const auto padNum = Mapper::getGlobalPadNumber(irow, ipad, region);
        const auto coordinate = coords[padNum];
        const float yPos = -0.5f * static_cast<float>(coordinate.yVals[0] + coordinate.yVals[2]); // local coordinate system is mirrored
        const float xPos = 0.5f * static_cast<float>(coordinate.xVals[0] + coordinate.xVals[2]);

        switch (type) {
          case IDCType::IDC:
          default:
            poly->Fill(xPos, yPos, getIDCValUngrouped(sector, region, irow, ipad, integrationInterval));
            break;
          case IDCType::IDCZero:
            poly->Fill(xPos, yPos, getIDCZeroVal(sector, region, irow, ipad));
            break;
          case IDCType::IDCDelta:
            switch (compression) {
              case IDCDeltaCompression::NO:
              default: {
                poly->Fill(xPos, yPos, getIDCDeltaVal(sector, region, irow, ipad, chunk, localintegrationInterval));
                break;
              }
              case IDCDeltaCompression::MEDIUM: {
                const static auto idcDeltaMedium = getIDCDeltaMediumCompressed(chunk); // make object static to avoid multiple creations of the object in the loop
                const float val = idcDeltaMedium.getValue(Sector(sector).side(), getIndexUngrouped(sector, region, irow, ipad, localintegrationInterval));
                poly->Fill(xPos, yPos, val);
                break;
              }
              case IDCDeltaCompression::HIGH: {
                const static auto idcDeltaHigh = getIDCDeltaHighCompressed(chunk); // make object static to avoid multiple creations of the object in the loop
                const float val = idcDeltaHigh.getValue(Sector(sector).side(), getIndexUngrouped(sector, region, irow, ipad, localintegrationInterval));
                poly->Fill(xPos, yPos, val);
                break;
              }
            }
          case IDCType::IDCOne:
            break;
        }
        // draw global pad number
        lat.SetTextAlign(12);
        lat.DrawLatex(xPos, yPos, Form("%i", ipad));
      }
    }
  }
  if (!filename.empty()) {
    can->SaveAs(filename.data());
    delete poly;
    delete can;
  }
}

void o2::tpc::IDCFactorization::drawSide(const IDCType type, const o2::tpc::Side side, const unsigned int integrationInterval, const std::string filename, const IDCDeltaCompression compression) const
{
  const auto coords = o2::tpc::painter::getPadCoordinatesSector();
  TH2Poly* poly = o2::tpc::painter::makeSideHist(side);
  poly->SetContour(255);
  poly->SetTitle(nullptr);
  poly->GetXaxis()->SetTitleOffset(1.2f);
  poly->GetYaxis()->SetTitleOffset(1.3f);
  poly->GetZaxis()->SetTitleOffset(1.2f);
  poly->GetZaxis()->SetTitle(getZAxisTitle(type, compression).data());
  poly->GetZaxis()->SetMaxDigits(3); // force exponential axis
  poly->SetStats(0);

  TCanvas* can = new TCanvas("can", "can", 650, 600);
  can->SetTopMargin(0.04f);
  can->SetRightMargin(0.14f);
  can->SetLeftMargin(0.1f);
  poly->Draw("colz");

  unsigned int chunk = 0;
  unsigned int localintegrationInterval = 0;
  getLocalIntegrationInterval(0, integrationInterval, chunk, localintegrationInterval);
  unsigned int sectorStart = (side == Side::A) ? 0 : o2::tpc::SECTORSPERSIDE;
  unsigned int sectorEnd = (side == Side::A) ? o2::tpc::SECTORSPERSIDE : Mapper::NSECTORS;
  for (unsigned int sector = sectorStart; sector < sectorEnd; ++sector) {
    for (unsigned int region = 0; region < Mapper::NREGIONS; ++region) {
      for (unsigned int irow = 0; irow < Mapper::ROWSPERREGION[region]; ++irow) {
        for (unsigned int ipad = 0; ipad < Mapper::PADSPERROW[region][irow]; ++ipad) {
          const auto padNum = Mapper::getGlobalPadNumber(irow, ipad, region);
          const float angDeg = 10.f + sector * 20;
          auto coordinate = coords[padNum];
          coordinate.rotate(angDeg);
          const float yPos = 0.25f * static_cast<float>(coordinate.yVals[0] + coordinate.yVals[1] + coordinate.yVals[2] + coordinate.yVals[3]);
          const float xPos = 0.25f * static_cast<float>(coordinate.xVals[0] + coordinate.xVals[1] + coordinate.xVals[2] + coordinate.xVals[3]);
          const auto padTmp = (side == Side::A) ? ipad : (Mapper::PADSPERROW[region][irow] - ipad); // C-Side is mirrored
          switch (type) {
            case IDCType::IDC:
            default:
              poly->Fill(xPos, yPos, getIDCValUngrouped(sector, region, irow, padTmp, integrationInterval));
              break;
            case IDCType::IDCZero:
              poly->Fill(xPos, yPos, getIDCZeroVal(sector, region, irow, padTmp));
              break;
            case IDCType::IDCDelta:
              switch (compression) {
                case IDCDeltaCompression::NO:
                default: {
                  poly->Fill(xPos, yPos, getIDCDeltaVal(sector, region, irow, padTmp, chunk, localintegrationInterval));
                  break;
                }
                case IDCDeltaCompression::MEDIUM: {
                  const static auto idcDeltaMedium = getIDCDeltaMediumCompressed(chunk); // make object static to avoid multiple creations of the object in the loop
                  const float val = idcDeltaMedium.getValue(Sector(sector).side(), getIndexUngrouped(sector, region, irow, padTmp, localintegrationInterval));
                  poly->Fill(xPos, yPos, val);
                  break;
                }
                case IDCDeltaCompression::HIGH: {
                  const static auto idcDeltaHigh = getIDCDeltaHighCompressed(chunk); // make object static to avoid multiple creations of the object in the loop
                  const float val = idcDeltaHigh.getValue(Sector(sector).side(), getIndexUngrouped(sector, region, irow, padTmp, localintegrationInterval));
                  poly->Fill(xPos, yPos, val);
                  break;
                }
              }
            case IDCType::IDCOne:
              break;
          }
        }
      }
    }
  }
  if (!filename.empty()) {
    can->SaveAs(filename.data());
    delete poly;
    delete can;
  }
}

std::string o2::tpc::IDCFactorization::getZAxisTitle(const IDCType type, const IDCDeltaCompression compression) const
{
  switch (type) {
    case IDCType::IDC:
    default:
      return "#it{IDC}";
      break;
    case IDCType::IDCZero:
      return "#it{IDC_{0}}";
      break;
    case IDCType::IDCDelta:
      switch (compression) {
        case IDCDeltaCompression::NO:
        default:
          return "#Delta#it{IDC}";
          break;
        case IDCDeltaCompression::MEDIUM:
          return "#Delta#it{IDC}_{medium compressed}";
          break;
        case IDCDeltaCompression::HIGH:
          return "#Delta#it{IDC}_{high compressed}";
          break;
      }
    case IDCType::IDCOne:
      return "#Delta#it{IDC}_{1}";
      break;
  }
}

void o2::tpc::IDCFactorization::dumpToFile(const char* outFileName, const char* outName) const
{
  TFile fOut(outFileName, "RECREATE");
  fOut.WriteObject(this, outName);
  fOut.Close();
}

void o2::tpc::IDCFactorization::dumpToTree(int integrationIntervals) const
{
  const Mapper& mapper = Mapper::instance();
  o2::utils::TreeStreamRedirector pcstream("IDCTree.root", "RECREATE");
  pcstream.GetFile()->cd();

  if (integrationIntervals <= 0) {
    integrationIntervals = static_cast<int>(getNIntegrationIntervals());
  }

  std::vector<float> idcOneA = mIDCOne.mIDCOne[0];
  std::vector<float> idcOneC = mIDCOne.mIDCOne[1];
  for (unsigned int integrationInterval = 0; integrationInterval < integrationIntervals; ++integrationInterval) {
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
    std::vector<float> idcsDeltaMedium(nIDCsSector);
    std::vector<float> idcsDeltaHigh(nIDCsSector);
    std::vector<unsigned int> sectorv(nIDCsSector);

    unsigned int chunk = 0;
    unsigned int localintegrationInterval = 0;
    getLocalIntegrationInterval(0, integrationInterval, chunk, localintegrationInterval);
    const auto idcDeltaMedium = getIDCDeltaMediumCompressed(chunk);
    const auto idcDeltaHigh = getIDCDeltaHighCompressed(chunk);

    unsigned int index = 0;
    for (unsigned int sector = 0; sector < Mapper::NSECTORS; ++sector) {
      for (unsigned int region = 0; region < Mapper::NREGIONS; ++region) {
        for (unsigned int irow = 0; irow < Mapper::ROWSPERREGION[region]; ++irow) {
          for (unsigned int ipad = 0; ipad < Mapper::PADSPERROW[region][irow]; ++ipad) {
            const auto padNum = Mapper::getGlobalPadNumber(irow, ipad, region);
            const auto padTmp = (sector < SECTORSPERSIDE) ? ipad : (Mapper::PADSPERROW[region][irow] - ipad); // C-Side is mirrored
            const auto& padPosLocal = mapper.padPos(padNum);
            vRow[index] = padPosLocal.getRow();
            vPad[index] = padPosLocal.getPad();
            vXPos[index] = mapper.getPadCentre(padPosLocal).X();
            vYPos[index] = mapper.getPadCentre(padPosLocal).Y();
            const GlobalPosition2D globalPos = mapper.LocalToGlobal(LocalPosition2D(vXPos[index], vYPos[index]), Sector(sector));
            vGlobalXPos[index] = globalPos.X();
            vGlobalYPos[index] = globalPos.Y();
            idcs[index] = getIDCValUngrouped(sector, region, irow, padTmp, integrationInterval);
            idcsZero[index] = getIDCZeroVal(sector, region, irow, padTmp);
            idcsDelta[index] = getIDCDeltaVal(sector, region, irow, padTmp, chunk, localintegrationInterval);
            idcsDeltaMedium[index] = idcDeltaMedium.getValue(Sector(sector).side(), getIndexUngrouped(sector, region, irow, padTmp, localintegrationInterval));
            idcsDeltaHigh[index] = idcDeltaHigh.getValue(Sector(sector).side(), getIndexUngrouped(sector, region, irow, padTmp, localintegrationInterval));
            sectorv[index++] = sector;
          }
        }
      }
    }
    float idcOneATmp = idcOneA[integrationInterval];
    float idcOneCTmp = idcOneC[integrationInterval];
    unsigned int timeFrame = 0;
    unsigned int interval = 0;
    getTF(0, integrationInterval, timeFrame, interval);

    pcstream << "tree"
             << "integrationInterval=" << integrationInterval
             << "localintervalinTF=" << interval
             << "indexinchunk=" << localintegrationInterval
             << "chunk=" << chunk
             << "timeframe=" << timeFrame
             << "IDC.=" << idcs
             << "IDC0.=" << idcsZero
             << "IDC1A=" << idcOneATmp
             << "IDC1C=" << idcOneCTmp
             << "IDCDeltaNoComp.=" << idcsDelta
             << "IDCDeltaMediumComp.=" << idcsDeltaMedium
             << "IDCDeltaHighComp.=" << idcsDeltaHigh
             << "pad.=" << vPad
             << "row.=" << vRow
             << "lx.=" << vXPos
             << "ly.=" << vYPos
             << "gx.=" << vGlobalXPos
             << "gy.=" << vGlobalYPos
             << "sector.=" << sectorv
             << "\n";
  }
  pcstream.Close();
}

void o2::tpc::IDCFactorization::calcIDCZero()
{
  const unsigned int nIDCsSide = mNIDCsPerSector * o2::tpc::SECTORSPERSIDE;
  mIDCZero.mIDCZero[Side::A].resize(nIDCsSide);
  mIDCZero.mIDCZero[Side::C].resize(nIDCsSide);

#pragma omp parallel for num_threads(sNThreads)
  for (unsigned int cru = 0; cru < mIDCs.size(); ++cru) {
    const o2::tpc::CRU cruTmp(cru);
    const unsigned int region = cruTmp.region();
    for (unsigned int timeframe = 0; timeframe < mTimeFrames; ++timeframe) {
      for (unsigned int idcs = 0; idcs < mIDCs[cru][timeframe].size(); ++idcs) {
        const unsigned int indexGlob = (idcs % mNIDCsPerCRU[region]) + mRegionOffs[region] + mNIDCsPerSector * cruTmp.sector();
        mIDCZero.fillValueIDCZero(mIDCs[cru][timeframe][idcs], cruTmp.side(), indexGlob % nIDCsSide);
      }
    }
  }
  std::transform(mIDCZero.mIDCZero[Side::A].begin(), mIDCZero.mIDCZero[Side::A].end(), mIDCZero.mIDCZero[Side::A].begin(), [norm = getNIntegrationIntervals()](auto& val) { return val / norm; });
  std::transform(mIDCZero.mIDCZero[Side::C].begin(), mIDCZero.mIDCZero[Side::C].end(), mIDCZero.mIDCZero[Side::C].begin(), [norm = getNIntegrationIntervals()](auto& val) { return val / norm; });
}

void o2::tpc::IDCFactorization::calcIDCOne()
{
  const unsigned int nIDCsSide = mNIDCsPerSector * SECTORSPERSIDE;
  const unsigned int integrationIntervals = getNIntegrationIntervals();
  mIDCOne.mIDCOne[Side::A].resize(integrationIntervals);
  mIDCOne.mIDCOne[Side::C].resize(integrationIntervals);
  const unsigned int crusPerSide = Mapper::NREGIONS * SECTORSPERSIDE;
#pragma omp parallel for num_threads(sNThreads)
  for (unsigned int cru = 0; cru < mIDCs.size(); ++cru) {
    const o2::tpc::CRU cruTmp(cru);
    const unsigned int region = cruTmp.region();
    unsigned int integrationIntervallast = 0;
    for (unsigned int timeframe = 0; timeframe < mTimeFrames; ++timeframe) {
      for (unsigned int idcs = 0; idcs < mIDCs[cru][timeframe].size(); ++idcs) {
        const unsigned int integrationInterval = idcs / mNIDCsPerCRU[region] + integrationIntervallast;
        const auto side = cruTmp.side();
        const unsigned int indexGlob = (idcs % mNIDCsPerCRU[region]) + mRegionOffs[region] + mNIDCsPerSector * cruTmp.sector();
        mIDCOne.mIDCOne[side][integrationInterval] += mIDCs[cru][timeframe][idcs] / (crusPerSide * mNIDCsPerCRU[region] * mIDCZero.mIDCZero[side][indexGlob % nIDCsSide]);
      }
      integrationIntervallast += mIDCs[cru][timeframe].size() / mNIDCsPerCRU[region];
    }
  }
}

void o2::tpc::IDCFactorization::calcIDCDelta()
{
  const unsigned int nIDCsSide = mNIDCsPerSector * SECTORSPERSIDE;
  for (unsigned int i = 0; i < getNChunks(); ++i) {
    const auto idcsSide = nIDCsSide * getNIntegrationIntervals(i);
    mIDCDelta[i].getIDCDelta(Side::A).resize(idcsSide);
    mIDCDelta[i].getIDCDelta(Side::C).resize(idcsSide);
  }

#pragma omp parallel for num_threads(sNThreads)
  for (unsigned int cru = 0; cru < mIDCs.size(); ++cru) {
    const o2::tpc::CRU cruTmp(cru);
    const unsigned int region = cruTmp.region();
    unsigned int integrationIntervallast = 0;
    unsigned int integrationIntervallastLocal = 0;
    unsigned int lastChunk = 0;

    for (unsigned int timeframe = 0; timeframe < mTimeFrames; ++timeframe) {
      const unsigned int chunk = getChunk(timeframe);
      if (lastChunk != chunk) {
        integrationIntervallastLocal = 0;
      }

      for (unsigned int idcs = 0; idcs < mIDCs[cru][timeframe].size(); ++idcs) {
        const unsigned int intervallocal = idcs / mNIDCsPerCRU[region];
        const unsigned int integrationIntervalGlobal = intervallocal + integrationIntervallast;
        const unsigned int integrationIntervalLocal = intervallocal + integrationIntervallastLocal;
        const auto side = cruTmp.side();
        const unsigned int indexGlob = (idcs % mNIDCsPerCRU[region]) + mRegionOffs[region] + mNIDCsPerSector * cruTmp.sector();
        const unsigned int indexGlobMod = indexGlob % nIDCsSide;
        const auto idcZero = mIDCZero.mIDCZero[side][indexGlobMod];
        const auto idcOne = mIDCOne.mIDCOne[side][integrationIntervalGlobal];
        const auto mult = idcZero * idcOne;
        const auto val = (mult > 0) ? mIDCs[cru][timeframe][idcs] / mult : 0;
        mIDCDelta[chunk].getIDCDelta(side)[indexGlobMod + integrationIntervalLocal * nIDCsSide] = val - 1;
      }

      const unsigned int intervals = mIDCs[cru][timeframe].size() / mNIDCsPerCRU[region];
      integrationIntervallast += intervals;
      integrationIntervallastLocal += intervals;
      lastChunk = chunk;
    }
  }
}

float o2::tpc::IDCFactorization::getIDCValUngrouped(const unsigned int sector, const unsigned int region, unsigned int urow, unsigned int upad, unsigned int integrationInterval) const
{
  unsigned int timeFrame = 0;
  unsigned int interval = 0;
  getTF(region, integrationInterval, timeFrame, interval);
  return mIDCs[sector * Mapper::NREGIONS + region][timeFrame][interval * mNIDCsPerCRU[region] + mOffsRow[region][getGroupedRow(region, urow)] + getGroupedPad(region, urow, upad)];
}

void o2::tpc::IDCFactorization::getTF(const unsigned int region, unsigned int integrationInterval, unsigned int& timeFrame, unsigned int& interval) const
{
  unsigned int nintervals = 0;
  unsigned int intervalTmp = 0;
  for (unsigned int tf = 0; tf < mTimeFrames; ++tf) {
    nintervals += mIDCs[region][tf].size() / mNIDCsPerCRU[region];
    if (integrationInterval < nintervals) {
      timeFrame = tf;
      interval = integrationInterval - intervalTmp;
      return;
    }
    intervalTmp = nintervals;
  }
}

void o2::tpc::IDCFactorization::factorizeIDCs()
{
  LOGP(info, "Using {} threads for factorization of IDCs", sNThreads);
  calcIDCZero();
  calcIDCOne();
  calcIDCDelta();
}

unsigned long o2::tpc::IDCFactorization::getNIntegrationIntervals(const unsigned int chunk) const
{
  std::size_t sum = 0;
  const auto firstTF = chunk * getNTFsPerChunk(0);
  for (unsigned int i = firstTF; i < firstTF + getNTFsPerChunk(chunk); ++i) {
    sum += mIDCs[0][i].size();
  }
  return sum / mNIDCsPerCRU[0];
}

unsigned long o2::tpc::IDCFactorization::getNIntegrationIntervals() const
{
  std::size_t sum = 0;
  for (auto&& idcsTF : mIDCs[0]) {
    sum += idcsTF.size();
  }
  return sum / mNIDCsPerCRU[0];
}

void o2::tpc::IDCFactorization::getLocalIntegrationInterval(const unsigned int region, const unsigned int integrationInterval, unsigned int& chunk, unsigned int& localintegrationInterval) const
{
  unsigned int nintervals = 0;
  unsigned int nitervalsChunk = 0;
  unsigned int globalTF = 0;
  for (unsigned int ichunk = 0; ichunk < getNChunks(); ++ichunk) {
    const auto nTFsPerChunk = getNTFsPerChunk(ichunk);
    for (unsigned int tf = 0; tf < nTFsPerChunk; ++tf) {
      nintervals += mIDCs[region][globalTF].size() / mNIDCsPerCRU[region];
      if (integrationInterval < nintervals) {
        chunk = getChunk(globalTF);
        localintegrationInterval = integrationInterval - nitervalsChunk;
        return;
      }
      ++globalTF;
    }
    nitervalsChunk = nintervals;
  }
}

unsigned int o2::tpc::IDCFactorization::getChunk(const unsigned int timeframe) const
{
  return timeframe / mTimeFramesDeltaIDC;
}

unsigned int o2::tpc::IDCFactorization::getNTFsPerChunk(const unsigned int chunk) const
{
  const unsigned int remain = mTimeFrames % mTimeFramesDeltaIDC;
  return ((chunk == getNChunks() - 1) && remain) ? remain : mTimeFramesDeltaIDC;
}

std::vector<unsigned int> o2::tpc::IDCFactorization::getIntegrationIntervalsPerTF(const unsigned int region) const
{
  std::vector<unsigned int> integrationIntervalsPerTF(mTimeFrames);
  for (unsigned int tf = 0; tf < mTimeFrames; ++tf) {
    integrationIntervalsPerTF[tf] = mIDCs[region][tf].size() / mNIDCsPerCRU[region];
  }
  return integrationIntervalsPerTF;
}

void o2::tpc::IDCFactorization::reset()
{
  for (auto& tf : mIDCs) {
    for (auto& idcs : tf) {
      idcs.clear();
    }
  }
}
