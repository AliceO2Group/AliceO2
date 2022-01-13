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

#include "TPCCalibration/IDCFactorization.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "Framework/Logger.h"
#include "TPCCalibration/IDCDrawHelper.h"
#include "TFile.h"
#include <functional>

o2::tpc::IDCFactorization::IDCFactorization(const std::array<unsigned char, Mapper::NREGIONS>& groupPads, const std::array<unsigned char, Mapper::NREGIONS>& groupRows, const std::array<unsigned char, Mapper::NREGIONS>& groupLastRowsThreshold, const std::array<unsigned char, Mapper::NREGIONS>& groupLastPadsThreshold, const unsigned int groupNotnPadsSectorEdges, const unsigned int timeFrames, const unsigned int timeframesDeltaIDC)
  : IDCGroupHelperSector{groupPads, groupRows, groupLastRowsThreshold, groupLastPadsThreshold, groupNotnPadsSectorEdges}, mTimeFrames{timeFrames}, mTimeFramesDeltaIDC{timeframesDeltaIDC}, mIDCDelta{timeFrames / timeframesDeltaIDC + (timeFrames % timeframesDeltaIDC != 0)}
{
  for (auto& idc : mIDCs) {
    idc.resize(mTimeFrames);
  }
}

void o2::tpc::IDCFactorization::dumpToFile(const char* outFileName, const char* outName) const
{
  TFile fOut(outFileName, "RECREATE");
  fOut.WriteObject(this, outName);
  fOut.Close();
}

void o2::tpc::IDCFactorization::dumpToTree(int integrationIntervals, const char* outFileName) const
{
  const Mapper& mapper = Mapper::instance();
  o2::utils::TreeStreamRedirector pcstream(outFileName, "RECREATE");
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
            const auto padTmp = (sector < SECTORSPERSIDE) ? ipad : (Mapper::PADSPERROW[region][irow] - ipad - 1); // C-Side is mirrored
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

void o2::tpc::IDCFactorization::calcIDCZero(const bool norm)
{
  const unsigned int nIDCsSide = mNIDCsPerSector * o2::tpc::SECTORSPERSIDE;
  mIDCZero.clear();
  mIDCZero.resize(nIDCsSide);

#pragma omp parallel for num_threads(sNThreads)
  for (unsigned int cru = 0; cru < mIDCs.size(); ++cru) {
    const o2::tpc::CRU cruTmp(cru);
    const unsigned int region = cruTmp.region();
    const auto factorIndexGlob = mRegionOffs[region] + mNIDCsPerSector * cruTmp.sector();
    for (unsigned int timeframe = 0; timeframe < mTimeFrames; ++timeframe) {
      for (unsigned int idcs = 0; idcs < mIDCs[cru][timeframe].size(); ++idcs) {
        const unsigned int indexGlob = (idcs % mNIDCsPerCRU[region]) + factorIndexGlob;
        if (norm) {
          mIDCs[cru][timeframe][idcs] *= Mapper::INVPADAREA[region];
        }
        mIDCZero.fillValueIDCZero(mIDCs[cru][timeframe][idcs], cruTmp.side(), indexGlob % nIDCsSide);
      }
    }
  }
  std::transform(mIDCZero.mIDCZero[Side::A].begin(), mIDCZero.mIDCZero[Side::A].end(), mIDCZero.mIDCZero[Side::A].begin(), [normVal = getNIntegrationIntervals()](auto& val) { return val / normVal; });
  std::transform(mIDCZero.mIDCZero[Side::C].begin(), mIDCZero.mIDCZero[Side::C].end(), mIDCZero.mIDCZero[Side::C].begin(), [normVal = getNIntegrationIntervals()](auto& val) { return val / normVal; });
}

void o2::tpc::IDCFactorization::calcIDCOne()
{
  const unsigned int nIDCsSide = mNIDCsPerSector * SECTORSPERSIDE;
  const unsigned int integrationIntervals = getNIntegrationIntervals();
  mIDCOne.clear();
  mIDCOne.resize(integrationIntervals);
  const unsigned int crusPerSide = Mapper::NREGIONS * SECTORSPERSIDE;

#pragma omp parallel for num_threads(sNThreads)
  for (unsigned int cru = 0; cru < mIDCs.size(); ++cru) {
    const o2::tpc::CRU cruTmp(cru);
    const unsigned int region = cruTmp.region();
    const auto factorIDCOne = crusPerSide * mNIDCsPerCRU[region];
    const auto side = cruTmp.side();
    const auto factorIndexGlob = mRegionOffs[region] + mNIDCsPerSector * cruTmp.sector();
    unsigned int integrationIntervallast = 0;
    for (unsigned int timeframe = 0; timeframe < mTimeFrames; ++timeframe) {
      for (unsigned int idcs = 0; idcs < mIDCs[cru][timeframe].size(); ++idcs) {
        const unsigned int integrationInterval = idcs / mNIDCsPerCRU[region] + integrationIntervallast;
        const unsigned int indexGlob = (idcs % mNIDCsPerCRU[region]) + factorIndexGlob;
        const auto idcZeroVal = mIDCZero.mIDCZero[side][indexGlob % nIDCsSide];
        if (idcZeroVal) {
          mIDCOne.mIDCOne[side][integrationInterval] += mIDCs[cru][timeframe][idcs] / (factorIDCOne * idcZeroVal);
        }
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
    const auto side = cruTmp.side();
    const auto factorIndexGlob = mRegionOffs[region] + mNIDCsPerSector * cruTmp.sector();
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
        const unsigned int indexGlob = (idcs % mNIDCsPerCRU[region]) + factorIndexGlob;
        const unsigned int indexGlobMod = indexGlob % nIDCsSide;
        const auto idcZero = mIDCZero.mIDCZero[side][indexGlobMod];
        const auto idcOne = mIDCOne.mIDCOne[side][integrationIntervalGlobal];
        const auto mult = idcZero * idcOne;
        const auto val = (mult > 0) ? mIDCs[cru][timeframe][idcs] / mult : 0;
        mIDCDelta[chunk].setIDCDelta(side, indexGlobMod + integrationIntervalLocal * nIDCsSide, val - 1.f);
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
  if (mIDCs[sector * Mapper::NREGIONS + region][timeFrame].empty()) {
    return 0.f;
  }

  const int index = interval * mNIDCsPerCRU[region] + mOffsRow[region][getGroupedRow(region, urow)] + getGroupedPad(region, urow, upad) + getOffsetForEdgePad(upad, urow, region);
  return mIDCs[sector * Mapper::NREGIONS + region][timeFrame][index];
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

void o2::tpc::IDCFactorization::factorizeIDCs(const bool norm)
{
  LOGP(info, "Using {} threads for factorization of IDCs", sNThreads);
  calcIDCZero(norm);
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
  std::vector<unsigned int> integrationIntervalsPerTF;
  integrationIntervalsPerTF.reserve(mTimeFrames);
  for (unsigned int tf = 0; tf < mTimeFrames; ++tf) {
    integrationIntervalsPerTF.emplace_back(mIDCs[region][tf].size() / mNIDCsPerCRU[region]);
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

void o2::tpc::IDCFactorization::drawIDCDeltaHelper(const bool type, const Sector sector, const unsigned int integrationInterval, const IDCDeltaCompression compression, const std::string filename) const
{
  std::function<float(const unsigned int, const unsigned int, const unsigned int, const unsigned int)> idcFunc;

  unsigned int chunk = 0;
  unsigned int localintegrationInterval = 0;
  getLocalIntegrationInterval(0, integrationInterval, chunk, localintegrationInterval);
  const std::string zAxisTitle = IDCDrawHelper::getZAxisTitle(IDCType::IDCDelta, compression);

  IDCDrawHelper::IDCDraw drawFun;
  switch (compression) {
    case IDCDeltaCompression::NO:
    default: {
      idcFunc = [this, chunk, localintegrationInterval](const unsigned int sector, const unsigned int region, const unsigned int irow, const unsigned int pad) {
        return this->getIDCDeltaVal(sector, region, irow, pad, chunk, localintegrationInterval);
      };
      drawFun.mIDCFunc = idcFunc;
      type ? IDCDrawHelper::drawSide(drawFun, sector.side(), zAxisTitle, filename) : IDCDrawHelper::drawSector(drawFun, 0, Mapper::NREGIONS, sector, zAxisTitle, filename);
      break;
    }
    case IDCDeltaCompression::MEDIUM: {
      const auto idcDeltaMedium = this->getIDCDeltaMediumCompressed(chunk);
      idcFunc = [this, &idcDeltaMedium, chunk, localintegrationInterval = localintegrationInterval](const unsigned int sector, const unsigned int region, const unsigned int irow, const unsigned int pad) {
        return idcDeltaMedium.getValue(Sector(sector).side(), this->getIndexUngrouped(sector, region, irow, pad, localintegrationInterval));
      };
      drawFun.mIDCFunc = idcFunc;
      type ? IDCDrawHelper::drawSide(drawFun, sector.side(), zAxisTitle, filename) : IDCDrawHelper::drawSector(drawFun, 0, Mapper::NREGIONS, sector, zAxisTitle, filename);
      break;
    }
    case IDCDeltaCompression::HIGH: {
      const auto idcDeltaHigh = this->getIDCDeltaHighCompressed(chunk);
      idcFunc = [this, &idcDeltaHigh, chunk, localintegrationInterval](const unsigned int sector, const unsigned int region, const unsigned int irow, const unsigned int pad) {
        return idcDeltaHigh.getValue(Sector(sector).side(), this->getIndexUngrouped(sector, region, irow, pad, localintegrationInterval));
      };
      drawFun.mIDCFunc = idcFunc;
      type ? IDCDrawHelper::drawSide(drawFun, sector.side(), zAxisTitle, filename) : IDCDrawHelper::drawSector(drawFun, 0, Mapper::NREGIONS, sector, zAxisTitle, filename);
      break;
    }
  }
}

void o2::tpc::IDCFactorization::drawIDCZeroHelper(const bool type, const Sector sector, const std::string filename) const
{
  std::function<float(const unsigned int, const unsigned int, const unsigned int, const unsigned int)> idcFunc = [this](const unsigned int sector, const unsigned int region, const unsigned int irow, const unsigned int pad) {
    return this->getIDCZeroVal(sector, region, irow, pad);
  };

  IDCDrawHelper::IDCDraw drawFun;
  drawFun.mIDCFunc = idcFunc;
  const std::string zAxisTitle = IDCDrawHelper::getZAxisTitle(IDCType::IDCZero);
  type ? IDCDrawHelper::drawSide(drawFun, sector.side(), zAxisTitle, filename) : IDCDrawHelper::drawSector(drawFun, 0, Mapper::NREGIONS, sector, zAxisTitle, filename);
}

void o2::tpc::IDCFactorization::drawIDCHelper(const bool type, const Sector sector, const unsigned int integrationInterval, const std::string filename) const
{
  std::function<float(const unsigned int, const unsigned int, const unsigned int, const unsigned int)> idcFunc = [this, integrationInterval](const unsigned int sector, const unsigned int region, const unsigned int irow, const unsigned int pad) {
    return this->getIDCValUngrouped(sector, region, irow, pad, integrationInterval);
  };

  IDCDrawHelper::IDCDraw drawFun;
  drawFun.mIDCFunc = idcFunc;

  const std::string zAxisTitleDraw = IDCDrawHelper::getZAxisTitle(IDCType::IDC);
  type ? IDCDrawHelper::drawSide(drawFun, sector.side(), zAxisTitleDraw, filename) : IDCDrawHelper::drawSector(drawFun, 0, Mapper::NREGIONS, sector, zAxisTitleDraw, filename);
}
