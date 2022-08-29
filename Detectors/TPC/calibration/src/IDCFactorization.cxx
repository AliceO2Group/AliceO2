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
#include "TPCCalibration/RobustAverage.h"
#include "TFile.h"
#include "TPCBase/CalDet.h"
#include <functional>
#include "MemoryResources/MemoryResources.h"

#if (defined(WITH_OPENMP) || defined(_OPENMP))
#include <omp.h>
#endif

o2::tpc::IDCFactorization::IDCFactorization(const std::array<unsigned char, Mapper::NREGIONS>& groupPads, const std::array<unsigned char, Mapper::NREGIONS>& groupRows, const std::array<unsigned char, Mapper::NREGIONS>& groupLastRowsThreshold, const std::array<unsigned char, Mapper::NREGIONS>& groupLastPadsThreshold, const unsigned int groupNotnPadsSectorEdges, const unsigned int timeFrames, const unsigned int timeframesDeltaIDC, const std::vector<uint32_t>& crus)
  : IDCGroupHelperSector{groupPads, groupRows, groupLastRowsThreshold, groupLastPadsThreshold, groupNotnPadsSectorEdges}, mTimeFrames{timeFrames}, mTimeFramesDeltaIDC{timeframesDeltaIDC}, mCRUs{crus}
{
  mSides = o2::tpc::IDCFactorization::getSides(crus);
  if (mSides.size() == 1) {
    mSideIndex[1] = 0;
  }

  for (int i = 0; i < mSides.size(); ++i) {
    mIDCDelta.emplace_back(std::vector<IDCDelta<float>>{timeFrames / timeframesDeltaIDC + (timeFrames % timeframesDeltaIDC != 0)});
  }

  mIDCZero.resize(mSides.size());
  mIDCOne.resize(mSides.size());

  for (auto& idc : mIDCs) {
    idc.resize(mTimeFrames);
  }
  // check if the input IDCs are grouped
  for (int region = 0; region < Mapper::NREGIONS; ++region) {
    if (mNIDCsPerCRU[region] != Mapper::PADSPERREGION[region]) {
      mInputGrouped = true;
      break;
    }
  }
}

o2::tpc::IDCFactorization::~IDCFactorization() = default;

std::vector<o2::tpc::Side> o2::tpc::IDCFactorization::getSides(const std::vector<uint32_t>& crus)
{
  std::unordered_map<o2::tpc::Side, bool> map;
  for (auto cru : crus) {
    map[CRU(cru).side()] = true;
  }

  std::vector<o2::tpc::Side> sides;
  if (!(map.find(o2::tpc::Side::A) == map.end())) {
    sides.emplace_back(o2::tpc::Side::A);
  }
  if (!(map.find(o2::tpc::Side::C) == map.end())) {
    sides.emplace_back(o2::tpc::Side::C);
  }
  return sides;
}

void o2::tpc::IDCFactorization::dumpToFile(const char* outFileName, const char* outName) const
{
  TFile fOut(outFileName, "RECREATE");
  fOut.WriteObject(this, outName);
  fOut.Close();
}

void o2::tpc::IDCFactorization::dumpIDCZeroToFile(const Side side, const char* outFileName, const char* outName) const
{
  TFile fOut(outFileName, "RECREATE");
  fOut.WriteObject(&mIDCZero[mSideIndex[side]], outName);
  fOut.Close();
}

void o2::tpc::IDCFactorization::dumpIDCOneToFile(const Side side, const char* outFileName, const char* outName) const
{
  TFile fOut(outFileName, "RECREATE");
  fOut.WriteObject(&mIDCOne[mSideIndex[side]], outName);
  fOut.Close();
}

void o2::tpc::IDCFactorization::dumpToTree(int integrationIntervals, const char* outFileName) const
{
  const Mapper& mapper = Mapper::instance();
  o2::utils::TreeStreamRedirector pcstream(outFileName, "RECREATE");
  pcstream.GetFile()->cd();

  for (auto side : mSides) {
    if (integrationIntervals <= 0) {
      integrationIntervals = static_cast<int>(getNIntegrationIntervals());
    }

    std::vector<float> idcOne = getIDCOneVec(side);
    for (unsigned int integrationInterval = 0; integrationInterval < integrationIntervals; ++integrationInterval) {
      const unsigned int nIDCsSector = Mapper::getPadsInSector() * SECTORSPERSIDE;
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
      getLocalIntegrationInterval(integrationInterval, chunk, localintegrationInterval);

      unsigned int index = 0;
      const auto idcDeltaMedium = getIDCDeltaMediumCompressed(chunk, side);
      const auto idcDeltaHigh = getIDCDeltaHighCompressed(chunk, side);

      unsigned int sectorStart = (side == Side::A) ? 0 : SECTORSPERSIDE;
      unsigned int sectorEnd = (side == Side::A) ? SECTORSPERSIDE : Mapper::NSECTORS;
      for (unsigned int sector = sectorStart; sector < sectorEnd; ++sector) {
        for (unsigned int region = 0; region < Mapper::NREGIONS; ++region) {
          for (unsigned int irow = 0; irow < Mapper::ROWSPERREGION[region]; ++irow) {
            for (unsigned int ipad = 0; ipad < Mapper::PADSPERROW[region][irow]; ++ipad) {
              const auto padNum = Mapper::getGlobalPadNumber(irow, ipad, region);
              const auto padTmp = (side == Side::A) ? ipad : (Mapper::PADSPERROW[region][irow] - ipad - 1); // C-Side is mirrored
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
              idcsDeltaMedium[index] = idcDeltaMedium.getValue(getIndexUngrouped(sector, region, irow, padTmp, localintegrationInterval));
              idcsDeltaHigh[index] = idcDeltaHigh.getValue(getIndexUngrouped(sector, region, irow, padTmp, localintegrationInterval));
              sectorv[index++] = sector;
            }
          }
        }
      }

      float idcOneTmp = idcOne[integrationInterval];
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
               << "IDC1=" << idcOneTmp
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
               << "side.=" << side
               << "\n";
    }
  }
  pcstream.Close();
}

void o2::tpc::IDCFactorization::calcIDCZero(const bool norm)
{
  const unsigned int nIDCsSide = mNIDCsPerSector * o2::tpc::SECTORSPERSIDE;
  for (auto& idcZero : mIDCZero) {
    idcZero.clear();
    idcZero.resize(nIDCsSide);
  }

#pragma omp parallel for num_threads(sNThreads)
  for (unsigned int cruInd = 0; cruInd < mCRUs.size(); ++cruInd) {
    const unsigned int cru = mCRUs[cruInd];
    const o2::tpc::CRU cruTmp(cru);
    const auto side = cruTmp.side();
    const unsigned int region = cruTmp.region();
    const auto factorIndexGlob = mRegionOffs[region] + mNIDCsPerSector * (cruTmp.sector() % o2::tpc::SECTORSPERSIDE);
    for (unsigned int timeframe = 0; timeframe < mTimeFrames; ++timeframe) {
      for (unsigned int idcs = 0; idcs < mIDCs[cru][timeframe].size(); ++idcs) {
        if (mIDCs[cru][timeframe][idcs] == -1) {
          continue;
        }
        if (norm) {
          mIDCs[cru][timeframe][idcs] *= Mapper::INVPADAREA[region];
        }
        const unsigned int indexGlob = (idcs % mNIDCsPerCRU[region]) + factorIndexGlob;
        mIDCZero[mSideIndex[side]].fillValueIDCZero(mIDCs[cru][timeframe][idcs], indexGlob);
      }
    }
  }

// perform normalization per CRU (in case some CRUs lack data)
#pragma omp parallel for num_threads(sNThreads)
  for (unsigned int cruInd = 0; cruInd < mCRUs.size(); ++cruInd) {
    const unsigned int cru = mCRUs[cruInd];
    const o2::tpc::CRU cruTmp(cru);
    const auto side = cruTmp.side();
    const unsigned int region = cruTmp.region();
    const auto indexStart = mRegionOffs[region] + mNIDCsPerSector * (cruTmp.sector() % o2::tpc::SECTORSPERSIDE);
    const auto indexEnd = indexStart + mNIDCsPerCRU[region];

    const auto normFac = getNIntegrationIntervals(cru);
    if (normFac == 0) {
      LOGP(info, "number of integration intervals is zero for CRU {}! Skipping normalization of IDC0", cru);
      continue;
    }

    std::transform(mIDCZero[mSideIndex[side]].mIDCZero.begin() + indexStart, mIDCZero[mSideIndex[side]].mIDCZero.begin() + indexEnd, mIDCZero[mSideIndex[side]].mIDCZero.begin() + indexStart, [normVal = normFac](auto& val) { return val / normVal; });
  }
}

void o2::tpc::IDCFactorization::fillIDCZeroDeadPads()
{
#pragma omp parallel for num_threads(sNThreads)
  for (unsigned int cruInd = 0; cruInd < mCRUs.size(); ++cruInd) {
    const unsigned int cru = mCRUs[cruInd];
    const o2::tpc::CRU cruTmp(cru);
    const Side side = cruTmp.side();
    const int region = cruTmp.region();
    const int sector = cruTmp.sector();
    std::vector<unsigned int> padTmp;
    padTmp.reserve(Mapper::PADSPERROW[region].back());
    for (unsigned int lrow = 0; lrow < Mapper::ROWSPERREGION[region]; ++lrow) {
      float idcRow = 0;
      int count = 0;
      const unsigned int integrationInterval = 0;
      // loop over pad row and calculate mean of IDC0
      for (unsigned int pad = 0; pad < Mapper::PADSPERROW[region][lrow]; ++pad) {
        const auto index = getIndexUngrouped(sector, region, lrow, pad, integrationInterval);
        const auto idcZeroVal = mIDCZero[mSideIndex[side]].mIDCZero[index];
        if (idcZeroVal > 0) {
          const auto globalPad = Mapper::getGlobalPadNumber(lrow, pad, region);
          const float gain = (mGainMap ? mGainMap->getValue(sector, globalPad) : 1);
          idcRow += idcZeroVal / gain;
          ++count;
        } else {
          padTmp.emplace_back(pad);
        }
      }
      // loop over dead pads and set value
      for (const auto pad : padTmp) {
        const float meanIDC0 = idcRow / count;
        const auto globalPad = Mapper::getGlobalPadNumber(lrow, pad, region);
        const float gain = (mGainMap ? mGainMap->getValue(sector, globalPad) : 1);
        const float idcZero = gain * meanIDC0;
        const auto index = getIndexUngrouped(sector, region, lrow, pad, integrationInterval);
        mIDCZero[mSideIndex[side]].setValueIDCZero(idcZero, index);
      }
      padTmp.clear();
    }
  }
}

void o2::tpc::IDCFactorization::calcIDCOne()
{
  const unsigned int integrationIntervals = getNIntegrationIntervals();
  for (auto& idcOne : mIDCOne) {
    idcOne.clear();
    idcOne.resize(integrationIntervals);
  }

  std::vector<std::vector<std::vector<float>>> idcOneSafe(mSides.size());
  std::vector<std::vector<std::vector<unsigned int>>> weightsSafe(mSides.size());

  for (auto& vecSide : idcOneSafe) {
    vecSide.resize(sNThreads);
    for (auto& interavalvec : vecSide) {
      interavalvec.resize(integrationIntervals);
    }
  }

  for (auto& vecSide : weightsSafe) {
    vecSide.resize(sNThreads);
    for (auto& interavalvec : vecSide) {
      interavalvec.resize(integrationIntervals);
    }
  }

#pragma omp parallel for num_threads(sNThreads)
  for (unsigned int cruInd = 0; cruInd < mCRUs.size(); ++cruInd) {
    const unsigned int cru = mCRUs[cruInd];
#ifdef WITH_OPENMP
    const int ithread = omp_get_thread_num();
#else
    const int ithread = 0;
#endif
    const o2::tpc::CRU cruTmp(cru);
    const unsigned int region = cruTmp.region();
    const auto side = cruTmp.side();
    const unsigned int indexSide = mSideIndex[side];

    const auto factorIndexGlob = mRegionOffs[region] + mNIDCsPerSector * (cruTmp.sector() % o2::tpc::SECTORSPERSIDE);
    unsigned int integrationIntervalOffset = 0;
    for (unsigned int timeframe = 0; timeframe < mTimeFrames; ++timeframe) {
      calcIDCOne(mIDCs[cru][timeframe], mNIDCsPerCRU[region], integrationIntervalOffset, factorIndexGlob, cru, idcOneSafe[indexSide][ithread], weightsSafe[indexSide][ithread], &mIDCZero[indexSide], mInputGrouped ? nullptr : mPadFlagsMap.get(), mUsePadStatusMap);
      integrationIntervalOffset += mIntegrationIntervalsPerTF[timeframe];
    }
  }

#pragma omp parallel for num_threads(sNThreads)
  for (int side = 0; side < mSides.size(); ++side) {
    const unsigned int indexSide = mSideIndex[side];
    for (int i = 1; i < sNThreads; ++i) {
      std::transform(idcOneSafe[indexSide].front().begin(), idcOneSafe[indexSide].front().end(), idcOneSafe[indexSide][i].begin(), idcOneSafe[indexSide].front().begin(), std::plus<float>());
      std::transform(weightsSafe[indexSide].front().begin(), weightsSafe[indexSide].front().end(), weightsSafe[indexSide][i].begin(), weightsSafe[indexSide].front().begin(), std::plus<unsigned int>());
    }

    // replace all 0 with 1 to avoid division by 0
    std::replace(weightsSafe[indexSide].front().begin(), weightsSafe[indexSide].front().end(), 0, 1);

    // move IDC1 to member
    mIDCOne[indexSide].mIDCOne = std::move(idcOneSafe[indexSide].front());

    // normalize IDC1 to number of IDC values used
    std::transform(mIDCOne[indexSide].mIDCOne.begin(), mIDCOne[indexSide].mIDCOne.end(), weightsSafe[side].front().begin(), mIDCOne[indexSide].mIDCOne.begin(), std::divides<float>());

    // replace all 0 i.e. where all data for one TF was dropped
    std::replace(mIDCOne[indexSide].mIDCOne.begin(), mIDCOne[indexSide].mIDCOne.end(), 0, 1);
  }
}

template <typename DataVec>
void o2::tpc::IDCFactorization::calcIDCOne(const DataVec& idcsData, const int idcsPerCRU, const int integrationIntervalOffset, const unsigned int indexOffset, const CRU cru, std::vector<float>& idcOneTmp, std::vector<unsigned int>& weights, const IDCZero* idcZero, const CalDet<PadFlags>* flagMap, const bool usePadStatusMap)
{
  for (unsigned int idcs = 0; idcs < idcsData.size(); ++idcs) {
    if (idcsData[idcs] == -1) {
      continue;
    }

    const unsigned int integrationInterval = idcs / idcsPerCRU + integrationIntervalOffset;
    const unsigned int localPad = (idcs % idcsPerCRU);
    const unsigned int indexGlob = localPad + indexOffset;
    const auto idcZeroVal = idcZero ? idcZero->mIDCZero[indexGlob] : 1;

    // check pad in case of input is not grouped
    if (usePadStatusMap && flagMap) {
      const o2::tpc::PadFlags flag = flagMap->getCalArray(cru).getValue(localPad);
      if ((flag & PadFlags::flagSkip) == PadFlags::flagSkip) {
        continue;
      }
    }

    idcOneTmp[integrationInterval] += (idcZeroVal == 0) ? idcsData[idcs] : idcsData[idcs] / idcZeroVal;
    ++weights[integrationInterval];
  }
}

void o2::tpc::IDCFactorization::calcIDCDelta()
{
  const unsigned int nIDCsSide = mNIDCsPerSector * SECTORSPERSIDE;
  for (auto side : mSides) {
    for (unsigned int i = 0; i < getNChunks(side); ++i) {
      const auto idcsSide = nIDCsSide * getNIntegrationIntervalsInChunk(i);
      mIDCDelta[mSideIndex[side]][i].getIDCDelta().clear();
      mIDCDelta[mSideIndex[side]][i].getIDCDelta().resize(idcsSide);
    }
  }

#pragma omp parallel for num_threads(sNThreads)
  for (unsigned int cruInd = 0; cruInd < mCRUs.size(); ++cruInd) {
    const unsigned int cru = mCRUs[cruInd];
    const o2::tpc::CRU cruTmp(cru);
    const unsigned int region = cruTmp.region();
    const auto side = cruTmp.side();
    const auto factorIndexGlob = mRegionOffs[region] + mNIDCsPerSector * (cruTmp.sector() % o2::tpc::SECTORSPERSIDE);
    unsigned int integrationIntervallast = 0;
    unsigned int integrationIntervallastLocal = 0;
    unsigned int lastChunk = 0;

    for (unsigned int timeframe = 0; timeframe < mTimeFrames; ++timeframe) {
      const unsigned int chunk = getChunk(timeframe);
      if (lastChunk != chunk) {
        integrationIntervallastLocal = 0;
      }

      for (unsigned int idcs = 0; idcs < mIDCs[cru][timeframe].size(); ++idcs) {
        if (mIDCs[cru][timeframe][idcs] == -1) {
          continue;
        }
        const unsigned int intervallocal = idcs / mNIDCsPerCRU[region];
        const unsigned int integrationIntervalGlobal = intervallocal + integrationIntervallast;
        const unsigned int integrationIntervalLocal = intervallocal + integrationIntervallastLocal;
        const unsigned int localPad = idcs % mNIDCsPerCRU[region];
        const unsigned int indexGlob = localPad + factorIndexGlob;
        const auto idcZero = mIDCZero[mSideIndex[side]].mIDCZero[indexGlob];
        const auto idcOne = mIDCOne[mSideIndex[side]].mIDCOne[integrationIntervalGlobal];
        const auto mult = idcZero * idcOne;
        auto val = (mult != 0) ? mIDCs[cru][timeframe][idcs] / mult - 1 : 0;
        if (mUsePadStatusMap && !mInputGrouped && mPadFlagsMap) {
          const o2::tpc::PadFlags flag = mPadFlagsMap->getCalArray(cru).getValue(localPad);
          if ((flag & PadFlags::flagSkip) == PadFlags::flagSkip) {
            val = 0;
          }
        }
        mIDCDelta[mSideIndex[side]][chunk].setIDCDelta(indexGlob + integrationIntervalLocal * nIDCsSide, val);
      }

      const unsigned int intervals = mIntegrationIntervalsPerTF[timeframe];
      integrationIntervallast += intervals;
      integrationIntervallastLocal += intervals;
      lastChunk = chunk;
    }
  }
}

void o2::tpc::IDCFactorization::createStatusMap()
{
  const static auto& paramIDCGroup = ParameterIDCGroup::Instance();
  mPadFlagsMap = std::make_unique<CalDet<PadFlags>>(CalDet<PadFlags>("flags", PadSubset::Region));
#pragma omp parallel for num_threads(sNThreads)
  for (unsigned int cruInd = 0; cruInd < mCRUs.size(); ++cruInd) {
    const unsigned int cru = mCRUs[cruInd];
    const o2::tpc::CRU cruTmp(cru);
    const Side side = cruTmp.side();
    const int region = cruTmp.region();
    const int sector = cruTmp.sector();
    const auto maxValues = Mapper::PADSPERROW[region].back();
    o2::tpc::RobustAverage average(maxValues);

    for (unsigned int lrow = 0; lrow < Mapper::ROWSPERREGION[region]; ++lrow) {

      // loop over pads in row in the first iteration and calculate median at the end
      // in the second iteration check if the IDC value is to far away from the median
      for (int iter = 0; iter < 2; ++iter) {
        const unsigned int integrationInterval = 0;
        const float median = (iter == 1) ? average.getMedian() : 0;
        const float stdDev = (iter == 1) ? average.getStdDev() : 0;
        // loop over pad row and calculate mean of IDC0
        for (unsigned int pad = 0; pad < Mapper::PADSPERROW[region][lrow]; ++pad) {
          const auto index = getIndexUngrouped(sector, region, lrow, pad, integrationInterval);
          const auto idcZeroVal = mIDCZero[mSideIndex[side]].mIDCZero[index];

          if (iter == 0) {
            // exclude dead pads
            if (idcZeroVal != -1) {
              average.addValue(idcZeroVal);
            }
          } else {
            const unsigned int padInRegion = Mapper::OFFSETCRULOCAL[region][lrow] + pad;
            o2::tpc::PadFlags flag = o2::tpc::PadFlags::flagGoodPad;
            if (idcZeroVal == -1) {
              flag = o2::tpc::PadFlags::flagDeadPad | o2::tpc::PadFlags::flagSkip | mPadFlagsMap->getCalArray(cru).getValue(padInRegion);
            } else if (idcZeroVal > median + stdDev * paramIDCGroup.maxIDC0Median) {
              flag = o2::tpc::PadFlags::flagHighPad | o2::tpc::PadFlags::flagSkip | mPadFlagsMap->getCalArray(cru).getValue(padInRegion);
            } else if (idcZeroVal < median - stdDev * paramIDCGroup.minIDC0Median) {
              flag = o2::tpc::PadFlags::flagLowPad | o2::tpc::PadFlags::flagSkip | mPadFlagsMap->getCalArray(cru).getValue(padInRegion);
            }
            mPadFlagsMap->getCalArray(cru).setValue(padInRegion, flag);
          }
        } // loop over pads in row
      }   // end iteration
      average.clear();
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
  const auto& intervalsPerTF = mIntegrationIntervalsPerTF.empty() ? getAllIntegrationIntervalsPerTF() : mIntegrationIntervalsPerTF;
  unsigned int nintervals = 0;
  unsigned int intervalTmp = 0;
  for (unsigned int tf = 0; tf < mTimeFrames; ++tf) {
    nintervals += intervalsPerTF[tf];
    if (integrationInterval < nintervals) {
      timeFrame = tf;
      interval = integrationInterval - intervalTmp;
      return;
    }
    intervalTmp = nintervals;
  }
}

std::vector<unsigned int> o2::tpc::IDCFactorization::getAllIntegrationIntervalsPerTF() const
{
  // find three consecutive time frames
  const int nMaxTFs = 3;
  const int nTFs = (mTimeFrames >= nMaxTFs) ? nMaxTFs : mTimeFrames;

  // store the number of integration intervals like 10 -> 11 -> 11
  std::vector<unsigned int> ordering;
  ordering.reserve(nTFs);
  unsigned int order = 0;
  int firstTF = 0;

  // find ordering
  for (int iTF = 0; iTF < mTimeFrames; ++iTF) {
    bool found = false;
    for (unsigned int cruInd = 0; cruInd < mCRUs.size(); ++cruInd) {
      const unsigned int cru = mCRUs[cruInd];
      if (!mIDCs[cru][iTF].empty()) {
        order = mIDCs[cru][iTF].size() / mNIDCsPerCRU[CRU(cru).region()];
        ordering.emplace_back(order);
        found = true;
        break;
      }
    }
    if (ordering.size() == nTFs) {
      firstTF = iTF - nTFs + 1;
      break;
    }
    if (!found) {
      ordering.clear();
    }
  }

  // in case TFs < 3
  if ((nTFs == mTimeFrames) && (ordering.size() == nTFs)) {
    return ordering;
  }

  if (ordering.size() != nTFs) {
    LOGP(warning, "Couldnt find {} consecutive TFs with data", nTFs);
    ordering = std::vector<unsigned int>(nTFs, order);
    firstTF = 0;
  } else {
    std::sort(ordering.begin(), ordering.end());
  }

  std::vector<unsigned int> integrationIntervalsPerTF(mTimeFrames);
  for (int iter = 0; iter < 2; ++iter) {
    const int end = (iter == 0) ? (mTimeFrames - firstTF) : firstTF;
    for (int iTFLoop = 0; iTFLoop < end; ++iTFLoop) {
      const int iTF = (iter == 0) ? (firstTF + iTFLoop) : (firstTF - iTFLoop - 1);
      int intervalsInTF = -1;
      for (unsigned int cruInd = 0; cruInd < mCRUs.size(); ++cruInd) {
        const unsigned int cru = mCRUs[cruInd];
        if (!mIDCs[cru][iTF].empty()) {
          intervalsInTF = mIDCs[cru][iTF].size() / mNIDCsPerCRU[CRU(cru).region()];
          break;
        }
      }

      if (intervalsInTF == -1) {
        if ((ordering.size() != nTFs)) {
          if (iTF == 0) {
            intervalsInTF = ordering.front();
          } else if (iTF == 1) {
            intervalsInTF = ordering[1];
          }
        } else {
          const auto idcsPerCRU = mNIDCsPerCRU[CRU(mCRUs.front()).region()];
          const int m1Val = (iter == 0) ? -1 : +1;
          const int intervalsLastm1 = integrationIntervalsPerTF[iTF + m1Val];
          const int intervalsLastm2 = integrationIntervalsPerTF[iTF + 2 * m1Val];
          if ((intervalsLastm1 == ordering[1]) && (intervalsLastm2 == ordering[2])) {
            intervalsInTF = ordering[0];
          } else {
            intervalsInTF = ordering[1];
          }
        }
      }

      if (intervalsInTF < -1) {
        LOGP(warning, "interval is smaller than 0!");
        continue;
      }

      integrationIntervalsPerTF[iTF] = intervalsInTF;
    }
  }
  return integrationIntervalsPerTF;
}

void o2::tpc::IDCFactorization::factorizeIDCs(const bool norm)
{
  using timer = std::chrono::high_resolution_clock;

  LOGP(info, "Using {} threads for factorization of IDCs", sNThreads);
  LOGP(info, "Calculating IDC0");

  auto start = timer::now();
  calcIDCZero(norm);
  auto stop = timer::now();
  std::chrono::duration<float> time = stop - start;
  float totalTime = time.count();
  LOGP(info, "IDCZero time: {}", time.count());

  if (!mInputGrouped) {
    LOGP(info, "Creating pad status map");
    start = timer::now();
    createStatusMap();
    stop = timer::now();
    time = stop - start;
    LOGP(info, "Pad-by-pad status map time: {}", time.count());
    totalTime += time.count();
  }

  start = timer::now();
  mIntegrationIntervalsPerTF = getAllIntegrationIntervalsPerTF();
  stop = timer::now();
  time = stop - start;
  totalTime = time.count();
  LOGP(info, "Getting integration intervals for all TFs time: {}", time.count());

  LOGP(info, "Calculating IDC1");
  start = timer::now();
  calcIDCOne();
  stop = timer::now();
  time = stop - start;
  LOGP(info, "IDC1 time: {}", time.count());
  totalTime += time.count();

  LOGP(info, "Calculating IDCDelta");
  start = timer::now();
  calcIDCDelta();
  stop = timer::now();
  time = stop - start;
  LOGP(info, "IDCDelta time: {}", time.count());
  totalTime += time.count();

  LOGP(info, "Factorization done. Total time: {}", totalTime);
}

unsigned long o2::tpc::IDCFactorization::getNIntegrationIntervalsInChunk(const unsigned int chunk) const
{
  const auto firstTF = chunk * getNTFsPerChunk(0);
  const auto& intervalsPerTF = mIntegrationIntervalsPerTF.empty() ? getAllIntegrationIntervalsPerTF() : mIntegrationIntervalsPerTF;
  const auto intervals = std::reduce(intervalsPerTF.begin() + firstTF, intervalsPerTF.begin() + firstTF + getNTFsPerChunk(chunk));
  return intervals;
}

unsigned long o2::tpc::IDCFactorization::getNIntegrationIntervalsToChunk(const unsigned int chunk) const
{
  const auto chunkTF = chunk * getNTFsPerChunk(0);
  const auto& intervalsPerTF = mIntegrationIntervalsPerTF.empty() ? getAllIntegrationIntervalsPerTF() : mIntegrationIntervalsPerTF;
  const auto intervals = std::reduce(intervalsPerTF.begin(), intervalsPerTF.begin() + chunkTF);
  return intervals;
}

unsigned long o2::tpc::IDCFactorization::getNIntegrationIntervals(const int cru) const
{
  std::size_t sum = 0;
  for (auto&& idcsTF : mIDCs[cru]) {
    sum += idcsTF.size();
  }
  return sum / mNIDCsPerCRU[CRU(cru).region()];
}

unsigned long o2::tpc::IDCFactorization::getNIntegrationIntervals() const
{
  const auto& intervalsPerTF = mIntegrationIntervalsPerTF.empty() ? getAllIntegrationIntervalsPerTF() : mIntegrationIntervalsPerTF;
  const auto intervals = std::reduce(intervalsPerTF.begin(), intervalsPerTF.end());
  return intervals;
}

void o2::tpc::IDCFactorization::getLocalIntegrationInterval(const unsigned int integrationInterval, unsigned int& chunk, unsigned int& localintegrationInterval) const
{
  const auto& intervalsPerTF = mIntegrationIntervalsPerTF.empty() ? getAllIntegrationIntervalsPerTF() : mIntegrationIntervalsPerTF;
  unsigned int nintervals = 0;
  unsigned int nitervalsChunk = 0;
  unsigned int globalTF = 0;
  for (unsigned int ichunk = 0; ichunk < getNChunks(mSides.front()); ++ichunk) {
    const auto nTFsPerChunk = getNTFsPerChunk(ichunk);
    for (unsigned int tf = 0; tf < nTFsPerChunk; ++tf) {
      nintervals += intervalsPerTF[globalTF];
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
  return ((chunk == getNChunks(mSides.front()) - 1) && remain) ? remain : mTimeFramesDeltaIDC;
}

std::vector<unsigned int> o2::tpc::IDCFactorization::getIntegrationIntervalsPerTF(const int cru) const
{
  const uint32_t cruTmp = (cru < 0) ? mCRUs.front() : cru;
  const auto region = CRU(cruTmp).region();
  std::vector<unsigned int> integrationIntervalsPerTF;
  integrationIntervalsPerTF.reserve(mTimeFrames);
  for (unsigned int tf = 0; tf < mTimeFrames; ++tf) {
    integrationIntervalsPerTF.emplace_back(mIDCs[cruTmp][tf].size() / mNIDCsPerCRU[region]);
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

void o2::tpc::IDCFactorization::drawIDCDeltaHelper(const bool type, const Sector sector, const unsigned int integrationInterval, const IDCDeltaCompression compression, const std::string filename, const float minZ, const float maxZ) const
{
  std::function<float(const unsigned int, const unsigned int, const unsigned int, const unsigned int)> idcFunc;

  unsigned int chunk = 0;
  unsigned int localintegrationInterval = 0;
  getLocalIntegrationInterval(integrationInterval, chunk, localintegrationInterval);
  const std::string zAxisTitle = IDCDrawHelper::getZAxisTitle(IDCType::IDCDelta, compression);

  IDCDrawHelper::IDCDraw drawFun;
  switch (compression) {
    case IDCDeltaCompression::NO:
    default: {
      idcFunc = [this, chunk, localintegrationInterval](const unsigned int sector, const unsigned int region, const unsigned int irow, const unsigned int pad) {
        return this->getIDCDeltaVal(sector, region, irow, pad, chunk, localintegrationInterval);
      };
      drawFun.mIDCFunc = idcFunc;
      type ? IDCDrawHelper::drawSide(drawFun, sector.side(), zAxisTitle, filename, minZ, maxZ) : IDCDrawHelper::drawSector(drawFun, 0, Mapper::NREGIONS, sector, zAxisTitle, filename, minZ, maxZ);
      break;
    }
    case IDCDeltaCompression::MEDIUM: {
      const auto idcDeltaMedium = this->getIDCDeltaMediumCompressed(chunk, sector.side());
      idcFunc = [this, &idcDeltaMedium, chunk, localintegrationInterval = localintegrationInterval](const unsigned int sector, const unsigned int region, const unsigned int irow, const unsigned int pad) {
        return idcDeltaMedium.getValue(this->getIndexUngrouped(sector, region, irow, pad, localintegrationInterval));
      };
      drawFun.mIDCFunc = idcFunc;
      type ? IDCDrawHelper::drawSide(drawFun, sector.side(), zAxisTitle, filename, minZ, maxZ) : IDCDrawHelper::drawSector(drawFun, 0, Mapper::NREGIONS, sector, zAxisTitle, filename, minZ, maxZ);
      break;
    }
    case IDCDeltaCompression::HIGH: {
      const auto idcDeltaHigh = this->getIDCDeltaHighCompressed(chunk, sector.side());
      idcFunc = [this, &idcDeltaHigh, chunk, localintegrationInterval](const unsigned int sector, const unsigned int region, const unsigned int irow, const unsigned int pad) {
        return idcDeltaHigh.getValue(this->getIndexUngrouped(sector, region, irow, pad, localintegrationInterval));
      };
      drawFun.mIDCFunc = idcFunc;
      type ? IDCDrawHelper::drawSide(drawFun, sector.side(), zAxisTitle, filename, minZ, maxZ) : IDCDrawHelper::drawSector(drawFun, 0, Mapper::NREGIONS, sector, zAxisTitle, filename, minZ, maxZ);
      break;
    }
  }
}

void o2::tpc::IDCFactorization::drawIDCZeroHelper(const bool type, const Sector sector, const std::string filename, const float minZ, const float maxZ) const
{
  std::function<float(const unsigned int, const unsigned int, const unsigned int, const unsigned int)> idcFunc = [this](const unsigned int sector, const unsigned int region, const unsigned int irow, const unsigned int pad) {
    return this->getIDCZeroVal(sector, region, irow, pad);
  };

  IDCDrawHelper::IDCDraw drawFun;
  drawFun.mIDCFunc = idcFunc;
  const std::string zAxisTitle = IDCDrawHelper::getZAxisTitle(IDCType::IDCZero);
  type ? IDCDrawHelper::drawSide(drawFun, sector.side(), zAxisTitle, filename, minZ, maxZ) : IDCDrawHelper::drawSector(drawFun, 0, Mapper::NREGIONS, sector, zAxisTitle, filename, minZ, maxZ);
}

void o2::tpc::IDCFactorization::drawIDCHelper(const bool type, const Sector sector, const unsigned int integrationInterval, const std::string filename, const float minZ, const float maxZ) const
{
  std::function<float(const unsigned int, const unsigned int, const unsigned int, const unsigned int)> idcFunc = [this, integrationInterval](const unsigned int sector, const unsigned int region, const unsigned int irow, const unsigned int pad) {
    return this->getIDCValUngrouped(sector, region, irow, pad, integrationInterval);
  };

  IDCDrawHelper::IDCDraw drawFun;
  drawFun.mIDCFunc = idcFunc;

  const std::string zAxisTitleDraw = IDCDrawHelper::getZAxisTitle(IDCType::IDC);
  type ? IDCDrawHelper::drawSide(drawFun, sector.side(), zAxisTitleDraw, filename, minZ, maxZ) : IDCDrawHelper::drawSector(drawFun, 0, Mapper::NREGIONS, sector, zAxisTitleDraw, filename, minZ, maxZ);
}

void o2::tpc::IDCFactorization::setGainMap(const char* inpFile, const char* mapName)
{
  TFile f(inpFile, "READ");
  o2::tpc::CalDet<float>* gainMap = nullptr;
  f.GetObject(mapName, gainMap);

  if (!gainMap) {
    LOGP(info, "GainMap {} not found returning", mapName);
    return;
  }
  setGainMap(*gainMap);
  delete gainMap;
}

void o2::tpc::IDCFactorization::setGainMap(const CalDet<float>& gainmap)
{
  mGainMap = std::make_unique<CalDet<float>>(gainmap);
}

void o2::tpc::IDCFactorization::setPadFlagMap(const char* inpFile, const char* mapName)
{
  TFile f(inpFile, "READ");
  o2::tpc::CalDet<PadFlags>* statusmap = nullptr;
  f.GetObject(mapName, statusmap);

  if (!statusmap) {
    LOGP(info, "Pad flag map {} not found returning", mapName);
    return;
  }
  setPadFlagMap(*statusmap);
  delete statusmap;
}

void o2::tpc::IDCFactorization::setPadFlagMap(const CalDet<PadFlags>& flagmap)
{
  mPadFlagsMap = std::make_unique<CalDet<PadFlags>>(flagmap);
}

void o2::tpc::IDCFactorization::drawPadFlagMap(const bool type, const Sector sector, const std::string filename, const PadFlags flag) const
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

void o2::tpc::IDCFactorization::dumpPadFlagMap(const char* outFile, const char* mapName)
{
  if (!mPadFlagsMap) {
    LOGP(info, "Status map not set returning");
  }
  TFile fOut(outFile, "RECREATE");
  fOut.WriteObject(mPadFlagsMap.get(), mapName);
  fOut.Close();
}

template void o2::tpc::IDCFactorization::calcIDCOne(const o2::pmr::vector<float>&, const int, const int, const unsigned int, const CRU, std::vector<float>&, std::vector<unsigned int>&, const IDCZero*, const CalDet<PadFlags>*, const bool);
