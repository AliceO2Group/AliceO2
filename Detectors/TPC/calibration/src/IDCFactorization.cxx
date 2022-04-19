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
#include "TPCBase/CalDet.h"
#include <functional>

o2::tpc::IDCFactorization::IDCFactorization(const std::array<unsigned char, Mapper::NREGIONS>& groupPads, const std::array<unsigned char, Mapper::NREGIONS>& groupRows, const std::array<unsigned char, Mapper::NREGIONS>& groupLastRowsThreshold, const std::array<unsigned char, Mapper::NREGIONS>& groupLastPadsThreshold, const unsigned int groupNotnPadsSectorEdges, const unsigned int timeFrames, const unsigned int timeframesDeltaIDC, const std::vector<uint32_t>& crus)
  : IDCGroupHelperSector{groupPads, groupRows, groupLastRowsThreshold, groupLastPadsThreshold, groupNotnPadsSectorEdges}, mTimeFrames{timeFrames}, mTimeFramesDeltaIDC{timeframesDeltaIDC}, mIDCDelta{timeFrames / timeframesDeltaIDC + (timeFrames % timeframesDeltaIDC != 0)}, mCRUs{crus}
{
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
    integrationIntervals = static_cast<int>(getNIntegrationIntervals(mCRUs.front()));
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
  for (unsigned int cruInd = 0; cruInd < mCRUs.size(); ++cruInd) {
    const unsigned int cru = mCRUs[cruInd];
    const o2::tpc::CRU cruTmp(cru);
    const auto side = cruTmp.side();
    const unsigned int region = cruTmp.region();
    const auto factorIndexGlob = mRegionOffs[region] + mNIDCsPerSector * (cruTmp.sector() % o2::tpc::SECTORSPERSIDE);
    for (unsigned int timeframe = 0; timeframe < mTimeFrames; ++timeframe) {
      for (unsigned int idcs = 0; idcs < mIDCs[cru][timeframe].size(); ++idcs) {
        const unsigned int indexGlob = (idcs % mNIDCsPerCRU[region]) + factorIndexGlob;
        if (norm && mIDCs[cru][timeframe][idcs] > 0) {
          mIDCs[cru][timeframe][idcs] *= Mapper::INVPADAREA[region];
        }
        mIDCZero.fillValueIDCZero(mIDCs[cru][timeframe][idcs], side, indexGlob);
      }
    }
  }
  const auto normFac = getNIntegrationIntervals(mCRUs.front());
  if (normFac == 0) {
    LOGP(error, "number of integraion intervals is zero! Skipping normalization of IDC0");
  } else {
    std::transform(mIDCZero.mIDCZero[Side::A].begin(), mIDCZero.mIDCZero[Side::A].end(), mIDCZero.mIDCZero[Side::A].begin(), [normVal = normFac](auto& val) { return val / normVal; });
    std::transform(mIDCZero.mIDCZero[Side::C].begin(), mIDCZero.mIDCZero[Side::C].end(), mIDCZero.mIDCZero[Side::C].begin(), [normVal = normFac](auto& val) { return val / normVal; });
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
        const auto idcZeroVal = mIDCZero.mIDCZero[side][index];
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
        mIDCZero.setValueIDCZero(idcZero, side, index);
      }
      padTmp.clear();
    }
  }
}

void o2::tpc::IDCFactorization::calcIDCOne()
{
  const unsigned int integrationIntervals = getNIntegrationIntervals(mCRUs.front());
  mIDCOne.clear();
  mIDCOne.resize(integrationIntervals);

  std::array<std::vector<std::vector<float>>, SIDES> idcOneSafe;
  std::array<std::vector<std::vector<unsigned int>>, SIDES> weightsSafe;

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
  for (unsigned int cru = 0; cru < mCRUs.size(); ++cru) {
#ifdef WITH_OPENMP
    const int ithread = omp_get_thread_num();
#else
    const int ithread = 0;
#endif
    const o2::tpc::CRU cruTmp(cru);
    const unsigned int region = cruTmp.region();
    const auto side = cruTmp.side();
    const auto factorIndexGlob = mRegionOffs[region] + mNIDCsPerSector * (cruTmp.sector() % o2::tpc::SECTORSPERSIDE);
    unsigned int integrationIntervalOffset = 0;
    for (unsigned int timeframe = 0; timeframe < mTimeFrames; ++timeframe) {
      calcIDCOne(mIDCs[cru][timeframe], mNIDCsPerCRU[region], integrationIntervalOffset, factorIndexGlob, cru, idcOneSafe[side][ithread], weightsSafe[side][ithread], &mIDCZero, mInputGrouped ? nullptr : mPadFlagsMap.get());
      integrationIntervalOffset += mIDCs[cru][timeframe].size() / mNIDCsPerCRU[region];
    }
  }

#pragma omp parallel for num_threads(sNThreads)
  for (int side = 0; side < SIDES; ++side) {
    for (int i = 1; i < sNThreads; ++i) {
      std::transform(idcOneSafe[side].front().begin(), idcOneSafe[side].front().end(), idcOneSafe[side][i].begin(), idcOneSafe[side].front().begin(), std::plus<float>());
      std::transform(weightsSafe[side].front().begin(), weightsSafe[side].front().end(), weightsSafe[side][i].begin(), weightsSafe[side].front().begin(), std::plus<unsigned int>());
    }
    // replace all 0 with 1 to avoid division by 0
    std::replace(weightsSafe[side].front().begin(), weightsSafe[side].front().end(), 0, 1);

    // move IDC1 to member
    mIDCOne.mIDCOne[side] = std::move(idcOneSafe[side].front());

    // normalize IDC1 to number of IDC values used
    std::transform(mIDCOne.mIDCOne[side].begin(), mIDCOne.mIDCOne[side].end(), weightsSafe[side].front().begin(), mIDCOne.mIDCOne[side].begin(), std::divides<float>());
  }
}

void o2::tpc::IDCFactorization::calcIDCOne(const std::vector<float>& idcsData, const int idcsPerCRU, const int integrationIntervalOffset, const unsigned int indexOffset, const CRU cru, std::vector<float>& idcOneTmp, std::vector<unsigned int>& weights, const IDCZero* idcZero, const CalDet<PadFlags>* flagMap)
{
  const Side side = cru.side();
  for (unsigned int idcs = 0; idcs < idcsData.size(); ++idcs) {
    const unsigned int integrationInterval = idcs / idcsPerCRU + integrationIntervalOffset;
    const unsigned int localPad = (idcs % idcsPerCRU);
    const unsigned int indexGlob = localPad + indexOffset;
    const auto idcZeroVal = idcZero ? idcZero->mIDCZero[side][indexGlob] : 1;

    // check pad in case of input is not grouped
    if (flagMap) {
      const o2::tpc::PadFlags flag = flagMap->getCalArray(cru).getValue(localPad);
      if ((flag & PadFlags::flagSkip) == PadFlags::flagSkip) {
        continue;
      }
    }

    if (idcZeroVal > 0 && idcsData[idcs] >= 0) {
      idcOneTmp[integrationInterval] += idcsData[idcs] / idcZeroVal;
      ++weights[integrationInterval];
    }
  }
}

void o2::tpc::IDCFactorization::calcIDCDelta()
{
  const unsigned int nIDCsSide = mNIDCsPerSector * SECTORSPERSIDE;
  for (unsigned int i = 0; i < getNChunks(); ++i) {
    const auto idcsSide = nIDCsSide * getNIntegrationIntervals(i, mCRUs.front());
    mIDCDelta[i].getIDCDelta(Side::A).resize(idcsSide);
    mIDCDelta[i].getIDCDelta(Side::C).resize(idcsSide);
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
        const unsigned int intervallocal = idcs / mNIDCsPerCRU[region];
        const unsigned int integrationIntervalGlobal = intervallocal + integrationIntervallast;
        const unsigned int integrationIntervalLocal = intervallocal + integrationIntervallastLocal;
        const unsigned int localPad = idcs % mNIDCsPerCRU[region];
        const unsigned int indexGlob = localPad + factorIndexGlob;
        const auto idcZero = mIDCZero.mIDCZero[side][indexGlob];
        const auto idcOne = mIDCOne.mIDCOne[side][integrationIntervalGlobal];
        const auto mult = idcZero * idcOne;
        auto val = (mult > 0 && mIDCs[cru][timeframe][idcs] > 0) ? mIDCs[cru][timeframe][idcs] / mult - 1 : 0;
        if (!mInputGrouped && mPadFlagsMap) {
          const o2::tpc::PadFlags flag = mPadFlagsMap->getCalArray(cru).getValue(localPad);
          if ((flag & PadFlags::flagSkip) == PadFlags::flagSkip) {
            val = 0;
          }
        }
        mIDCDelta[chunk].setIDCDelta(side, indexGlob + integrationIntervalLocal * nIDCsSide, val);
      }

      const unsigned int intervals = mIDCs[cru][timeframe].size() / mNIDCsPerCRU[region];
      integrationIntervallast += intervals;
      integrationIntervallastLocal += intervals;
      lastChunk = chunk;
    }
  }
}

float o2::tpc::IDCFactorization::getMedian(std::vector<float>& values)
{
  if (values.empty()) {
    return 0;
  }
  size_t n = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + n, values.end());
  return values[n];
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
    std::vector<float> idcsRow;
    const auto maxValues = Mapper::PADSPERROW[region].back();
    idcsRow.reserve(maxValues);
    for (unsigned int lrow = 0; lrow < Mapper::ROWSPERREGION[region]; ++lrow) {

      // loop over pads in row in the first iteration and calculate median at the end
      // in the second iteration check if the IDC value is to far away from the median
      for (int iter = 0; iter < 2; ++iter) {
        const unsigned int integrationInterval = 0;
        const float median = (iter == 1) ? getMedian(idcsRow) : 0;
        // loop over pad row and calculate mean of IDC0
        for (unsigned int pad = 0; pad < Mapper::PADSPERROW[region][lrow]; ++pad) {
          const auto index = getIndexUngrouped(sector, region, lrow, pad, integrationInterval);
          const auto idcZeroVal = mIDCZero.mIDCZero[side][index];

          if (iter == 0) {
            // exclude dead pads
            if (idcZeroVal > 0) {
              idcsRow.emplace_back(idcZeroVal);
            }
          } else {
            if (idcZeroVal <= 0) {
              const unsigned int padInRegion = Mapper::OFFSETCRULOCAL[region][lrow] + pad;
              const o2::tpc::PadFlags flag = o2::tpc::PadFlags::flagDeadPad | o2::tpc::PadFlags::flagSkip | mPadFlagsMap->getCalArray(cru).getValue(padInRegion);
              mPadFlagsMap->getCalArray(cru).setValue(padInRegion, flag);
            } else if (idcZeroVal > paramIDCGroup.maxIDC0Median * median) {
              const unsigned int padInRegion = Mapper::OFFSETCRULOCAL[region][lrow] + pad;
              const o2::tpc::PadFlags flag = o2::tpc::PadFlags::flagHighPad | o2::tpc::PadFlags::flagSkip | mPadFlagsMap->getCalArray(cru).getValue(padInRegion);
              mPadFlagsMap->getCalArray(cru).setValue(padInRegion, flag);
            } else if (idcZeroVal < paramIDCGroup.minIDC0Median * median) {
              const unsigned int padInRegion = Mapper::OFFSETCRULOCAL[region][lrow] + pad;
              const o2::tpc::PadFlags flag = o2::tpc::PadFlags::flagLowPad | o2::tpc::PadFlags::flagSkip | mPadFlagsMap->getCalArray(cru).getValue(padInRegion);
              mPadFlagsMap->getCalArray(cru).setValue(padInRegion, flag);
            }
          }
        } // loop over pads in row
      }   // end iteration
      idcsRow.clear();
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
  LOGP(info, "Calculating IDC0");
  calcIDCZero(norm);
  if (!mInputGrouped) {
    LOGP(info, "Creating pad status map");
    createStatusMap();
  }
  LOGP(info, "Calculating IDC1");
  calcIDCOne();
  LOGP(info, "Calculating IDCDelta");
  calcIDCDelta();
}

unsigned long o2::tpc::IDCFactorization::getNIntegrationIntervals(const unsigned int chunk, const int cru) const
{
  std::size_t sum = 0;
  const auto firstTF = chunk * getNTFsPerChunk(0);
  for (unsigned int i = firstTF; i < firstTF + getNTFsPerChunk(chunk); ++i) {
    sum += mIDCs[cru][i].size();
  }
  return sum / mNIDCsPerCRU[CRU(cru).region()];
}

unsigned long o2::tpc::IDCFactorization::getNIntegrationIntervals(const int cru) const
{
  std::size_t sum = 0;
  for (auto&& idcsTF : mIDCs[cru]) {
    sum += idcsTF.size();
  }
  return sum / mNIDCsPerCRU[CRU(cru).region()];
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

void o2::tpc::IDCFactorization::drawIDCDeltaHelper(const bool type, const Sector sector, const unsigned int integrationInterval, const IDCDeltaCompression compression, const std::string filename, const float minZ, const float maxZ) const
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
      type ? IDCDrawHelper::drawSide(drawFun, sector.side(), zAxisTitle, filename, minZ, maxZ) : IDCDrawHelper::drawSector(drawFun, 0, Mapper::NREGIONS, sector, zAxisTitle, filename, minZ, maxZ);
      break;
    }
    case IDCDeltaCompression::MEDIUM: {
      const auto idcDeltaMedium = this->getIDCDeltaMediumCompressed(chunk);
      idcFunc = [this, &idcDeltaMedium, chunk, localintegrationInterval = localintegrationInterval](const unsigned int sector, const unsigned int region, const unsigned int irow, const unsigned int pad) {
        return idcDeltaMedium.getValue(Sector(sector).side(), this->getIndexUngrouped(sector, region, irow, pad, localintegrationInterval));
      };
      drawFun.mIDCFunc = idcFunc;
      type ? IDCDrawHelper::drawSide(drawFun, sector.side(), zAxisTitle, filename, minZ, maxZ) : IDCDrawHelper::drawSector(drawFun, 0, Mapper::NREGIONS, sector, zAxisTitle, filename, minZ, maxZ);
      break;
    }
    case IDCDeltaCompression::HIGH: {
      const auto idcDeltaHigh = this->getIDCDeltaHighCompressed(chunk);
      idcFunc = [this, &idcDeltaHigh, chunk, localintegrationInterval](const unsigned int sector, const unsigned int region, const unsigned int irow, const unsigned int pad) {
        return idcDeltaHigh.getValue(Sector(sector).side(), this->getIndexUngrouped(sector, region, irow, pad, localintegrationInterval));
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
