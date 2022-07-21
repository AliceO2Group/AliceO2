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

#include "TPCCalibration/SACFactorization.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "TPCCalibration/SACDrawHelper.h"
#include "TPCCalibration/RobustAverage.h"
#include "TPCCalibration/SACParameter.h"
#include "Framework/Logger.h"
#include "TFile.h"
#include <functional>
#include <numeric>

#if (defined(WITH_OPENMP) || defined(_OPENMP)) && !defined(__CLING__)
#include <omp.h>
#endif

void o2::tpc::SACFactorization::dumpToFile(const char* outFileName, const char* outName) const
{
  TFile fOut(outFileName, "RECREATE");
  fOut.WriteObject(this, outName);
  fOut.Close();
}

void o2::tpc::SACFactorization::dumpToTree(int integrationIntervals, const char* outFileName) const
{
  o2::utils::TreeStreamRedirector pcstream(outFileName, "RECREATE");
  pcstream.GetFile()->cd();

  if (integrationIntervals <= 0) {
    integrationIntervals = getNIntegrationIntervals();
  }

  const auto SACDeltaMedium = getSACDeltaMediumCompressed();
  const auto SACDeltaHigh = getSACDeltaHighCompressed();
  for (unsigned int integrationInterval = 0; integrationInterval < integrationIntervals; ++integrationInterval) {
    std::vector<int32_t> vSACs(GEMSTACKS);
    std::vector<float> vSACsZero(GEMSTACKS);
    std::vector<float> vSACsDelta(GEMSTACKS);
    std::vector<float> vSACsDeltaMedium(GEMSTACKS);
    std::vector<float> vSACsDeltaHigh(GEMSTACKS);
    std::vector<unsigned int> vStack(GEMSTACKS);
    std::vector<int> vSACZeroCut(GEMSTACKS);

    unsigned int index = 0;
    for (unsigned int stack = 0; stack < GEMSTACKS; ++stack) {
      vSACs[index] = getSACValue(stack, integrationInterval);
      vSACsZero[index] = getSACZeroVal(stack);
      vSACsDelta[index] = getSACDeltaVal(stack, integrationInterval);
      vSACsDeltaMedium[index] = SACDeltaMedium.getValue(getSide(stack), getSACDeltaIndex(stack, integrationInterval));
      vSACsDeltaHigh[index] = SACDeltaHigh.getValue(getSide(stack), getSACDeltaIndex(stack, integrationInterval));
      vSACZeroCut[index] = mOutlierMap[stack];
      vStack[index++] = stack;
    }
    float sacOneA = getSACOneVal(Side::A, integrationInterval);
    float sacOneC = getSACOneVal(Side::C, integrationInterval);

    pcstream << "tree"
             << "integrationInterval=" << integrationInterval
             << "SAC.=" << vSACs
             << "SAC0.=" << vSACsZero
             << "SAC1A=" << sacOneA
             << "SAC1C=" << sacOneC
             << "SACDeltaNoComp.=" << vSACsDelta
             << "SACDeltaMediumComp.=" << vSACsDeltaMedium
             << "SACDeltaHighComp.=" << vSACsDeltaHigh
             << "stack.=" << vStack
             << "SACZeroOutlier.=" << vSACZeroCut
             << "\n";
  }
  pcstream.Close();
}

void o2::tpc::SACFactorization::calcSACZero()
{
  const unsigned int nSACsSide = GEMSTACKSPERSIDE;
  mSACZero.clear();
  mSACZero.resize(nSACsSide);

#pragma omp parallel for num_threads(sNThreads)
  for (unsigned int stack = 0; stack < GEMSTACKS; ++stack) {
    const auto side = getSide(stack);
    const auto sacZero = std::accumulate(mSACs[stack].begin(), mSACs[stack].end(), 0.f) / mSACs[stack].size();
    mSACZero.setValueSACZero(sacZero, side, stack % GEMSTACKSPERSIDE);
  }
}

void o2::tpc::SACFactorization::calcSACOne()
{
  const unsigned int integrationIntervals = getNIntegrationIntervals();
  mSACOne.clear();
  mSACOne.resize(integrationIntervals);

#pragma omp parallel for num_threads(sNThreads)
  for (int iSide = 0; iSide < SIDES; ++iSide) {

    const int firstStack = (iSide == 0) ? 0 : GEMSTACKSPERSIDE;
    const int lastStack = (iSide == 0) ? GEMSTACKSPERSIDE : GEMSTACKS;
    const auto normFac = GEMSTACKSPERSIDE - std::accumulate(mOutlierMap.begin() + firstStack, mOutlierMap.begin() + lastStack, 0);
    const Side side = (iSide == 0) ? Side::A : Side::C;
    for (unsigned int interval = 0; interval < integrationIntervals; ++interval) {

      float sacOne = 0;
      for (unsigned int stackLocal = 0; stackLocal < GEMSTACKSPERSIDE; ++stackLocal) {
        const int stack = stackLocal + iSide * GEMSTACKSPERSIDE;
        const float sacZero = mSACZero.getValueIDCZero(side, stackLocal);
        const float sacValue = mSACs[stack][interval];
        if (mOutlierMap[stack] == 0) {
          sacOne += (sacZero == 0) ? sacZero : sacValue / sacZero;
        }
      }

      mSACOne.setValueIDCOne(sacOne / normFac, side, interval);
    }
  }
}

void o2::tpc::SACFactorization::calcSACDelta()
{
  const unsigned int integrationIntervals = getNIntegrationIntervals();
  const unsigned int nSACs = integrationIntervals * GEMSTACKSPERSIDE;
  mSACDelta.resize(nSACs);

#pragma omp parallel for num_threads(sNThreads)
  for (unsigned int stack = 0; stack < GEMSTACKS; ++stack) {
    const Side side = getSide(stack);
    for (unsigned int interval = 0; interval < integrationIntervals; ++interval) {
      const auto SACZero = getSACZeroVal(stack);
      const auto SACOne = mSACOne.getValue(side, interval);
      const auto sacVal = mSACs[stack][interval];
      const auto mult = SACZero * SACOne;
      const auto val = (mult != 0 && (mOutlierMap[stack] == 0)) ? sacVal / mult - 1 : 0;
      mSACDelta.setSACDelta(side, getSACDeltaIndex(stack, interval), val);
    }
  }
}

void o2::tpc::SACFactorization::factorizeSACs()
{
  LOGP(info, "Using {} threads for factorization of SACs", sNThreads);
  LOGP(info, "Calculating SAC0");
  calcSACZero();
  LOGP(info, "Creating pad status map");
  createStatusMap();
  LOGP(info, "Calculating SAC1");
  calcSACOne();
  LOGP(info, "Calculating SACDelta");
  calcSACDelta();
}

void o2::tpc::SACFactorization::reset()
{
  for (auto& val : mSACs) {
    val.clear();
  }
}

void o2::tpc::SACFactorization::drawSACDeltaHelper(const bool type, const Sector sector, const unsigned int integrationInterval, const SACDeltaCompression compression, const std::string filename, const float minZ, const float maxZ) const
{
  std::function<float(const unsigned int, const unsigned int)> SACFunc;

  const std::string zAxisTitle = SACDrawHelper::getZAxisTitle(SACType::IDCDelta);

  SACDrawHelper::SACDraw drawFun;
  switch (compression) {
    case SACDeltaCompression::NO:
    default: {
      SACFunc = [this, integrationInterval](const unsigned int sector, const unsigned int stack) {
        return this->getSACDeltaVal(getStack(sector, stack), integrationInterval);
      };
      drawFun.mSACFunc = SACFunc;
      type ? SACDrawHelper::drawSide(drawFun, sector.side(), zAxisTitle, filename, minZ, maxZ) : SACDrawHelper::drawSector(drawFun, sector, zAxisTitle, filename, minZ, maxZ);
      break;
    }
    case SACDeltaCompression::MEDIUM: {
      const auto SACDeltaMedium = this->getSACDeltaMediumCompressed();
      SACFunc = [this, &SACDeltaMedium, integrationInterval = integrationInterval](const unsigned int sector, const unsigned int stack) {
        return SACDeltaMedium.getValue(Sector(sector).side(), getSACDeltaIndex(getStack(sector, stack), integrationInterval));
      };
      drawFun.mSACFunc = SACFunc;
      type ? SACDrawHelper::drawSide(drawFun, sector.side(), zAxisTitle, filename, minZ, maxZ) : SACDrawHelper::drawSector(drawFun, sector, zAxisTitle, filename, minZ, maxZ);
      break;
    }
    case SACDeltaCompression::HIGH: {
      const auto SACDeltaHigh = this->getSACDeltaHighCompressed();
      SACFunc = [this, &SACDeltaHigh, integrationInterval](const unsigned int sector, const unsigned int stack) {
        return SACDeltaHigh.getValue(Sector(sector).side(), getSACDeltaIndex(getStack(sector, stack), integrationInterval));
      };
      drawFun.mSACFunc = SACFunc;
      type ? SACDrawHelper::drawSide(drawFun, sector.side(), zAxisTitle, filename, minZ, maxZ) : SACDrawHelper::drawSector(drawFun, sector, zAxisTitle, filename, minZ, maxZ);
      break;
    }
  }
}

void o2::tpc::SACFactorization::drawSACZeroHelper(const bool type, const Sector sector, const std::string filename, const float minZ, const float maxZ) const
{
  std::function<float(const unsigned int, const unsigned int)> SACFunc = [this](const unsigned int sector, const unsigned int stack) {
    return this->getSACZeroVal(getStack(sector, stack));
  };

  SACDrawHelper::SACDraw drawFun;
  drawFun.mSACFunc = SACFunc;
  const std::string zAxisTitle = SACDrawHelper::getZAxisTitle(SACType::IDCZero);
  type ? SACDrawHelper::drawSide(drawFun, sector.side(), zAxisTitle, filename, minZ, maxZ) : SACDrawHelper::drawSector(drawFun, sector, zAxisTitle, filename, minZ, maxZ);
}

void o2::tpc::SACFactorization::drawSACZeroOutlierHelper(const bool type, const Sector sector, const std::string filename, const float minZ, const float maxZ) const
{
  std::function<float(const unsigned int, const unsigned int)> SACFunc = [this](const unsigned int sector, const unsigned int stack) {
    return this->mOutlierMap[getStack(sector, stack)];
  };

  SACDrawHelper::SACDraw drawFun;
  drawFun.mSACFunc = SACFunc;
  const std::string zAxisTitle = SACDrawHelper::getZAxisTitle(SACType::IDCZero) + " outlier";
  type ? SACDrawHelper::drawSide(drawFun, sector.side(), zAxisTitle, filename, minZ, maxZ) : SACDrawHelper::drawSector(drawFun, sector, zAxisTitle, filename, minZ, maxZ);
}

void o2::tpc::SACFactorization::drawSACHelper(const bool type, const Sector sector, const unsigned int integrationInterval, const std::string filename, const float minZ, const float maxZ) const
{
  std::function<float(const unsigned int, const unsigned int)> SACFunc = [this, integrationInterval](const unsigned int sector, const unsigned int stack) {
    return this->getSACValue(getStack(sector, stack), integrationInterval);
  };

  SACDrawHelper::SACDraw drawFun;
  drawFun.mSACFunc = SACFunc;

  const std::string zAxisTitle = SACDrawHelper::getZAxisTitle(SACType::IDC);
  type ? SACDrawHelper::drawSide(drawFun, sector.side(), zAxisTitle, filename, minZ, maxZ) : SACDrawHelper::drawSector(drawFun, sector, zAxisTitle, filename, minZ, maxZ);
}

void o2::tpc::SACFactorization::createStatusMap()
{
  const auto& paramSAC = ParameterSAC::Instance();
#pragma omp parallel for num_threads(sNThreads)
  for (unsigned int stackInSector = 0; stackInSector < GEMSTACKSPERSECTOR; ++stackInSector) {
    o2::tpc::RobustAverage average(SECTORSPERSIDE);
    for (int iSide = 0; iSide < SIDES; ++iSide) {
      const Side side = (iSide == 0) ? Side::A : Side::C;
      for (int iter = 0; iter < 2; ++iter) {
        const float median = (iter == 1) ? average.getMedian() : 0;
        const float stdDev = (iter == 1) ? average.getStdDev() : 0;
        for (int sector = 0; sector < SECTORSPERSIDE; ++sector) {
          const int stackLocal = sector * GEMSTACKSPERSECTOR + stackInSector;
          const float sacZero = mSACZero.getValueIDCZero(side, stackLocal);
          if (iter == 0) {
            average.addValue(sacZero);
          } else {
            // define outlier
            const int stack = stackLocal + iSide * GEMSTACKSPERSIDE;
            if ((sacZero > median + stdDev * paramSAC.maxSAC0Median) || (sacZero < median - stdDev * paramSAC.minSAC0Median)) {
              mOutlierMap[stack] = 1;
            } else {
              mOutlierMap[stack] = 0;
            }
          }
        }
      }
      average.clear();
    }
  }
}
