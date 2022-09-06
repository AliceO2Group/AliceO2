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

#include "TPCCalibration/SACCCDBHelper.h"
#include "TPCCalibration/SACDrawHelper.h"
#include "TPCCalibration/IDCContainer.h"
#include "TPCCalibration/SACFactorization.h"
#include "CommonUtils/TreeStreamRedirector.h"

template <typename DataT>
unsigned int o2::tpc::SACCCDBHelper<DataT>::getNIntegrationIntervalsSACDelta(const o2::tpc::Side side) const
{
  return mSACDelta ? (mSACDelta->mSACDelta[side].getNIDCs()) / (GEMSTACKSPERSECTOR * SECTORSPERSIDE) : 0;
}

template <typename DataT>
unsigned int o2::tpc::SACCCDBHelper<DataT>::getNIntegrationIntervalsSACOne(const o2::tpc::Side side) const
{
  return mSACOne ? mSACOne->mSACOne[side].getNIDCs() : 0;
}

template <typename DataT>
float o2::tpc::SACCCDBHelper<DataT>::getSACZeroVal(const unsigned int sector, const unsigned int stack) const
{
  return !mSACZero ? -1 : mSACZero->getValueIDCZero(Sector(sector).side(), SACFactorization::getStackInSide(sector, stack));
}

template <typename DataT>
float o2::tpc::SACCCDBHelper<DataT>::getSACDeltaVal(const unsigned int sector, const unsigned int stack, unsigned int integrationInterval) const
{
  return !mSACDelta ? -1 : mSACDelta->getValue(Sector(sector).side(), SACFactorization::getSACDeltaIndex(SACFactorization::getStackInSide(sector, stack), integrationInterval));
}

template <typename DataT>
float o2::tpc::SACCCDBHelper<DataT>::getSACOneVal(const o2::tpc::Side side, const unsigned int integrationInterval) const
{
  return !mSACOne ? -1 : mSACOne->getValue(side, integrationInterval);
}

template <typename DataT>
float o2::tpc::SACCCDBHelper<DataT>::getSACVal(const unsigned int sector, const unsigned int stack, unsigned int integrationInterval) const
{
  return (getSACDeltaVal(sector, stack, integrationInterval) + 1.f) * getSACZeroVal(sector, stack) * getSACOneVal(Sector(sector).side(), integrationInterval);
}

template <typename DataT>
void o2::tpc::SACCCDBHelper<DataT>::drawSACZeroHelper(const bool type, const o2::tpc::Sector sector, const std::string filename, const float minZ, const float maxZ) const
{
  std::function<float(const unsigned int, const unsigned int)> SACFunc = [this](const unsigned int sector, const unsigned int stack) {
    return this->getSACZeroVal(sector, stack);
  };

  SACDrawHelper::SACDraw drawFun;
  drawFun.mSACFunc = SACFunc;
  const std::string zAxisTitle = SACDrawHelper::getZAxisTitle(SACType::IDCZero);
  type ? SACDrawHelper::drawSide(drawFun, sector.side(), zAxisTitle, filename, minZ, maxZ) : SACDrawHelper::drawSector(drawFun, sector, zAxisTitle, filename, minZ, maxZ);
}

template <typename DataT>
void o2::tpc::SACCCDBHelper<DataT>::drawSACDeltaHelper(const bool type, const Sector sector, const unsigned int integrationInterval, const std::string filename, const float minZ, const float maxZ) const
{
  std::function<float(const unsigned int, const unsigned int)> SACFunc = [this, integrationInterval](const unsigned int sector, const unsigned int stack) {
    return this->getSACDeltaVal(sector, stack, integrationInterval);
  };

  SACDrawHelper::SACDraw drawFun;
  drawFun.mSACFunc = SACFunc;
  const std::string zAxisTitle = SACDrawHelper::getZAxisTitle(SACType::IDCDelta);
  type ? SACDrawHelper::drawSide(drawFun, sector.side(), zAxisTitle, filename, minZ, maxZ) : SACDrawHelper::drawSector(drawFun, sector, zAxisTitle, filename, minZ, maxZ);
}

template <typename DataT>
void o2::tpc::SACCCDBHelper<DataT>::drawSACHelper(const bool type, const Sector sector, const unsigned int integrationInterval, const std::string filename, const float minZ, const float maxZ) const
{
  std::function<float(const unsigned int, const unsigned int)> SACFunc = [this, integrationInterval](const unsigned int sector, const unsigned int stack) {
    return this->getSACVal(sector, stack, integrationInterval);
  };

  SACDrawHelper::SACDraw drawFun;
  drawFun.mSACFunc = SACFunc;
  const std::string zAxisTitle = SACDrawHelper::getZAxisTitle(SACType::IDC);
  type ? SACDrawHelper::drawSide(drawFun, sector.side(), zAxisTitle, filename, minZ, maxZ) : SACDrawHelper::drawSector(drawFun, sector, zAxisTitle, filename, minZ, maxZ);
}

template <typename DataT>
void o2::tpc::SACCCDBHelper<DataT>::dumpToTree(const char* outFileName) const
{
  o2::utils::TreeStreamRedirector pcstream(outFileName, "RECREATE");
  pcstream.GetFile()->cd();

  const auto intervals = std::min(getNIntegrationIntervalsSACDelta(Side::A), getNIntegrationIntervalsSACDelta(Side::C));
  for (unsigned int integrationInterval = 0; integrationInterval < intervals; ++integrationInterval) {
    std::vector<int32_t> vSACs(GEMSTACKS);
    std::vector<float> vSACsZero(GEMSTACKS);
    std::vector<DataT> vSACsDelta(GEMSTACKS);
    std::vector<unsigned int> vStack(GEMSTACKS);

    unsigned int index = 0;
    for (unsigned int sector = 0; sector < Mapper::NSECTORS; ++sector) {
      for (unsigned int stack = 0; stack < GEMSTACKSPERSECTOR; ++stack) {
        vSACs[index] = getSACVal(sector, stack, integrationInterval);
        vSACsZero[index] = getSACZeroVal(sector, stack);
        vSACsDelta[index] = getSACDeltaVal(sector, stack, integrationInterval);
        vStack[index++] = SACFactorization::getStack(sector, stack);
      }
    }
    float sacOneA = getSACOneVal(Side::A, integrationInterval);
    float sacOneC = getSACOneVal(Side::C, integrationInterval);

    pcstream << "tree"
             << "integrationInterval=" << integrationInterval
             << "SAC.=" << vSACs
             << "SAC0.=" << vSACsZero
             << "SAC1A=" << sacOneA
             << "SAC1C=" << sacOneC
             << "SACDelta.=" << vSACsDelta
             << "stack.=" << vStack
             << "\n";
  }
  pcstream.Close();
}

template <typename DataT>
void o2::tpc::SACCCDBHelper<DataT>::dumpToFourierCoeffToTree(const char* outFileName) const
{
  o2::utils::TreeStreamRedirector pcstream("fourierCoeff.root", "RECREATE");
  pcstream.GetFile()->cd();

  for (int iside = 0; iside < SIDES; ++iside) {
    const Side side = (iside == 0) ? Side::A : Side::C;

    if (!mFourierCoeff) {
      continue;
    }

    const int nTFs = mFourierCoeff->mCoeff[side].getNCoefficients() / mFourierCoeff->mCoeff[side].getNCoefficientsPerTF();
    for (int iTF = 0; iTF < nTFs; ++iTF) {
      std::vector<float> coeff;
      std::vector<int> ind;
      int coeffPerTF = mFourierCoeff->mCoeff[side].getNCoefficientsPerTF();
      for (int i = 0; i < coeffPerTF; ++i) {
        const int index = mFourierCoeff->mCoeff[side].getIndex(iTF, i);
        coeff.emplace_back((mFourierCoeff->mCoeff[side])(index));
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

template class o2::tpc::SACCCDBHelper<float>;
template class o2::tpc::SACCCDBHelper<unsigned short>;
template class o2::tpc::SACCCDBHelper<unsigned char>;
