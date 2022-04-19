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
#include "TPCBase/Mapper.h"
#include "CommonUtils/TreeStreamRedirector.h"

template <typename DataT>
unsigned int o2::tpc::IDCCCDBHelper<DataT>::getNIntegrationIntervalsIDCDelta(const o2::tpc::Side side) const
{
  return (mIDCDelta && mHelperSector) ? mIDCDelta->getNIDCs(side) / (mHelperSector->getNIDCsPerSector() * SECTORSPERSIDE) : 0;
}

template <typename DataT>
unsigned int o2::tpc::IDCCCDBHelper<DataT>::getNIntegrationIntervalsIDCOne(const o2::tpc::Side side) const
{
  return mIDCOne ? mIDCOne->getNIDCs(side) : 0;
}

template <typename DataT>
float o2::tpc::IDCCCDBHelper<DataT>::getIDCZeroVal(const unsigned int sector, const unsigned int region, unsigned int urow, unsigned int upad) const
{
  /// if the number of pads of the IDC0 corresponds to the number of pads of one TPC side, then no grouping was applied
  return !mIDCZero ? -1 : mIDCZero->getNIDC0(Sector(sector).side()) == Mapper::getNumberOfPadsPerSide() ? mIDCZero->getValueIDCZero(Sector(sector).side(), getUngroupedIndexGlobal(sector, region, urow, upad, 0))
                                                                                                        : mIDCZero->getValueIDCZero(Sector(sector).side(), mHelperSector->getIndexUngrouped(sector, region, urow, upad, 0));
}

template <typename DataT>
float o2::tpc::IDCCCDBHelper<DataT>::getIDCDeltaVal(const unsigned int sector, const unsigned int region, unsigned int urow, unsigned int upad, unsigned int integrationInterval) const
{
  return (!mIDCDelta || !mHelperSector) ? -1 : mIDCDelta->getValue(Sector(sector).side(), mHelperSector->getIndexUngrouped(sector, region, urow, upad, integrationInterval));
}

template <typename DataT>
float o2::tpc::IDCCCDBHelper<DataT>::getIDCOneVal(const o2::tpc::Side side, const unsigned int integrationInterval) const
{
  return !mIDCOne ? -1 : mIDCOne->getValueIDCOne(side, integrationInterval);
}

template <typename DataT>
float o2::tpc::IDCCCDBHelper<DataT>::getIDCVal(const unsigned int sector, const unsigned int region, unsigned int urow, unsigned int upad, unsigned int integrationInterval) const
{
  return (getIDCDeltaVal(sector, region, urow, upad, integrationInterval) + 1.f) * getIDCZeroVal(sector, region, urow, upad) * getIDCOneVal(Sector(sector).side(), integrationInterval);
}

template <typename DataT>
void o2::tpc::IDCCCDBHelper<DataT>::drawIDCZeroHelper(const bool type, const o2::tpc::Sector sector, const std::string filename) const
{
  std::function<float(const unsigned int, const unsigned int, const unsigned int, const unsigned int)> idcFunc = [this](const unsigned int sector, const unsigned int region, const unsigned int irow, const unsigned int pad) {
    return this->getIDCZeroVal(sector, region, irow, pad);
  };

  IDCDrawHelper::IDCDraw drawFun;
  drawFun.mIDCFunc = idcFunc;
  const std::string zAxisTitle = IDCDrawHelper::getZAxisTitle(IDCType::IDCZero);
  type ? IDCDrawHelper::drawSide(drawFun, sector.side(), zAxisTitle, filename) : IDCDrawHelper::drawSector(drawFun, 0, Mapper::NREGIONS, sector, zAxisTitle, filename);
}

template <typename DataT>
void o2::tpc::IDCCCDBHelper<DataT>::drawIDCDeltaHelper(const bool type, const Sector sector, const unsigned int integrationInterval, const std::string filename) const
{
  std::function<float(const unsigned int, const unsigned int, const unsigned int, const unsigned int)> idcFunc = [this, integrationInterval](const unsigned int sector, const unsigned int region, const unsigned int irow, const unsigned int pad) {
    return this->getIDCDeltaVal(sector, region, irow, pad, integrationInterval);
  };

  IDCDrawHelper::IDCDraw drawFun;
  drawFun.mIDCFunc = idcFunc;
  const std::string zAxisTitle = IDCDrawHelper::getZAxisTitle(IDCType::IDCDelta);
  type ? IDCDrawHelper::drawSide(drawFun, sector.side(), zAxisTitle, filename) : IDCDrawHelper::drawSector(drawFun, 0, Mapper::NREGIONS, sector, zAxisTitle, filename);
}

template <typename DataT>
void o2::tpc::IDCCCDBHelper<DataT>::drawIDCHelper(const bool type, const Sector sector, const unsigned int integrationInterval, const std::string filename) const
{
  std::function<float(const unsigned int, const unsigned int, const unsigned int, const unsigned int)> idcFunc = [this, integrationInterval](const unsigned int sector, const unsigned int region, const unsigned int irow, const unsigned int pad) {
    return this->getIDCVal(sector, region, irow, pad, integrationInterval);
  };

  IDCDrawHelper::IDCDraw drawFun;
  drawFun.mIDCFunc = idcFunc;
  const std::string zAxisTitle = IDCDrawHelper::getZAxisTitle(IDCType::IDC);
  type ? IDCDrawHelper::drawSide(drawFun, sector.side(), zAxisTitle, filename) : IDCDrawHelper::drawSector(drawFun, 0, Mapper::NREGIONS, sector, zAxisTitle, filename);
}

template <typename DataT>
unsigned int o2::tpc::IDCCCDBHelper<DataT>::getUngroupedIndexGlobal(const unsigned int sector, const unsigned int region, unsigned int urow, unsigned int upad, unsigned int integrationInterval) const
{
  return IDCGroupHelperSector::getUngroupedIndexGlobal(sector, region, urow, upad, integrationInterval);
}

template <typename DataT>
void o2::tpc::IDCCCDBHelper<DataT>::dumpToTree(int integrationIntervals, const char* outFileName) const
{
  const Mapper& mapper = Mapper::instance();
  o2::utils::TreeStreamRedirector pcstream(outFileName, "RECREATE");
  pcstream.GetFile()->cd();

  if (integrationIntervals <= 0) {
    integrationIntervals = std::min(getNIntegrationIntervalsIDCDelta(Side::A), getNIntegrationIntervalsIDCDelta(Side::C));
  }

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
    float idcOneA = getIDCOneVal(Side::A, integrationInterval);
    float idcOneC = getIDCOneVal(Side::C, integrationInterval);

    pcstream << "tree"
             << "integrationInterval=" << integrationInterval
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
  }
  pcstream.Close();
}

template class o2::tpc::IDCCCDBHelper<float>;
template class o2::tpc::IDCCCDBHelper<unsigned short>;
template class o2::tpc::IDCCCDBHelper<unsigned char>;
