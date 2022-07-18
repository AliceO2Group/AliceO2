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

#ifndef AliceO2_TPC_Mapper_H
#define AliceO2_TPC_Mapper_H

#include <map>
#include <vector>
#include <array>
#include <string>
#include <cmath>

#include "DataFormatsTPC/Defs.h"
#include "TPCBase/PadPos.h"
#include "TPCBase/PadSecPos.h"
#include "TPCBase/PadROCPos.h"
#include "TPCBase/DigitPos.h"
#include "TPCBase/FECInfo.h"
#include "TPCBase/PadRegionInfo.h"
#include "TPCBase/PartitionInfo.h"
#include "TPCBase/Sector.h"

#include "MathUtils/Cartesian.h"

// using o2::tpc::PadRegionInfo;
// using o2::tpc::PartitionInfo;

namespace o2
{
namespace tpc
{

class Mapper
{
 public:
  static Mapper& instance(const std::string mappingDir = "")
  {
    static Mapper mapper(mappingDir);
    return mapper;
  }

  const PadPos& padPos(GlobalPadNumber padNumber) const { return mMapGlobalPadToPadPos[padNumber % mPadsInSector]; }
  const PadCentre& padCentre(GlobalPadNumber padNumber) const { return mMapGlobalPadCentre[padNumber % mPadsInSector]; }
  const FECInfo& fecInfo(GlobalPadNumber padNumber) const { return mMapGlobalPadFECInfo[padNumber % mPadsInSector]; }

  // const GlobalPadNumber globalPadNumber(const PadPos& padPosition) const { return
  // mMapPadPosGlobalPad.find(padPosition)->second; }
  GlobalPadNumber globalPadNumber(const PadPos& globalPadPosition) const
  {
    return mMapPadOffsetPerRow[globalPadPosition.getRow()] + globalPadPosition.getPad();
  }

  /// \return returns the global pad number for given local pad row and pad
  /// \param lrow ungrouped local row in a region
  /// \param pad ungrouped pad in row
  GlobalPadNumber static getGlobalPadNumber(const unsigned int lrow, const unsigned int pad, const unsigned int region) { return GLOBALPADOFFSET[region] + OFFSETCRULOCAL[region][lrow] + pad; }

  /// \param row global pad row
  /// \param pad pad in row
  /// \return returns local pad number in region
  static unsigned int getLocalPadNumber(const unsigned int row, const unsigned int pad) { return OFFSETCRUGLOBAL[row] + pad; }

  /// \param row global pad row
  static unsigned int getLocalRowFromGlobalRow(const unsigned int row) { return row - ROWOFFSET[REGION[row]]; }

  /// return the cru number from sector and global pad number
  /// \param sec sector
  /// \param globalPad global pad number in sector
  /// \return global cru number
  int getCRU(const Sector& sec, GlobalPadNumber globalPad) const
  {
    const auto row = mMapGlobalPadToPadPos[globalPad].getRow();
    const auto nCRUPerSector = mMapPadRegionInfo.size();
    int region = 0;
    for (size_t i = 1; i < nCRUPerSector; ++i) {
      if (row < mMapPadRegionInfo[i].getGlobalRowOffset()) {
        break;
      }
      ++region;
    }

    return int(sec * nCRUPerSector + region);
  }

  /// return the global pad number in ROC for PadROCPos (ROC, row, pad)
  /// \return global pad number of PadROCPos (ROC, row, pad)
  /// \todo add check for row and pad limits
  GlobalPadNumber getPadNumberInROC(const PadROCPos& rocPadPosition) const
  {
    const size_t padOffset = (rocPadPosition.getROCType() == RocType::IROC) ? 0 : mPadsInIROC;
    const size_t rowOffset = (rocPadPosition.getROCType() == RocType::IROC) ? 0 : mNumberOfPadRowsIROC;

    return mMapPadOffsetPerRow[rocPadPosition.getRow() + rowOffset] + rocPadPosition.getPad() - padOffset;
  }

  /// return the global pad number in a partition for partition, row, pad
  /// \return global pad number in a partition for partition, row, pad
  /// \todo add check for row and pad limits
  GlobalPadNumber getPadNumberInPartition(int partition, int row, int pad) const
  {
    const auto& info = mMapPartitionInfo[partition % mMapPartitionInfo.size()];
    const size_t rowOffset = info.getGlobalRowOffset();
    const size_t padOffset = mMapPadOffsetPerRow[rowOffset];

    return mMapPadOffsetPerRow[row + rowOffset] + pad - padOffset;
  }

  /// return the global pad number in a region for region, row, pad
  /// \return global pad number in a region for region, row, pad
  /// \todo add check for row and pad limits
  GlobalPadNumber getPadNumberInRegion(CRU cru, int row, int pad) const
  {
    const auto& info = mMapPadRegionInfo[cru.region() % mMapPadRegionInfo.size()];
    const size_t rowOffset = info.getGlobalRowOffset();
    const size_t padOffset = mMapPadOffsetPerRow[rowOffset];

    return mMapPadOffsetPerRow[row + rowOffset] + pad - padOffset;
  }

  /// return pad number for a pad subset type
  /// \return global pad number in a padsubset
  /// \param padSubset pad subset type (e.g. PadSubset::ROC)
  /// \param padSubsetNumber number of the pad subset (e.g. 10 for ROC 10)
  /// \param row row
  /// \param pad pad
  GlobalPadNumber getPadNumber(const PadSubset padSubset, const size_t padSubsetNumber, const int row,
                               const int pad) const
  {
    switch (padSubset) {
      case PadSubset::ROC: {
        return getPadNumberInROC(PadROCPos(padSubsetNumber, row, pad));
        break;
      }
      case PadSubset::Partition: {
        return getPadNumberInPartition(padSubsetNumber, row, pad);
        break;
      }
      case PadSubset::Region: {
        return getPadNumberInRegion(CRU(padSubsetNumber), row, pad);
        break;
      }
    }
    return 0;
  }

  GlobalPadNumber globalPadNumber(const FECInfo& fec) const
  {
    return mMapFECIDGlobalPad[FECInfo::globalSAMPAId(fec.getIndex(), fec.getSampaChip(), fec.getSampaChannel())];
  }
  GlobalPadNumber globalPadNumber(const int fecInSector, const int sampaOnFEC, const int channelOnSAMPA) const
  {
    return mMapFECIDGlobalPad[FECInfo::globalSAMPAId(fecInSector, sampaOnFEC, channelOnSAMPA)];
  }

  GlobalPosition2D getPadCentre(const PadSecPos& padSec) const
  {
    PadCentre padcent = getPadCentre(padSec.getPadPos());
    if (padSec.getSector().side() == Side::A) {
      padcent.SetY(-1.f * padcent.Y());
    }
    return LocalToGlobal(padcent, padSec.getSector());
  }

  GlobalPosition2D getPadCentre(const PadROCPos& padRoc) const
  {
    const int row = (padRoc.getROCType() == RocType::IROC) ? padRoc.getRow() : padRoc.getRow() + mNumberOfPadRowsIROC;
    const PadSecPos pos(padRoc.getSector(), PadPos(row, padRoc.getPad()));
    return getPadCentre(pos);
  }

  const FECInfo& getFECInfo(const PadROCPos& padROC) const
  {
    const PadPos globalPadPosition = getGlobalPadPos(padROC);
    const GlobalPadNumber padNum = globalPadNumber(globalPadPosition);
    return fecInfo(padNum);
  }

  // ===| global sector mappings |==============================================
  const PadCentre& getPadCentre(const PadPos& pad) const
  {
    const GlobalPadNumber padNumber = globalPadNumber(pad);
    return padCentre(padNumber);
  }
  const PadPos& padPos(const FECInfo& fec) const
  {
    const GlobalPadNumber padNumber = globalPadNumber(fec);
    return padPos(padNumber);
  }

  const PadPos& padPos(const int fecInSector, const int sampaOnFEC, const int channelOnSAMPA) const
  {
    const GlobalPadNumber padNumber = globalPadNumber(fecInSector, sampaOnFEC, channelOnSAMPA);
    return padPos(padNumber);
  }

  const PadCentre& padCentre(const FECInfo& fec) const
  {
    const GlobalPadNumber padNumber = globalPadNumber(fec);
    return padCentre(padNumber);
  }

  const PadCentre& padCentre(const int fecInSector, const int sampaOnFEC, const int channelOnSAMPA) const
  {
    const GlobalPadNumber padNumber = globalPadNumber(fecInSector, sampaOnFEC, channelOnSAMPA);
    return padCentre(padNumber);
  }

  // ===| partition mappings |==================================================
  const PadPos& padPos(const int partition, const int fecInPartition, const int sampaOnFEC,
                       const int channelOnSAMPA) const
  {
    const int fecInSector = mMapPartitionInfo[partition].getSectorFECOffset() + fecInPartition;
    const GlobalPadNumber padNumber = globalPadNumber(fecInSector, sampaOnFEC, channelOnSAMPA);
    return padPos(padNumber);
  }

  const PadPos padPosPartition(const int partition, const int fecInPartition, const int sampaOnFEC,
                               const int channelOnSAMPA) const
  {
    const PartitionInfo& partInfo = mMapPartitionInfo[partition];
    const int fecInSector = partInfo.getSectorFECOffset() + fecInPartition;
    const GlobalPadNumber padNumber = globalPadNumber(fecInSector, sampaOnFEC, channelOnSAMPA);
    PadPos pos = padPos(padNumber);
    pos.setRow(pos.getRow() - partInfo.getGlobalRowOffset());
    return pos;
  }

  const PadPos padPosRegion(const int cruNumber, const int fecInPartition, const int sampaOnFEC,
                            const int channelOnSAMPA) const
  {
    const CRU cru(cruNumber);
    const PadRegionInfo& regionInfo = mMapPadRegionInfo[cru.region()];
    const PartitionInfo& partInfo = mMapPartitionInfo[cru.partition()];
    const int fecInSector = partInfo.getSectorFECOffset() + fecInPartition;
    const GlobalPadNumber padNumber = globalPadNumber(fecInSector, sampaOnFEC, channelOnSAMPA);
    PadPos pos = padPos(padNumber);
    pos.setRow(pos.getRow() - regionInfo.getGlobalRowOffset());
    return pos;
  }

  const PadROCPos padROCPos(const CRU cru, const int fecInPartition, const int sampaOnFEC,
                            const int channelOnSAMPA) const
  {
    const PartitionInfo& partInfo = mMapPartitionInfo[cru.partition()];
    const int fecInSector = partInfo.getSectorFECOffset() + fecInPartition;
    const GlobalPadNumber padNumber = globalPadNumber(fecInSector, sampaOnFEC, channelOnSAMPA);
    const ROC roc = cru.roc();
    PadROCPos pos(roc, padPos(padNumber));
    if (roc.isOROC()) {
      pos.getPadPos().setRow(pos.getRow() - mNumberOfPadRowsIROC);
    }
    return pos;
  }

  const PadSecPos padSecPos(const CRU cru, const int fecInPartition, const int sampaOnFEC,
                            const int channelOnSAMPA) const
  {
    const PartitionInfo& partInfo = mMapPartitionInfo[cru.partition()];
    const int fecInSector = partInfo.getSectorFECOffset() + fecInPartition;
    const GlobalPadNumber padNumber = globalPadNumber(fecInSector, sampaOnFEC, channelOnSAMPA);
    PadSecPos pos(cru.sector(), padPos(padNumber));
    return pos;
  }

  /// sampa on FEC and channel on SAMPA from the cru and the raw FEC channel (0-79) in the half FEC
  /// this is required for the link-based zero suppression
  static constexpr void getSampaAndChannelOnFEC(const int cruID, const size_t rawFECChannel, int& sampaOnFEC, int& channelOnSAMPA)
  {
    constexpr int sampaMapping[10] = {0, 0, 1, 1, 2, 3, 3, 4, 4, 2};
    constexpr int channelOffset[10] = {0, 16, 0, 16, 0, 0, 16, 0, 16, 16};

    const int regionIter = cruID % 2;
    const int istreamm = ((rawFECChannel % 10) / 2);
    const int partitionStream = istreamm + regionIter * 5;
    sampaOnFEC = sampaMapping[partitionStream];
    const int channel = (rawFECChannel % 2) + 2 * (rawFECChannel / 10);
    channelOnSAMPA = channel + channelOffset[partitionStream];
  }

  const PadCentre& padCentre(const int partition, const int fecInPartition, const int sampaOnFEC,
                             const int channelOnSAMPA) const
  {
    const int fecInSector = mMapPartitionInfo[partition].getSectorFECOffset() + fecInPartition;
    const GlobalPadNumber padNumber = globalPadNumber(fecInSector, sampaOnFEC, channelOnSAMPA);
    return padCentre(padNumber);
  }

  // ===| pad number and pad row mappings |=====================================
  int getNumberOfRows() const { return mNumberOfPadRowsIROC + mNumberOfPadRowsOROC; }

  /// \return returns number of pad rows in IROCs
  static constexpr auto getNumberOfRowsInIROC() { return mNumberOfPadRowsIROC; }

  /// \return returns number of pad rows in OROCs
  static constexpr auto getNumberOfRowsInOROC() { return mNumberOfPadRowsOROC; }

  int getNumberOfRowsROC(ROC roc) const
  {
    return (roc.rocType() == RocType::IROC) ? mNumberOfPadRowsIROC : mNumberOfPadRowsOROC;
  }
  int getNumberOfRowsRegion(int region) const
  {
    return mMapPadRegionInfo[region % getNumberOfPadRegions()].getNumberOfPadRows();
  }
  int getGlobalRowOffsetRegion(int region) const
  {
    return mMapPadRegionInfo[region % getNumberOfPadRegions()].getGlobalRowOffset();
  }
  int getNumberOfRowsPartition(CRU cru) const
  {
    return mMapPartitionInfo[cru % getNumberOfPartitions()].getNumberOfPadRows();
  }
  int getNumberOfPadRows(PadSubset padSubset, int position) const
  {
    switch (padSubset) {
      case PadSubset::ROC: {
        return getNumberOfRowsROC(position);
        break;
      }
      case PadSubset::Partition: {
        return getNumberOfRowsPartition(position);
        break;
      }
      case PadSubset::Region: {
        return getNumberOfRowsRegion(position);
        break;
      }
    }
    return 0;
  }

  int getNumberOfPadsInRowSector(int row) const { return mMapNumberOfPadsPerRow[row]; }
  int getPadOffsetInRowSector(int row) const { return mMapPadOffsetPerRow[row]; }
  int getNumberOfPadsInRowROC(int roc, int row) const
  {
    return mMapNumberOfPadsPerRow[row + (roc % 72 >= getNumberOfIROCs()) * mNumberOfPadRowsIROC];
  }
  int getNumberOfPadsInRowRegion(int region, int row) const
  {
    return mMapNumberOfPadsPerRow[row + mMapPadRegionInfo[region % getNumberOfPadRegions()].getGlobalRowOffset()];
  }
  int getNumberOfPadsInRowPartition(int partition, int row) const
  {
    return mMapNumberOfPadsPerRow[row + mMapPartitionInfo[partition % getNumberOfPartitions()].getGlobalRowOffset()];
  }
  int getNumberOfPadsInRow(PadSubset padSubset, int position, int row) const
  {
    switch (padSubset) {
      case PadSubset::ROC: {
        return getNumberOfPadsInRowROC(position, row);
        break;
      }
      case PadSubset::Partition: {
        return getNumberOfPadsInRowPartition(position, row);
        break;
      }
      case PadSubset::Region: {
        return getNumberOfPadsInRowRegion(position, row);
        break;
      }
    }
    return 0;
  }

  /// \return returns number of pads per side of the TPC
  static constexpr int getNumberOfPadsPerSide() { return getPadsInSector() * SECTORSPERSIDE; }

  /// Convert sector, row, pad to global pad row in sector and pad number
  const PadPos getGlobalPadPos(const PadROCPos& padROC) const
  {
    const char globalRow = padROC.getRow() + (padROC.getROCType() == RocType::OROC) * mNumberOfPadRowsIROC;
    const char pad = padROC.getPad();
    return PadPos(globalRow, pad);
  }

  // ===| Partition and Region mappings |=======================================
  const PadRegionInfo& getPadRegionInfo(const unsigned char region) const { return mMapPadRegionInfo[region]; }
  const std::array<PadRegionInfo, 10>& getMapPadRegionInfo() const { return mMapPadRegionInfo; }
  int getNumberOfPadRegions() const { return int(mMapPadRegionInfo.size()); }

  const PartitionInfo& getPartitionInfo(const unsigned char partition) const { return mMapPartitionInfo[partition]; }
  const std::array<PartitionInfo, 5>& getMapPartitionInfo() const { return mMapPartitionInfo; }
  int getNumberOfPartitions() const { return int(mMapPartitionInfo.size()); }

  const DigitPos findDigitPosFromLocalPosition(const LocalPosition3D& pos, const Sector& sec) const;
  const DigitPos findDigitPosFromGlobalPosition(const GlobalPosition3D& pos) const;
  const DigitPos findDigitPosFromGlobalPosition(const GlobalPosition3D& pos, const Sector& sector) const;

  bool isOutOfSector(GlobalPosition3D posEle, const Sector& sector, const float margin = 0.f) const;

  static constexpr unsigned short getNumberOfIROCs() { return 36; }
  static constexpr unsigned short getNumberOfOROCs() { return 36; }
  static constexpr unsigned short getPadsInIROC() { return mPadsInIROC; }
  static constexpr unsigned short getPadsInOROC1() { return mPadsInOROC1; }
  static constexpr unsigned short getPadsInOROC2() { return mPadsInOROC2; }
  static constexpr unsigned short getPadsInOROC3() { return mPadsInOROC3; }
  static constexpr unsigned short getPadsInOROC() { return mPadsInOROC; }
  static constexpr unsigned short getPadsInSector() { return mPadsInSector; }

  unsigned short getNumberOfPads(const GEMstack gemStack) const
  {
    switch (gemStack) {
      case IROCgem: {
        return getPadsInIROC();
        break;
      }
      case OROC1gem: {
        return getPadsInOROC1();
        break;
      }
      case OROC2gem: {
        return getPadsInOROC2();
        break;
      }
      case OROC3gem: {
        return getPadsInOROC3();
        break;
      }
    }
    return 0;
  }

  unsigned short getNumberOfPads(const ROC roc) const
  {
    if (roc.rocType() == RocType::IROC) {
      return getPadsInIROC();
    }
    return getPadsInOROC();
  }

  const std::vector<PadPos>& getMapGlobalPadToPadPos() const { return mMapGlobalPadToPadPos; }
  const std::vector<int>& getMapFECIDGlobalPad() const { return mMapFECIDGlobalPad; }

  const std::vector<float>& getTraceLengthsIROC() const { return mTraceLengthsIROC; }
  const std::vector<float>& getTraceLengthsOROC() const { return mTraceLengthsOROC; }

  //   bool loadFECInfo();
  //   bool loadPositions();

  // c++11 feature don't work with root dictionary :(
  //   Mapper(const Mapper&) = delete;
  //   void operator=(const Mapper&) = delete;

  // ===| rotation functions |==================================================
  static GlobalPosition3D LocalToGlobal(const LocalPosition3D& pos, const double alpha)
  {
    const double cs = std::cos(alpha), sn = std::sin(alpha);
    return GlobalPosition3D(float(double(pos.X()) * cs - double(pos.Y()) * sn),
                            float(double(pos.X()) * sn + double(pos.Y() * cs)), pos.Z());
  }

  static LocalPosition3D GlobalToLocal(const GlobalPosition3D& pos, const double alpha)
  {
    ///@todo: Lookup over sector number
    const double cs = std::cos(-alpha), sn = std::sin(-alpha);
    return LocalPosition3D(float(double(pos.X()) * cs - double(pos.Y()) * sn),
                           float(double(pos.X()) * sn + double(pos.Y() * cs)), pos.Z());
  }

  static GlobalPosition3D LocalToGlobal(const LocalPosition3D& pos, const Sector sec)
  {
    const double cs = CosinsPerSector[sec.getSector() % SECTORSPERSIDE],
                 sn = SinsPerSector[sec.getSector() % SECTORSPERSIDE];
    return GlobalPosition3D(float(double(pos.X()) * cs - double(pos.Y()) * sn),
                            float(double(pos.X()) * sn + double(pos.Y() * cs)), pos.Z());
  }

  static LocalPosition3D GlobalToLocal(const GlobalPosition3D& pos, const Sector sec)
  {
    ///@todo: Lookup over sector number
    const double cs = CosinsPerSector[sec.getSector() % SECTORSPERSIDE],
                 sn = -SinsPerSector[sec.getSector() % SECTORSPERSIDE];
    return LocalPosition3D(float(double(pos.X()) * cs - double(pos.Y()) * sn),
                           float(double(pos.X()) * sn + double(pos.Y() * cs)), pos.Z());
  }

  // --- 2D
  static GlobalPosition2D LocalToGlobal(const LocalPosition2D& pos, const double alpha)
  {
    const double cs = std::cos(alpha), sn = std::sin(alpha);
    return GlobalPosition2D(float(double(pos.X()) * cs - double(pos.Y()) * sn),
                            float(double(pos.X()) * sn + double(pos.Y() * cs)));
  }

  static LocalPosition2D GlobalToLocal(const GlobalPosition2D& pos, const double alpha)
  {
    ///@todo: Lookup over sector number
    const double cs = std::cos(-alpha), sn = std::sin(-alpha);
    return LocalPosition2D(float(double(pos.X()) * cs - double(pos.Y()) * sn),
                           float(double(pos.X()) * sn + double(pos.Y() * cs)));
  }

  static GlobalPosition2D LocalToGlobal(const LocalPosition2D& pos, const Sector sec)
  {
    const double cs = CosinsPerSector[sec.getSector() % SECTORSPERSIDE],
                 sn = SinsPerSector[sec.getSector() % SECTORSPERSIDE];
    return GlobalPosition2D(float(double(pos.X()) * cs - double(pos.Y()) * sn),
                            float(double(pos.X()) * sn + double(pos.Y() * cs)));
  }

  static LocalPosition2D GlobalToLocal(const GlobalPosition2D& pos, const Sector sec)
  {
    ///@todo: Lookup over sector number
    const double cs = CosinsPerSector[sec.getSector() % SECTORSPERSIDE],
                 sn = -SinsPerSector[sec.getSector() % SECTORSPERSIDE];
    return LocalPosition2D(float(double(pos.X()) * cs - double(pos.Y()) * sn),
                           float(double(pos.X()) * sn + double(pos.Y() * cs)));
  }

  static constexpr unsigned int NSECTORS{36};                                                                                                                         ///< total number of sectors in the TPC
  static constexpr unsigned int NREGIONS{10};                                                                                                                         ///< total number of regions in one sector
  static constexpr unsigned int PADROWS{152};                                                                                                                         ///< total number of pad rows
  static constexpr unsigned int PADSPERREGION[NREGIONS]{1200, 1200, 1440, 1440, 1440, 1440, 1600, 1600, 1600, 1600};                                                  ///< number of pads per CRU
  static constexpr unsigned int GLOBALPADOFFSET[NREGIONS]{0, 1200, 2400, 3840, 5280, 6720, 8160, 9760, 11360, 12960};                                                 ///< offset of number of pads for region
  static constexpr unsigned int ROWSPERREGION[NREGIONS]{17, 15, 16, 15, 18, 16, 16, 14, 13, 12};                                                                      ///< number of pad rows for region
  static constexpr unsigned int ROWOFFSET[NREGIONS]{0, 17, 32, 48, 63, 81, 97, 113, 127, 140};                                                                        ///< offset to calculate local row from global row
  static constexpr float REGIONAREA[NREGIONS]{374.4f, 378.f, 453.6f, 470.88f, 864.f, 864.f, 1167.36f, 1128.96f, 1449.6f, 1456.8f};                                    ///< volume of each region in cm^2
  static constexpr float INVPADAREA[NREGIONS]{1 / 0.312f, 1 / 0.315f, 1 / 0.315f, 1 / 0.327f, 1 / 0.6f, 1 / 0.6f, 1 / 0.7296f, 1 / 0.7056f, 1 / 0.906f, 1 / 0.9105f}; ///< inverse size of the pad area padwidth*padLength
  static constexpr unsigned REGION[PADROWS] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9}; ///< region for global pad row
  const inline static std::vector<unsigned int> ADDITIONALPADSPERROW[NREGIONS]{
    {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5},    // region 0
    {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4},          // region 1
    {0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4},       // region 2
    {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4},          // region 3
    {0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4}, // region 4
    {0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4},       // region 5
    {0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6},       // region 6
    {0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4},             // region 7
    {0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5},                // region 8
    {0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5}                    // region 9
  };                                                        ///< additional pads per row compared to first row
  const inline static std::vector<unsigned int> OFFSETCRULOCAL[NREGIONS]{
    {0, 66, 132, 198, 266, 334, 402, 472, 542, 612, 684, 756, 828, 902, 976, 1050, 1124},         // region 0
    {0, 76, 152, 228, 306, 384, 462, 542, 622, 702, 784, 866, 948, 1032, 1116},                   // region 1
    {0, 86, 172, 258, 346, 434, 522, 612, 702, 792, 882, 974, 1066, 1158, 1252, 1346},            // region 2
    {0, 92, 184, 276, 370, 464, 558, 654, 750, 846, 944, 1042, 1140, 1240, 1340},                 // region 3
    {0, 76, 152, 228, 304, 382, 460, 538, 618, 698, 778, 858, 940, 1022, 1104, 1188, 1272, 1356}, // region 4
    {0, 86, 172, 258, 346, 434, 522, 612, 702, 792, 882, 974, 1066, 1158, 1252, 1346},            // region 5
    {0, 94, 190, 286, 382, 480, 578, 676, 776, 876, 978, 1080, 1182, 1286, 1390, 1494},           // region 6
    {0, 110, 220, 332, 444, 556, 670, 784, 898, 1014, 1130, 1246, 1364, 1482},                    // region 7
    {0, 118, 236, 356, 476, 598, 720, 844, 968, 1092, 1218, 1344, 1472},                          // region 8
    {0, 128, 258, 388, 520, 652, 784, 918, 1052, 1188, 1324, 1462}                                // region 9
  };                                                                                              ///< row offset in cru for given local pad row
  const inline static std::vector<unsigned int> PADSPERROW[NREGIONS]{
    {66, 66, 66, 68, 68, 68, 70, 70, 70, 72, 72, 72, 74, 74, 74, 74, 76},      // region 0
    {76, 76, 76, 78, 78, 78, 80, 80, 80, 82, 82, 82, 84, 84, 84},              // region 1
    {86, 86, 86, 88, 88, 88, 90, 90, 90, 90, 92, 92, 92, 94, 94, 94},          // region 2
    {92, 92, 92, 94, 94, 94, 96, 96, 96, 98, 98, 98, 100, 100, 100},           // region 3
    {76, 76, 76, 76, 78, 78, 78, 80, 80, 80, 80, 82, 82, 82, 84, 84, 84, 84},  // region 4
    {86, 86, 86, 88, 88, 88, 90, 90, 90, 90, 92, 92, 92, 94, 94, 94},          // region 5
    {94, 96, 96, 96, 98, 98, 98, 100, 100, 102, 102, 102, 104, 104, 104, 106}, // region 6
    {110, 110, 112, 112, 112, 114, 114, 114, 116, 116, 116, 118, 118, 118},    // region 7
    {118, 118, 120, 120, 122, 122, 124, 124, 124, 126, 126, 128, 128},         // region 8
    {128, 130, 130, 132, 132, 132, 134, 134, 136, 136, 138, 138}               // region 9
  };                                                                           ///< number of pads per row in region
  static constexpr unsigned int OFFSETCRUGLOBAL[PADROWS]{
    0, 66, 132, 198, 266, 334, 402, 472, 542, 612, 684, 756, 828, 902, 976, 1050, 1124,         // region 0
    0, 76, 152, 228, 306, 384, 462, 542, 622, 702, 784, 866, 948, 1032, 1116,                   // region 1
    0, 86, 172, 258, 346, 434, 522, 612, 702, 792, 882, 974, 1066, 1158, 1252, 1346,            // region 2
    0, 92, 184, 276, 370, 464, 558, 654, 750, 846, 944, 1042, 1140, 1240, 1340,                 // region 3
    0, 76, 152, 228, 304, 382, 460, 538, 618, 698, 778, 858, 940, 1022, 1104, 1188, 1272, 1356, // region 4
    0, 86, 172, 258, 346, 434, 522, 612, 702, 792, 882, 974, 1066, 1158, 1252, 1346,            // region 5
    0, 94, 190, 286, 382, 480, 578, 676, 776, 876, 978, 1080, 1182, 1286, 1390, 1494,           // region 6
    0, 110, 220, 332, 444, 556, 670, 784, 898, 1014, 1130, 1246, 1364, 1482,                    // region 7
    0, 118, 236, 356, 476, 598, 720, 844, 968, 1092, 1218, 1344, 1472,                          // region 8
    0, 128, 258, 388, 520, 652, 784, 918, 1052, 1188, 1324, 1462                                // region 9
  };                                                                                            ///< row offset in cru for given global pad row

 private:
  Mapper(const std::string& mappingDir);
  // use old c++03 due to root
  Mapper(const Mapper&) {}
  void operator=(const Mapper&) {}

  void load(const std::string& mappingDir);

  /// load trace lengths
  void loadTraceLengths(std::string_view mappingDir = "");
  void setTraceLengths(std::string_view inputFile, std::vector<float>& length);

  void initPadRegionsAndPartitions();
  bool readMappingFile(std::string file);

  static constexpr unsigned short mPadsInIROC{5280};        ///< number of pads in IROC
  static constexpr unsigned short mPadsInOROC1{2880};       ///< number of pads in OROC1
  static constexpr unsigned short mPadsInOROC2{3200};       ///< number of pads in OROC2
  static constexpr unsigned short mPadsInOROC3{3200};       ///< number of pads in OROC3
  static constexpr unsigned short mPadsInOROC{9280};        ///< number of pads in OROC
  static constexpr unsigned short mPadsInSector{14560};     ///< number of pads in one sector
  static constexpr unsigned short mNumberOfPadRowsIROC{63}; ///< number of pad rows in IROC
  static constexpr unsigned short mNumberOfPadRowsOROC{89}; ///< number of pad rows in IROC

  std::vector<float> mTraceLengthsIROC; ///< trace lengths IROC
  std::vector<float> mTraceLengthsOROC; ///< trace lengths OROC

  // ===| lookup tables |=======================================================
  //   static constexpr std::array<double, SECTORSPERSIDE> SinsPerSector;   ///< Sinus values of sectors
  //   static constexpr std::array<double, SECTORSPERSIDE> CosinsPerSector; ///< Cosinus values of sectors
  //   for (double i=0; i<18; ++i) { cout << std::setprecision(40) << std::sin(TMath::DegToRad()*(10.+i*20.))
  //   <<","<<std::endl; }
  static constexpr std::array<double, SECTORSPERSIDE> SinsPerSector{
    {0.1736481776669303311866343619840336032212, 0.4999999999999999444888487687421729788184,
     0.7660444431189780134516809084743726998568, 0.9396926207859083168827396548294927924871, 1,
     0.9396926207859084279050421173451468348503, 0.7660444431189780134516809084743726998568,
     0.4999999999999999444888487687421729788184, 0.1736481776669302756754831307262065820396,
     -0.1736481776669304699645124401286011561751, -0.5000000000000001110223024625156540423632,
     -0.7660444431189779024293784459587186574936, -0.9396926207859084279050421173451468348503, -1,
     -0.9396926207859083168827396548294927924871, -0.7660444431189781244739833709900267422199,
     -0.5000000000000004440892098500626161694527, -0.1736481776669303866977855932418606244028}};

  //     static constexpr std::array<int, 2> test{1,2};

  // for (double i=0; i<18; ++i) { cout << std::setprecision(40) << std::cos(TMath::DegToRad()*(10.+i*20.))
  // <<","<<std::endl; }
  static constexpr std::array<double, SECTORSPERSIDE> CosinsPerSector{
    {0.9848077530122080203156542665965389460325, 0.866025403784438707610604524234076961875,
     0.6427876096865393629187224178167525678873, 0.34202014332566882393038554255326744169, 0.,
     -0.3420201433256687129080830800376133993268, -0.6427876096865393629187224178167525678873,
     -0.866025403784438707610604524234076961875, -0.9848077530122080203156542665965389460325,
     -0.9848077530122080203156542665965389460325, -0.8660254037844385965883020617184229195118,
     -0.6427876096865394739410248803324066102505, -0.3420201433256685463746293862641323357821, 0.,
     0.3420201433256689904638392363267485052347, 0.6427876096865392518964199553010985255241,
     0.8660254037844383745436971366871148347855, 0.9848077530122080203156542665965389460325}};

  static constexpr std::array<double, SECTORSPERSIDE> SinsPerSectorNotShifted{
    {0, 0.3420201433256687129080830800376133993268, 0.6427876096865392518964199553010985255241,
     0.8660254037844385965883020617184229195118, 0.9848077530122080203156542665965389460325,
     0.9848077530122080203156542665965389460325, 0.866025403784438707610604524234076961875,
     0.6427876096865394739410248803324066102505, 0.3420201433256688794415367738110944628716, 0.,
     -0.3420201433256686573969318487797863781452, -0.6427876096865392518964199553010985255241,
     -0.8660254037844383745436971366871148347855, -0.9848077530122080203156542665965389460325,
     -0.9848077530122081313379567291121929883957, -0.8660254037844385965883020617184229195118,
     -0.6427876096865395849633273428480606526136, -0.3420201433256686018857806175219593569636}}; ///< Array of sin for all sectors

  static constexpr std::array<double, SECTORSPERSIDE> CosinsPerSectorNotShifted{
    {1, 0.9396926207859084279050421173451468348503, 0.7660444431189780134516809084743726998568,
     0.5000000000000001110223024625156540423632, 0.1736481776669304144533612088707741349936,
     -0.1736481776669303034310587463551200926304, -0.4999999999999997779553950749686919152737,
     -0.7660444431189779024293784459587186574936, -0.9396926207859083168827396548294927924871, -1,
     -0.9396926207859084279050421173451468348503, -0.7660444431189780134516809084743726998568,
     -0.5000000000000004440892098500626161694527, -0.1736481776669303311866343619840336032212,
     0.1736481776669299703641513588081579655409, 0.5000000000000001110223024625156540423632,
     0.7660444431189777914070759834430646151304, 0.9396926207859084279050421173451468348503}}; ///< Array of cos for all sectors

  // ===| Pad Mappings |========================================================
  std::vector<PadPos> mMapGlobalPadToPadPos;  ///< mapping of global pad number to row and pad
  std::vector<PadCentre> mMapGlobalPadCentre; ///< pad coordinates
  std::map<PadPos, GlobalPadNumber>
    mMapPadPosGlobalPad;                     ///< mapping pad position to global pad number, most probably needs to be changed to speed up
  std::vector<int> mMapFECIDGlobalPad;       ///< mapping sector global FEC id to global pad number
  std::vector<FECInfo> mMapGlobalPadFECInfo; ///< map global pad number to FEC info

  // ===| Pad region and partition mappings |===================================
  std::array<PadRegionInfo, 10> mMapPadRegionInfo; ///< pad region information
  std::array<PartitionInfo, 5> mMapPartitionInfo;  ///< partition information

  // ===| Pad number and row mappings |=========================================
  std::array<int, mNumberOfPadRowsIROC + mNumberOfPadRowsOROC>
    mMapNumberOfPadsPerRow; ///< number of pads per global pad row in sector
  std::array<int, mNumberOfPadRowsIROC + mNumberOfPadRowsOROC>
    mMapPadOffsetPerRow; ///< global pad number offset in a row
};

// ===| inline functions |======================================================
inline const DigitPos Mapper::findDigitPosFromLocalPosition(const LocalPosition3D& pos, const Sector& sec) const
{
  PadPos pad;
  CRU cru;
  for (const PadRegionInfo& padRegion : mMapPadRegionInfo) {
    cru = CRU(sec, padRegion.getRegion());
    pad = padRegion.findPad(pos.X(), pos.Y(), (pos.Z() >= 0) ? Side::A : Side::C); // <--- to avoid calling a non-inlined library function layer for LocalPosition3D
    if (pad.isValid()) {
      break;
    }
  }

  return DigitPos(cru, pad);
}

inline const DigitPos Mapper::findDigitPosFromGlobalPosition(const GlobalPosition3D& pos) const
{
  // ===| find sector |=========================================================
  float phi = std::atan2(pos.Y(), pos.X());
  if (phi < 0.) {
    phi += TWOPI;
  }
  const unsigned char secNum = std::floor(phi / SECPHIWIDTH);
  // const float secPhi = secNum * SECPHIWIDTH + SECPHIWIDTH / 2.;
  Sector sec(secNum + (pos.Z() < 0) * SECTORSPERSIDE);

  // ===| rotated position |====================================================
  //   LocalPosition3D posLoc=GlobalToLocal(pos, secPhi);
  LocalPosition3D posLoc = GlobalToLocal(pos, Sector(secNum));

  return findDigitPosFromLocalPosition(posLoc, sec);
}
inline const DigitPos Mapper::findDigitPosFromGlobalPosition(const GlobalPosition3D& pos, const Sector& sector) const
{
  LocalPosition3D posLoc = GlobalToLocal(pos, sector);
  return findDigitPosFromLocalPosition(posLoc, sector);
}

inline bool Mapper::isOutOfSector(GlobalPosition3D posEle, const Sector& sector, const float margin) const
{
  int secRight = int(sector);
  int secLeft = int(Sector::getLeft(sector));
  const float dSectorBoundaryRight = -SinsPerSectorNotShifted[secRight % SECTORSPERSIDE] * posEle.X() + CosinsPerSectorNotShifted[secRight % SECTORSPERSIDE] * posEle.Y();
  const float dSectorBoundaryLeft = -SinsPerSectorNotShifted[secLeft % SECTORSPERSIDE] * posEle.X() + CosinsPerSectorNotShifted[secLeft % SECTORSPERSIDE] * posEle.Y();

  if ((dSectorBoundaryLeft > 0 && dSectorBoundaryRight < 0) || (dSectorBoundaryLeft < 0 && dSectorBoundaryRight > 0)) {
    return false;
  }
  if (std::abs(dSectorBoundaryLeft) > margin && std::abs(dSectorBoundaryRight) > margin) {
    return true;
  }
  return false;
}

} // namespace tpc
} // namespace o2

#endif
