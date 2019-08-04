// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

  /// return the cru number from sector and global pad number
  /// \param sec sector
  /// \param globalPad global pad number in sector
  /// \return global cru number
  int getCRU(const Sector& sec, GlobalPadNumber globalPad)
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
    const PadCentre& padcent = getPadCentre(padSec.getPadPos());
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

  const PadPos padPosRegion(const int cruNumber, const int fecInRegion, const int sampaOnFEC,
                            const int channelOnSAMPA) const

  {
    const CRU cru(cruNumber);
    const PadRegionInfo& regionInfo = mMapPadRegionInfo[cru.region()];
    const PartitionInfo& partInfo = mMapPartitionInfo[cru.partition()];
    const int fecInSector = partInfo.getSectorFECOffset() + fecInRegion;
    const GlobalPadNumber padNumber = globalPadNumber(fecInSector, sampaOnFEC, channelOnSAMPA);
    PadPos pos = padPos(padNumber);
    pos.setRow(pos.getRow() - regionInfo.getGlobalRowOffset());
    return pos;
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
  }

  unsigned short getNumberOfPads(const ROC roc) const
  {
    if (roc.rocType() == RocType::IROC) {
      return getPadsInIROC();
    }
    return getPadsInOROC();
  }

  //   bool loadFECInfo();
  //   bool loadTraceLengh();
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

 private:
  Mapper(const std::string& mappingDir);
  // use old c++03 due to root
  Mapper(const Mapper&) {}
  void operator=(const Mapper&) {}

  void load(const std::string& mappingDir);
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
    cru = CRU(sec, padRegion.getPartition());
    pad = padRegion.findPad(pos);
    if (pad.isValid())
      break;
  }

  return DigitPos(cru, pad);
}

inline const DigitPos Mapper::findDigitPosFromGlobalPosition(const GlobalPosition3D& pos) const
{
  // ===| find sector |=========================================================
  float phi = std::atan2(pos.Y(), pos.X());
  if (phi < 0.)
    phi += TWOPI;
  const unsigned char secNum = std::floor(phi / SECPHIWIDTH);
  const float secPhi = secNum * SECPHIWIDTH + SECPHIWIDTH / 2.;
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
