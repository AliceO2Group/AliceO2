// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file IDCCCDBHelper.h
/// \brief helper class for accessing IDC0 and IDCDelta from CCDB
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#ifndef ALICEO2_TPC_IDCCCDBHELPER_H_
#define ALICEO2_TPC_IDCCCDBHELPER_H_

#include "TPCCalibration/IDCContainer.h"
#include "TPCCalibration/IDCGroupHelperSector.h"
#include "CCDB/BasicCCDBManager.h"
#include "Rtypes.h"

namespace o2::tpc
{

/// Usage
/// o2::tpc::IDCCCDBHelper<short> helper;
/// helper.setTimeStamp(0);
/// helper.loadAll();
/// helper.drawIDCZeroSide(o2::tpc::Side::A);
/// const unsigned int sector 10;
/// const unsigned int integrationInterval =3;
/// helper.drawIDCDeltaSector(sector, 3);
/// TODO add drawing of 1D-distributions

/// \tparam DataT the data type for the IDCDelta which are stored in the CCDB (short, char, float)
template <typename DataT = short>
class IDCCCDBHelper
{
 public:
  /// constructor
  /// \param uri path to CCDB
  IDCCCDBHelper(const char* uri = "http://ccdb-test.cern.ch:8080") { mCCDBManager.setURL(uri); }

  /// update timestamp (time frame)
  void setTimeStamp(const long long timestamp) { mCCDBManager.setTimestamp(timestamp); }

  /// load IDC-Delta, 0D-IDCs, grouping parameter needed for access
  void loadAll();

  /// load/update IDCDelta
  void loadIDCDelta() { mIDCDelta = mCCDBManager.get<o2::tpc::IDCDelta<DataT>>("TPC/Calib/IDC/IDCDELTA"); }

  /// load/update 0D-IDCs
  void loadIDCZero() { mIDCZero = mCCDBManager.get<o2::tpc::IDCZero>("TPC/Calib/IDC/IDC0"); }

  /// load/update grouping parameter
  void loadGroupingParameter() { mHelperSector = std::make_unique<IDCGroupHelperSector>(IDCGroupHelperSector{*mCCDBManager.get<o2::tpc::ParameterIDCGroupCCDB>("TPC/Calib/IDC/GROUPINGPAR")}); }

  /// \return returns the stored IDC0 value for local ungrouped pad row and ungrouped pad
  /// \param sector sector
  /// \param region region
  /// \param urow row of the ungrouped IDCs
  /// \param upad pad number of the ungrouped IDCs
  float getIDCZeroVal(const unsigned int sector, const unsigned int region, unsigned int urow, unsigned int upad) const { return mIDCZero->getValueIDCZero(Sector(sector).side(), mHelperSector->getIndexUngrouped(sector, region, urow, upad, 0)); }

  /// \return returns the stored DeltaIDC value for local ungrouped pad row and ungrouped pad
  /// \param sector sector
  /// \param region region
  /// \param urow row of the ungrouped IDCs
  /// \param upad pad number of the ungrouped IDCs
  /// \param chunk chunk of the Delta IDC (can be obtained with getLocalIntegrationInterval())
  /// \param localintegrationInterval local integration interval for chunk (can be obtained with getLocalIntegrationInterval())
  float getIDCDeltaVal(const unsigned int sector, const unsigned int region, unsigned int urow, unsigned int upad, unsigned int localintegrationInterval) const { return mIDCDelta->getValue(Sector(sector).side(), mHelperSector->getIndexUngrouped(sector, region, urow, upad, localintegrationInterval)); }

  /// draw IDC zero I_0(r,\phi) = <I(r,\phi,t)>_t
  /// \param side side which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawIDCZeroSide(const o2::tpc::Side side, const std::string filename = "IDCZeroSide.pdf") const { drawSide(IDCType::IDCZero, side, 0, filename); }

  /// draw IDCDelta for one side for one integration interval
  /// \param side side which will be drawn
  /// \param integrationInterval which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawIDCDeltaSide(const o2::tpc::Side side, const unsigned int integrationInterval, const std::string filename = "IDCDeltaSide.pdf") const { drawSide(IDCType::IDCDelta, side, integrationInterval, filename); }

  /// draw IDC zero I_0(r,\phi) = <I(r,\phi,t)>_t
  /// \param sector sector which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawIDCZeroSector(const unsigned int sector, const std::string filename = "IDCZeroSector.pdf") const { drawSector(IDCType::IDCZero, sector, 0, filename); }

  /// draw IDCDelta for one sector for one integration interval
  /// \param sector sector which will be drawn
  /// \param integrationInterval which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawIDCDeltaSector(const unsigned int sector, const unsigned int integrationInterval, const std::string filename = "IDCDeltaSector.pdf") const { drawSector(IDCType::IDCDelta, sector, integrationInterval, filename); }

 private:
  IDCZero* mIDCZero = nullptr;                                                      ///< 0D-IDCs: ///< I_0(r,\phi) = <I(r,\phi,t)>_t
  IDCDelta<DataT>* mIDCDelta = nullptr;                                             ///< compressed or uncompressed Delta IDC: \Delta I(r,\phi,t) = I(r,\phi,t) / ( I_0(r,\phi) * I_1(t) )
  std::unique_ptr<IDCGroupHelperSector> mHelperSector{};                            ///< helper for accessing IDC0 and IDC-Delta
  o2::ccdb::BasicCCDBManager mCCDBManager = o2::ccdb::BasicCCDBManager::instance(); ///< CCDB manager for loading objects

  /// draw IDCs for one side for one integration interval
  /// \param Side side which will be drawn
  /// \param integrationInterval which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawSide(const IDCType type, const o2::tpc::Side side, const unsigned int integrationInterval, const std::string filename) const;

  /// draw IDCs for one sector for one integration interval
  /// \param sector sector which will be drawn
  /// \param integrationInterval which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawSector(const IDCType type, const unsigned int sector, const unsigned int integrationInterval, const std::string filename) const;

  /// return returns title for z axis for given IDCType
  std::string getZAxisTitle(const o2::tpc::IDCType type) const;

  ClassDefNV(IDCCCDBHelper, 1)
};

} // namespace o2::tpc

#endif
