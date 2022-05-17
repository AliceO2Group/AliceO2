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

/// \file IDCCCDBHelper.h
/// \brief helper class for accessing IDC0 and IDCDelta from CCDB
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#ifndef ALICEO2_TPC_IDCCCDBHELPER_H_
#define ALICEO2_TPC_IDCCCDBHELPER_H_

#include "DataFormatsTPC/Defs.h"
#include "TPCBase/Sector.h"
#include "CommonUtils/NameConf.h"
#include "Rtypes.h"
#include "TCanvas.h"
#include "CCDB/BasicCCDBManager.h"

namespace o2::tpc
{

class IDCGroupHelperSector;
struct IDCZero;
struct IDCOne;
struct FourierCoeff;
template <typename DataT>
struct IDCDelta;

/*
 Usage
 o2::tpc::IDCCCDBHelper<short> helper;
 // setting the IDC members manually
 helper.setIDCDelta(IDCDelta<DataT>* idcDelta);
 helper.setIDCZero(IDCZero* idcZero);
 helper.setIDCOne(IDCOne* idcOne);
 helper.setGroupingParameter(IDCGroupHelperSector* helperSector);
 // draw or access the IDCs
 const unsigned int sector = 10;
 const unsigned int integrationInterval = 0;
 helper.drawIDCZeroSide(o2::tpc::Side::A);
 helper.drawIDCDeltaSector(sector, integrationInterval);
 helper.drawIDCDeltaSide(o2::tpc::Side::A, integrationInterval);
 helper.drawIDCSide(o2::tpc::Side::A, integrationInterval);
 TODO add drawing of 1D-distributions
*/

/// \tparam DataT the data type for the IDCDelta which are stored in the CCDB (unsigned short, unsigned char, float)
template <typename DataT = unsigned short>
class IDCCCDBHelper
{
 public:
  /// constructor
  IDCCCDBHelper() = default;

  /// setting the IDCDelta class member
  void setIDCDelta(IDCDelta<DataT>* idcDelta) { mIDCDelta = idcDelta; }

  /// setting the 0D-IDCs
  void setIDCZero(IDCZero* idcZero) { mIDCZero = idcZero; }

  /// setting the 1D-IDCs
  void setIDCOne(IDCOne* idcOne) { mIDCOne = idcOne; }

  /// setting the fourier coefficients
  void setFourierCoeffs(FourierCoeff* fourier) { mFourierCoeff = fourier; }

  /// setting the grouping parameters
  void setGroupingParameter(IDCGroupHelperSector* helperSector) { mHelperSector = helperSector; }

  /// \return returns the number of integration intervals for IDCDelta
  unsigned int getNIntegrationIntervalsIDCDelta(const o2::tpc::Side side) const;

  /// \return returns the number of integration intervals for IDCOne
  unsigned int getNIntegrationIntervalsIDCOne(const o2::tpc::Side side) const;

  /// \return returns the stored IDC0 value for local ungrouped pad row and ungrouped pad
  /// \param sector sector
  /// \param region region
  /// \param urow row of the ungrouped IDCs
  /// \param upad pad number of the ungrouped IDCs
  float getIDCZeroVal(const unsigned int sector, const unsigned int region, unsigned int urow, unsigned int upad) const;

  /// \return returns the stored DeltaIDC value for local ungrouped pad row and ungrouped pad
  /// \param sector sector
  /// \param region region
  /// \param urow row of the ungrouped IDCs
  /// \param upad pad number of the ungrouped IDCs
  /// \param integrationInterval integration interval
  float getIDCDeltaVal(const unsigned int sector, const unsigned int region, unsigned int urow, unsigned int upad, unsigned int integrationInterval) const;

  /// \return returns IDCOne value
  /// \param side side of the TPC
  /// \param integrationInterval integration interval
  float getIDCOneVal(const o2::tpc::Side side, const unsigned int integrationInterval) const;

  /// \return returns the IDC value which is calculated with: (IDCDelta + 1) * IDCOne * IDCZero
  /// \param sector sector
  /// \param region region
  /// \param urow row of the ungrouped IDCs
  /// \param upad pad number of the ungrouped IDCs
  /// \param integrationInterval integration interval
  float getIDCVal(const unsigned int sector, const unsigned int region, unsigned int urow, unsigned int upad, unsigned int integrationInterval) const;

  /// draw IDC zero I_0(r,\phi) = <I(r,\phi,t)>_t
  /// \param side side which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawIDCZeroSide(const o2::tpc::Side side, const std::string filename = "IDCZeroSide.pdf") const { drawIDCZeroHelper(true, side == Side::A ? Sector(0) : Sector(Sector::MAXSECTOR - 1), filename); }

  /// draw IDCDelta for one side for one integration interval
  /// \param side side which will be drawn
  /// \param integrationInterval which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawIDCDeltaSide(const o2::tpc::Side side, const unsigned int integrationInterval, const std::string filename = "IDCDeltaSide.pdf") const { drawIDCDeltaHelper(true, side == Side::A ? Sector(0) : Sector(Sector::MAXSECTOR - 1), integrationInterval, filename); }

  /// draw IDCs which is calculated with: (IDCDelta + 1) * IDCOne * IDCZero
  /// \param side side which will be drawn
  /// \param integrationInterval which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawIDCSide(const o2::tpc::Side side, const unsigned int integrationInterval, const std::string filename = "IDCSide.pdf") const { drawIDCHelper(true, side == Side::A ? Sector(0) : Sector(Sector::MAXSECTOR - 1), integrationInterval, filename); }

  /// draw IDC zero I_0(r,\phi) = <I(r,\phi,t)>_t
  /// \param sector sector which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawIDCZeroSector(const unsigned int sector, const std::string filename = "IDCZeroSector.pdf") const { drawIDCZeroHelper(false, Sector(sector), filename); }

  /// draw IDCDelta for one sector for one integration interval
  /// \param sector sector which will be drawn
  /// \param integrationInterval which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawIDCDeltaSector(const unsigned int sector, const unsigned int integrationInterval, const std::string filename = "IDCDeltaSector.pdf") const { drawIDCDeltaHelper(false, Sector(sector), integrationInterval, filename); }

  /// draw IDC zero I_0(r,\phi) = <I(r,\phi,t)>_t
  /// \param sector sector which will be drawn
  /// \param integrationInterval which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawIDCSector(const unsigned int sector, const unsigned int integrationInterval, const std::string filename = "IDCSector.pdf") const { drawIDCHelper(false, Sector(sector), integrationInterval, filename); }

  TCanvas* drawIDCZeroCanvas(TCanvas* outputCanvas, std::string_view type, int nbins1D, float xMin1D, float xMax1D, int integrationInterval = -1) const;

  TCanvas* drawIDCZeroRadialProfile(TCanvas* outputCanvas, int nbinsY, float yMin, float yMax) const;

  TCanvas* drawIDCZeroStackCanvas(TCanvas* outputCanvas, Side side, std::string_view type, int nbins1D, float xMin1D, float xMax1D, int integrationInterval = -1) const;

  TCanvas* drawIDCOneCanvas(TCanvas* outputCanvas, int nbins1D, float xMin1D, float xMax1D, int integrationIntervals = -1) const;

  TCanvas* drawFourierCoeff(TCanvas* outputCanvas, Side side, int nbins1D, float xMin1D, float xMax1D) const;

  /// dumping the loaded IDCs to a tree for debugging
  /// \param integrationIntervals number of integration intervals which will be dumped to the tree (-1: all integration intervalls)
  /// \param outFileName name of the output file
  void dumpToTree(int integrationIntervals = -1, const char* outFileName = "IDCCCDBTree.root") const;

 private:
  IDCZero* mIDCZero = nullptr;                   ///< 0D-IDCs: ///< I_0(r,\phi) = <I(r,\phi,t)>_t
  IDCDelta<DataT>* mIDCDelta = nullptr;          ///< compressed or uncompressed Delta IDC: \Delta I(r,\phi,t) = I(r,\phi,t) / ( I_0(r,\phi) * I_1(t) )
  IDCOne* mIDCOne = nullptr;                     ///< I_1(t) = <I(r,\phi,t) / I_0(r,\phi)>_{r,\phi}
  IDCGroupHelperSector* mHelperSector = nullptr; ///< helper for accessing IDC0 and IDC-Delta
  FourierCoeff* mFourierCoeff = nullptr;

  /// helper function for drawing IDCZero
  /// \param sector sector which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawIDCZeroHelper(const bool type, const Sector sector, const std::string filename) const;

  /// helper function for drawing IDCDelta
  /// \param sector sector which will be drawn
  /// \param integrationInterval which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawIDCDeltaHelper(const bool type, const Sector sector, const unsigned int integrationInterval, const std::string filename) const;

  /// helper function for drawing IDC
  /// \param sector sector which will be drawn
  /// \param integrationInterval which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawIDCHelper(const bool type, const Sector sector, const unsigned int integrationInterval, const std::string filename) const;

  /// \return returns index to data from ungrouped pad and row
  /// \param sector sector
  /// \param region region
  /// \param urow row of the ungrouped IDCs
  /// \param upad pad number of the ungrouped IDCs
  /// \param integrationInterval integration interval
  unsigned int getUngroupedIndexGlobal(const unsigned int sector, const unsigned int region, unsigned int urow, unsigned int upad, unsigned int integrationInterval) const;

  ClassDefNV(IDCCCDBHelper, 1)
};

} // namespace o2::tpc

#endif
