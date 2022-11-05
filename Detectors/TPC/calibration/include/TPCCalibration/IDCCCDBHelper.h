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
/// \brief helper class for accessing IDCs from CCDB
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#ifndef ALICEO2_TPC_IDCCCDBHELPER_H_
#define ALICEO2_TPC_IDCCCDBHELPER_H_

#include "DataFormatsTPC/Defs.h"
#include "TPCBase/Sector.h"
#include "Rtypes.h"

class TCanvas;

namespace o2::tpc
{

class IDCGroupHelperSector;
struct IDCZero;
struct IDCOne;
struct FourierCoeff;
template <typename DataT>
struct IDCDelta;
enum class PadFlags : unsigned short;

template <class T>
class CalDet;

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
  void setIDCDelta(IDCDelta<DataT>* idcDelta, const Side side = Side::A) { mIDCDelta[side] = idcDelta; }

  /// setting the 0D-IDCs
  void setIDCZero(IDCZero* idcZero, const Side side = Side::A) { mIDCZero[side] = idcZero; }

  /// setting the 1D-IDCs
  void setIDCOne(IDCOne* idcOne, const Side side = Side::A) { mIDCOne[side] = idcOne; }

  /// setting the fourier coefficients
  void setFourierCoeffs(FourierCoeff* fourier, const Side side = Side::A) { mFourierCoeff[side] = fourier; }

  /// setting the grouping parameters
  void setGroupingParameter(IDCGroupHelperSector* helperSector, const Side side = Side::A) { mHelperSector[side] = helperSector; }

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

  /// create the outlier map with the set unscaled IDC0 map
  void createOutlierMap();

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
  void drawIDCZeroSide(const o2::tpc::Side side, const std::string filename = "IDCZeroSide.pdf", const float minZ = 0, const float maxZ = -1) const { drawIDCZeroHelper(true, side == Side::A ? Sector(0) : Sector(Sector::MAXSECTOR - 1), filename, minZ, maxZ); }

  /// draw IDCDelta for one side for one integration interval
  /// \param side side which will be drawn
  /// \param integrationInterval which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawIDCDeltaSide(const o2::tpc::Side side, const unsigned int integrationInterval, const std::string filename = "IDCDeltaSide.pdf", const float minZ = 0, const float maxZ = -1) const { drawIDCDeltaHelper(true, side == Side::A ? Sector(0) : Sector(Sector::MAXSECTOR - 1), integrationInterval, filename, minZ, maxZ); }

  /// draw IDCs which is calculated with: (IDCDelta + 1) * IDCOne * IDCZero
  /// \param side side which will be drawn
  /// \param integrationInterval which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawIDCSide(const o2::tpc::Side side, const unsigned int integrationInterval, const std::string filename = "IDCSide.pdf", const float minZ = 0, const float maxZ = -1) const { drawIDCHelper(true, side == Side::A ? Sector(0) : Sector(Sector::MAXSECTOR - 1), integrationInterval, filename, minZ, maxZ); }

  /// draw IDC zero I_0(r,\phi) = <I(r,\phi,t)>_t
  /// \param sector sector which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawIDCZeroSector(const unsigned int sector, const std::string filename = "IDCZeroSector.pdf", const float minZ = 0, const float maxZ = -1) const { drawIDCZeroHelper(false, Sector(sector), filename, minZ, maxZ); }

  /// draw IDCDelta for one sector for one integration interval
  /// \param sector sector which will be drawn
  /// \param integrationInterval which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawIDCDeltaSector(const unsigned int sector, const unsigned int integrationInterval, const std::string filename = "IDCDeltaSector.pdf", const float minZ = 0, const float maxZ = -1) const { drawIDCDeltaHelper(false, Sector(sector), integrationInterval, filename, minZ, maxZ); }

  /// draw IDC zero I_0(r,\phi) = <I(r,\phi,t)>_t
  /// \param sector sector which will be drawn
  /// \param integrationInterval which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawIDCSector(const unsigned int sector, const unsigned int integrationInterval, const std::string filename = "IDCSector.pdf", const float minZ = 0, const float maxZ = -1) const { drawIDCHelper(false, Sector(sector), integrationInterval, filename, minZ, maxZ); }

  /// draw the status map for the flags (for debugging) for a sector
  /// \param sector sector which will be drawn
  /// \flag flag which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawPadStatusFlagsMapSector(const unsigned int sector, const PadFlags flag, const std::string filename = "PadStatusFlags_Sector.pdf") const { drawPadFlagMap(false, Sector(sector), filename, flag); }

  /// draw the status map for the flags (for debugging) for a full side
  /// \param side side which will be drawn
  /// \param flag which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawPadStatusFlagsMapSide(const o2::tpc::Side side, const PadFlags flag, const std::string filename = "PadStatusFlags_Side.pdf") const { drawPadFlagMap(true, side == Side::A ? Sector(0) : Sector(Sector::MAXSECTOR - 1), filename, flag); }

  TCanvas* drawIDCZeroCanvas(TCanvas* outputCanvas, std::string_view type, int nbins1D, float xMin1D, float xMax1D, int integrationInterval = -1) const;

  TCanvas* drawIDCZeroRadialProfile(TCanvas* outputCanvas, int nbinsY, float yMin, float yMax) const;

  TCanvas* drawIDCZeroStackCanvas(TCanvas* outputCanvas, Side side, std::string_view type, int nbins1D, float xMin1D, float xMax1D, int integrationInterval = -1) const;

  TCanvas* drawIDCOneCanvas(TCanvas* outputCanvas, int nbins1D, float xMin1D, float xMax1D, int integrationIntervals = -1) const;

  TCanvas* drawFourierCoeff(TCanvas* outputCanvas, Side side, int nbins1D, float xMin1D, float xMax1D) const;

  /// dumping the loaded IDC0, IDC1 to a tree for debugging
  /// \param outFileName name of the output file
  void dumpToTree(const char* outFileName = "IDCCCDBTree.root") const;

  /// dumping the loaded fourier coefficients to a tree
  /// \param outFileName name of the output file
  void dumpToFourierCoeffToTree(const char* outFileName = "FourierCCDBTree.root") const;

  /// dumping the loaded IDC0, IDC1 to a tree for debugging
  /// \param outFileName name of the output file
  void dumpToTreeIDCDelta(const char* outFileName = "IDCCCDBTreeDeltaIDC.root") const;

  /// convert the loaded IDC0 map to a CalDet<float>
  /// \return returns CalDet containing the IDCZero
  CalDet<float> getIDCZeroCalDet() const;

  /// convert the loaded IDCDelta to a vector of CalDets
  /// \return returns std vector of CalDets containing the IDCDelta
  std::vector<CalDet<float>> getIDCDeltaCalDet() const;

  /// scale the stored IDC0 to 1
  /// \param rejectOutlier do not take outlier into account
  /// \return returns the scaling factor which was used to scale the IDC0
  float scaleIDC0(const Side side, const bool rejectOutlier = true);

 private:
  std::array<IDCZero*, SIDES> mIDCZero = {nullptr, nullptr};                   ///< 0D-IDCs: ///< I_0(r,\phi) = <I(r,\phi,t)>_t
  std::array<IDCDelta<DataT>*, SIDES> mIDCDelta = {nullptr, nullptr};          ///< compressed or uncompressed Delta IDC: \Delta I(r,\phi,t) = I(r,\phi,t) / ( I_0(r,\phi) * I_1(t) )
  std::array<IDCOne*, SIDES> mIDCOne = {nullptr, nullptr};                     ///< I_1(t) = <I(r,\phi,t) / I_0(r,\phi)>_{r,\phi}
  std::array<IDCGroupHelperSector*, SIDES> mHelperSector = {nullptr, nullptr}; ///< helper for accessing IDC0 and IDC-Delta
  std::array<FourierCoeff*, SIDES> mFourierCoeff = {nullptr, nullptr};         ///< fourier coefficients of IDCOne
  std::unique_ptr<CalDet<PadFlags>> mPadFlagsMap;                              ///< status flag for each pad (i.e. if the pad is dead)

  /// helper function for drawing IDCZero
  /// \param sector sector which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawIDCZeroHelper(const bool type, const Sector sector, const std::string filename, const float minZ, const float maxZ) const;

  /// helper function for drawing IDCDelta
  /// \param sector sector which will be drawn
  /// \param integrationInterval which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawIDCDeltaHelper(const bool type, const Sector sector, const unsigned int integrationInterval, const std::string filename, const float minZ, const float maxZ) const;

  /// helper function for drawing IDC
  /// \param sector sector which will be drawn
  /// \param integrationInterval which will be drawn
  /// \param filename name of the output file. If empty the canvas is drawn.
  void drawIDCHelper(const bool type, const Sector sector, const unsigned int integrationInterval, const std::string filename, const float minZ, const float maxZ) const;

  /// helper function for drawing
  void drawPadFlagMap(const bool type, const Sector sector, const std::string filename, const PadFlags flag) const;

  /// \return returns index to data from ungrouped pad and row
  /// \param sector sector
  /// \param region region
  /// \param urow row of the ungrouped IDCs
  /// \param upad pad number of the ungrouped IDCs
  /// \param integrationInterval integration interval
  unsigned int getUngroupedIndexGlobal(const unsigned int sector, const unsigned int region, unsigned int urow, unsigned int upad, unsigned int integrationInterval) const;

  ClassDefNV(IDCCCDBHelper, 3)
};

} // namespace o2::tpc

#endif
