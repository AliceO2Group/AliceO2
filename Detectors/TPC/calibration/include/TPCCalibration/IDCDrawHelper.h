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

/// \file IDCDrawHelper.h
/// \brief helper class for drawing IDCs per region/side
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#ifndef ALICEO2_TPC_IDCDRAWHELPER_H_
#define ALICEO2_TPC_IDCDRAWHELPER_H_

#include "DataFormatsTPC/Defs.h"
#include "functional"
#include "TPCCalibration/IDCContainer.h"

class TH2Poly;
class TCanvas;
class TH1F;
class TH2F;

namespace o2::tpc
{

class IDCDrawHelper
{

 public:
  /// helper struct containing a function to give access to the IDCs which will be drawn
  /// \param sector sector of the TPC
  /// \param region region in the TPC
  /// \row local row in the region
  /// \param pad pad in the row
  struct IDCDraw {
    float getIDC(const unsigned int sector, const unsigned int region, const unsigned int row, const unsigned int pad) const { return mIDCFunc(sector, region, row, pad); }
    std::function<float(const unsigned int, const unsigned int, const unsigned int, const unsigned int)> mIDCFunc; ///< function returning the value which will be drawn for sector, region, row, pad
  };

  /// draw sector
  /// \param idc IDCDraw struct containing function to get the values which will be drawn
  /// \param startRegion first region which will be drawn
  /// \param endRegion last region which will be drawn
  /// \param sector sector which will be drawn
  /// \param zAxisTitle axis title of the z axis
  /// \param fileName name of the output file (if empty the canvas is drawn instead of writte to a file)
  /// \param minZ min z value for drawing (if minZ > maxZ automatic z axis)
  /// \param maxZ max z value for drawing (if minZ > maxZ automatic z axis)
  static void drawSector(const IDCDraw& idc, const unsigned int startRegion, const unsigned int endRegion, const unsigned int sector, const std::string zAxisTitle, const std::string filename, const float minZ = 0, const float maxZ = -1);

  /// draw side
  /// \param idc IDCDraw struct containing function to get the values which will be drawn
  /// \param side side which will be drawn
  /// \param zAxisTitle axis title of the z axis
  /// \param fileName name of the output file (if empty the canvas is drawn instead of writte to a file)
  /// \param minZ min z value for drawing (if minZ > maxZ automatic z axis)
  /// \param maxZ max z value for drawing (if minZ > maxZ automatic z axis)
  static void drawSide(const IDCDraw& idc, const o2::tpc::Side side, const std::string zAxisTitle, const std::string filename, const float minZ = 0, const float maxZ = -1);
  static TH2Poly* drawSide(const IDCDraw& idc, const o2::tpc::Side side, const std::string zAxisTitle);
  static TH1F* drawSide(const IDCDraw& idc, std::string_view type, const o2::tpc::Side side, const int nbins1D, const float xMin1D, const float xMax1D);
  static void drawRadialProfile(const IDCDraw& idc, TH2F& hist, const o2::tpc::Side side);
  static void drawIDCZeroStackCanvas(const IDCDraw& idc, const o2::tpc::Side side, const std::string_view type, const int nbins1D, const float xMin1D, const float xMax1D, TCanvas& outputCanvas, int integrationInterval);

  struct IDCDrawGIF {
    std::function<float(const unsigned int, const unsigned int, const unsigned int, const unsigned int, const unsigned int)> mIDCFunc; ///< function returning the value which will be drawn for sector, region, row, pad
    std::function<float(const o2::tpc::Side side, const unsigned int)> mIDCOneFunc;                                                    ///< function returning the value which will be drawn for side, slice
  };

  /// \brief make a GIF of IDCs for A and C side and the 1D IDCs
  /// \param idc IDCDraw struct containing function to get the IDCs + 1D IDCs which will be drawn
  /// \param zAxisTitle axis title of the z axis
  /// \param fileName name of the output file (.gif is added automatically)
  /// \param minZ min z value for drawing (if minZ > maxZ automatic z axis)
  /// \param maxZ max z value for drawing (if minZ > maxZ automatic z axis)
  /// \param run run of the IDCs (if =-1 then no run number is drawn)
  static void drawSideGIF(const IDCDrawGIF& idcs, const unsigned int slices, const std::string zAxisTitle, const std::string filename = "IDCs", const float minZ = 0, const float maxZ = -1, const int run = -1);

  /// \return returns z axis title
  /// \param type IDC type
  /// \param compression compression of the IDCs if used (only for IDCDelta)
  static std::string getZAxisTitle(const IDCType type, const IDCDeltaCompression compression = IDCDeltaCompression::NO);

 private:
  static unsigned int getPad(const unsigned int pad, const unsigned int region, const unsigned int row, const Side side);
};

} // namespace o2::tpc

#endif
