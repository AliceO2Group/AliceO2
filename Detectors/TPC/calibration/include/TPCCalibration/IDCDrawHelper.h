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

  /// \return returns z axis title
  /// \param type IDC type
  /// \param compression compression of the IDCs if used (only for IDCDelta)
  static std::string getZAxisTitle(const IDCType type, const IDCDeltaCompression compression = IDCDeltaCompression::NO);
};

} // namespace o2::tpc

#endif
