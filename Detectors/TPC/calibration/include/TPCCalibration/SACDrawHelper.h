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

/// \file SACDrawHelper.h
/// \brief helper class for drawing SACs per sector/side
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#ifndef ALICEO2_TPC_SACDRAWHELPER_H_
#define ALICEO2_TPC_SACDRAWHELPER_H_

#include "DataFormatsTPC/Defs.h"
#include "functional"
#include "TPCCalibration/IDCContainer.h"

class TH2Poly;

namespace o2::tpc
{

class SACDrawHelper
{

 public:
  /// helper struct containing a function to give access to the SACs which will be drawn
  /// \param sector sector of the TPC
  /// \param stack gem stack
  struct SACDraw {
    float getSAC(const unsigned int sector, const unsigned int stack) const { return mSACFunc(sector, stack); }
    std::function<float(const unsigned int, const unsigned int)> mSACFunc; ///< function returning the value which will be drawn for sector, stack
  };

  /// draw sector
  /// \param SAC SACDraw struct containing function to get the values which will be drawn
  /// \param sector sector which will be drawn
  /// \param zAxisTitle axis title of the z axis
  /// \param fileName name of the output file (if empty the canvas is drawn instead of writte to a file)
  /// \param minZ min z value for drawing (if minZ > maxZ automatic z axis)
  /// \param maxZ max z value for drawing (if minZ > maxZ automatic z axis)
  static void drawSector(const SACDraw& SAC, const unsigned int sector, const std::string zAxisTitle, const std::string filename, const float minZ = 0, const float maxZ = -1);

  /// draw side
  /// \param SAC SACDraw struct containing function to get the values which will be drawn
  /// \param side side which will be drawn
  /// \param zAxisTitle axis title of the z axis
  /// \param fileName name of the output file (if empty the canvas is drawn instead of writte to a file)
  /// \param minZ min z value for drawing (if minZ > maxZ automatic z axis)
  /// \param maxZ max z value for drawing (if minZ > maxZ automatic z axis)
  static void drawSide(const SACDraw& SAC, const o2::tpc::Side side, const std::string zAxisTitle, const std::string filename, const float minZ = 0, const float maxZ = -1);

  static TH2Poly* drawSide(const SACDraw& SAC, const o2::tpc::Side side, const std::string zAxisTitle);

  /// \return returns z axis title
  /// \param type SAC type
  static std::string getZAxisTitle(const SACType type);
};

} // namespace o2::tpc

#endif
