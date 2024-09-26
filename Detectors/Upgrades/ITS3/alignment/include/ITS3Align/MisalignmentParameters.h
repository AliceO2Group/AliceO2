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

/// \file MisalignmentParameters.h
/// \brief Definition of the MisalignmentParameters class

#ifndef ITS3_MISALIGNMENTPARAMETERS_H_
#define ITS3_MISALIGNMENTPARAMETERS_H_

#include "ITS3Base/SpecsV2.h"

#include "TNamed.h"
#include "TFile.h"
#include "TMatrixD.h"

#include <array>
#include <string>

namespace o2::its3::align
{

class MisalignmentParameters : public TNamed
{
 public:
  MisalignmentParameters();

  // IO
  bool store(const std::string& file) const;
  static MisalignmentParameters* load(const std::string& file);

  /// Global getters
  double getGloTransX(unsigned int detID) const { return mGloTransX[detID]; }
  double getGloTransY(unsigned int detID) const { return mGloTransY[detID]; }
  double getGloTransZ(unsigned int detID) const { return mGloTransZ[detID]; }
  double getGloRotX(unsigned int detID) const { return mGloRotX[detID]; }
  double getGloRotY(unsigned int detID) const { return mGloRotY[detID]; }
  double getGloRotZ(unsigned int detID) const { return mGloRotZ[detID]; }
  /// Global setters
  void setGloTransX(unsigned int detID, double v) { mGloTransX[detID] = v; }
  void setGloTransY(unsigned int detID, double v) { mGloTransY[detID] = v; }
  void setGloTransZ(unsigned int detID, double v) { mGloTransZ[detID] = v; }
  void setGloRotX(unsigned int detID, double v) { mGloRotX[detID] = v; }
  void setGloRotY(unsigned int detID, double v) { mGloRotY[detID] = v; }
  void setGloRotZ(unsigned int detID, double v) { mGloRotZ[detID] = v; }

  /// Legendre Coeff. getters
  const TMatrixD& getLegendreCoeffX(unsigned int sensorID) const { return mLegCoeffX[sensorID]; }
  const TMatrixD& getLegendreCoeffY(unsigned int sensorID) const { return mLegCoeffY[sensorID]; }
  const TMatrixD& getLegendreCoeffZ(unsigned int sensorID) const { return mLegCoeffZ[sensorID]; }
  /// Legendre Coeff. setters
  void setLegendreCoeffX(unsigned int sensorID, const TMatrixD& m) { setMatrix(mLegCoeffX[sensorID], m); }
  void setLegendreCoeffY(unsigned int sensorID, const TMatrixD& m) { setMatrix(mLegCoeffY[sensorID], m); }
  void setLegendreCoeffZ(unsigned int sensorID, const TMatrixD& m) { setMatrix(mLegCoeffZ[sensorID], m); }

  void printParams(unsigned int detID) const;
  void printLegendreParams(unsigned int sensorID) const;

 private:
  inline void setMatrix(TMatrixD& o, const TMatrixD& n)
  {
    o.ResizeTo(n.GetNrows(), n.GetNcols());
    o = n;
  }

  static constexpr unsigned int nDetectors{constants::detID::nChips}; ///! for now just the IB

  // Global parameters
  std::array<double, nDetectors> mGloTransX; ///< Array to hold the global misalignment in x-direction
  std::array<double, nDetectors> mGloTransY; ///< Array to hold the global misalignment in y-direction
  std::array<double, nDetectors> mGloTransZ; ///< Array to hold the global misalignment in z-direction
  std::array<double, nDetectors> mGloRotX;   ///< Array to hold the global misalignment in x-direction
  std::array<double, nDetectors> mGloRotY;   ///< Array to hold the global misalignment in y-direction
  std::array<double, nDetectors> mGloRotZ;   ///< Array to hold the global misalignment in z-direction

  // Legendre Polynominals coefficients
  std::array<TMatrixD, constants::nSensorsIB> mLegCoeffX;
  std::array<TMatrixD, constants::nSensorsIB> mLegCoeffY;
  std::array<TMatrixD, constants::nSensorsIB> mLegCoeffZ;

  ClassDefOverride(MisalignmentParameters, 1);
};

} // namespace o2::its3::align

#endif
