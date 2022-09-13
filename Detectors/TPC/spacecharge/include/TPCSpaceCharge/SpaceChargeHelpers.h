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

/// \file SpaceChargeHelpers.h
/// \brief This file provides all necesseray classes which are used during the calcution of the distortions and corrections
///
/// \author  Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
/// \date Aug 21, 2020

#ifndef ALICEO2_TPC_SPACECHARGEHELPERS_H_
#define ALICEO2_TPC_SPACECHARGEHELPERS_H_

#include <functional>
#include <cmath>
#include "TPCSpaceCharge/TriCubic.h"
#include "DataFormatsTPC/Defs.h"

namespace o2
{
namespace tpc
{

///
/// this class contains an analytical description of the space charge, potential and the electric fields.
/// The analytical functions can be used to test the poisson solver and the caluclation of distortions/corrections.
///
template <typename DataT = double>
class AnalyticalFields
{
 public:
  AnalyticalFields(const o2::tpc::Side side = o2::tpc::Side::A) : mSide{side} {};

  o2::tpc::Side getSide() const { return mSide; }

  void setSide(const o2::tpc::Side side) { mSide = side; }

  /// sets the parameters
  void setParameters(const DataT parA, const DataT parB, const DataT parC)
  {
    mParA = parA;
    mParB = parB;
    mParC = parC;
  }

  /// return parameter A
  DataT getParA() const { return mParA; }

  /// return parameter B
  DataT getParB() const { return mParB; }

  /// return parameter C
  DataT getParC() const { return mParC; }

  /// \param r r coordinate
  /// \param phi phi coordinate
  /// \param z z coordinate
  /// \return returns the function value for electric field Er for given coordinate
  DataT evalEr(DataT z, DataT r, DataT phi) const { return mErFunc(z, r, phi); }

  /// \param r r coordinate
  /// \param phi phi coordinate
  /// \param z z coordinate
  /// \return returns the function value for electric field Ez for given coordinate
  DataT evalEz(DataT z, DataT r, DataT phi) const { return mEzFunc(z, r, phi); }

  /// \param r r coordinate
  /// \param phi phi coordinate
  /// \param z z coordinate
  /// \return returns the function value for electric field Ephi for given coordinate
  DataT evalEphi(DataT z, DataT r, DataT phi) const { return mEphiFunc(z, r, phi); }

  /// \param r r coordinate
  /// \param phi phi coordinate
  /// \param z z coordinate
  /// \return returns the function value for the potential for given coordinate
  DataT evalPotential(DataT z, DataT r, DataT phi) const { return mPotentialFunc(z, r, phi); }

  /// \param r r coordinate
  /// \param phi phi coordinate
  /// \param z z coordinate
  /// \return returns the function value for the space charge density for given coordinate
  DataT evalDensity(DataT z, DataT r, DataT phi) const { return mDensityFunc(z, r, phi); }

  /// analytical potential
  std::function<DataT(DataT, DataT, DataT)> mPotentialFunc = [& mParA = mParA, &mParB = mParB, &mParC = mParC](const DataT z, const DataT r, const DataT phi) {
    const DataT zz = std::abs(z);
    return -mParA * (std::pow((-r + 254.5 + 83.5), 4) - 338.0 * std::pow((-r + 254.5 + 83.5), 3) + 21250.75 * std::pow((-r + 254.5 + 83.5), 2)) * std::cos(mParB * phi) * std::cos(mParB * phi) * std::exp(-1 * mParC * (zz - 125) * (zz - 125));
  };

  /// analytical space charge - NOTE: if the space charge density is calculated analytical there would be a - sign in the formula (-mParA)  - however since its an e- the sign is flipped (IS THIS CORRECT??? see for minus sign: AliTPCSpaceCharge3DCalc::SetPotentialBoundaryAndChargeFormula)-
  std::function<DataT(DataT, DataT, DataT)> mDensityFunc = [& mParA = mParA, &mParB = mParB, &mParC = mParC](const DataT z, const DataT r, const DataT phi) {
    const DataT zz = std::abs(z);
    return mParA * ((1 / r * 16 * (-3311250 + 90995.5 * r - 570.375 * r * r + r * r * r)) * std::cos(mParB * phi) * std::cos(mParB * phi) * std::exp(-1 * mParC * (zz - 125) * (zz - 125)) +
                    (std::pow(-r + 254.5 + 83.5, 4) - 338.0 * std::pow(-r + 254.5 + 83.5, 3) + 21250.75 * std::pow(-r + 254.5 + 83.5, 2)) / (r * r) * std::exp(-1 * mParC * (zz - 125) * (zz - 125)) * -2 * mParB * mParB * std::cos(2 * mParB * phi) +
                    (std::pow(-r + 254.5 + 83.5, 4) - 338.0 * std::pow(-r + 254.5 + 83.5, 3) + 21250.75 * std::pow(-r + 254.5 + 83.5, 2)) * std::cos(mParB * phi) * std::cos(mParB * phi) * 2 * mParC * std::exp(-1 * mParC * (zz - 125) * (zz - 125)) * (2 * mParC * (zz - 125) * (zz - 125) - 1));
  };

  /// analytical electric field Er
  std::function<DataT(DataT, DataT, DataT)> mErFunc = [& mParA = mParA, &mParB = mParB, &mParC = mParC](const DataT z, const DataT r, const DataT phi) {
    const DataT zz = std::abs(z);
    return mParA * 4 * (r * r * r - 760.5 * r * r + 181991 * r - 1.3245 * std::pow(10, 7)) * std::cos(mParB * phi) * std::cos(mParB * phi) * std::exp(-1 * mParC * (zz - 125) * (zz - 125));
  };

  /// analytical electric field Ephi
  std::function<DataT(DataT, DataT, DataT)> mEphiFunc = [& mParA = mParA, &mParB = mParB, &mParC = mParC](const DataT z, const DataT r, const DataT phi) {
    const DataT zz = std::abs(z);
    return mParA * (std::pow(-r + 254.5 + 83.5, 4) - 338.0 * std::pow(-r + 254.5 + 83.5, 3) + 21250.75 * (-r + 254.5 + 83.5) * (-r + 254.5 + 83.5)) / r * std::exp(-1 * mParC * (zz - 125) * (zz - 125)) * -mParB * std::sin(2 * mParB * phi);
  };

  /// analytical electric field Ez
  std::function<DataT(DataT, DataT, DataT)> mEzFunc = [& mParA = mParA, &mParB = mParB, &mParC = mParC](const DataT z, const DataT r, const DataT phi) {
    const DataT zz = std::abs(z);
    return mParA * (std::pow(-r + 254.5 + 83.5, 4) - 338.0 * std::pow(-r + 254.5 + 83.5, 3) + 21250.75 * (-r + 254.5 + 83.5) * (-r + 254.5 + 83.5)) * std::cos(mParB * phi) * std::cos(mParB * phi) * -2 * mParC * (zz - 125) * std::exp(-1 * mParC * (zz - 125) * (zz - 125));
  };

  static constexpr unsigned int getID() { return ID; }

 private:
  static constexpr unsigned int ID = 0;  ///< needed to distinguish between the differrent classes
  DataT mParA{1e-5};                     ///< parameter [0] of functions
  DataT mParB{0.5};                      ///< parameter [1] of functions
  DataT mParC{1e-4};                     ///< parameter [2] of functions
  o2::tpc::Side mSide{o2::tpc::Side::A}; ///< side of the TPC. Since the absolute value is taken during the calculations the choice of the side is arbitrary.
};

///
/// This class gives tricubic interpolation of the electric fields and can be used to calculate the distortions/corrections.
/// The electric fields have to be calculated by the poisson solver or given by the analytical formula.
///
template <typename DataT = double>
class NumericalFields
{
  using RegularGrid = o2::tpc::RegularGrid3D<DataT>;
  using DataContainer = o2::tpc::DataContainer3D<DataT>;
  using TriCubic = o2::tpc::TriCubicInterpolator<DataT>;

 public:
  /// constructor
  /// \param dataEr container for the data of the electrical field Er
  /// \param dataEz container for the data of the electrical field Ez
  /// \param dataEphi container for the data of the electrical field Ephi
  /// \param gridProperties properties of the grid
  /// \param side side of the tpc
  NumericalFields(const DataContainer& dataEr, const DataContainer& dataEz, const DataContainer& dataEphi, const RegularGrid& gridProperties, const o2::tpc::Side side) : mDataEr{dataEr}, mDataEz{dataEz}, mDataEphi{dataEphi}, mGridProperties{gridProperties}, mSide{side} {};

  /// \param r r coordinate
  /// \param phi phi coordinate
  /// \param z z coordinate
  /// \return returns the function value for electric field Er for given coordinate
  DataT evalEr(DataT z, DataT r, DataT phi) const { return mInterpolatorEr(z, r, phi); }

  /// \param r r coordinate
  /// \param phi phi coordinate
  /// \param z z coordinate
  /// \return returns the function value for electric field Ez for given coordinate
  DataT evalEz(DataT z, DataT r, DataT phi) const { return mInterpolatorEz(z, r, phi); }

  /// \param r r coordinate
  /// \param phi phi coordinate
  /// \param z z coordinate
  /// \return returns the function value for electric field Ephi for given coordinate
  DataT evalEphi(DataT z, DataT r, DataT phi) const { return mInterpolatorEphi(z, r, phi); }

  o2::tpc::Side getSide() const { return mSide; }

  static constexpr unsigned int getID() { return ID; }

 private:
  const DataContainer& mDataEr{};                         ///< adress to the data container of the grid
  const DataContainer& mDataEz{};                         ///< adress to the data container of the grid
  const DataContainer& mDataEphi{};                       ///< adress to the data container of the grid
  const RegularGrid& mGridProperties{};                   ///< properties of the regular grid
  const o2::tpc::Side mSide{};                            ///< side of the TPC
  TriCubic mInterpolatorEr{mDataEr, mGridProperties};     ///< TriCubic interpolator of the electric field Er
  TriCubic mInterpolatorEz{mDataEz, mGridProperties};     ///< TriCubic interpolator of the electric field Ez
  TriCubic mInterpolatorEphi{mDataEphi, mGridProperties}; ///< TriCubic interpolator of the electric field Ephi
  static constexpr unsigned int ID = 1;                   ///< needed to distinguish between the different classes
};

///
/// This class gives tricubic interpolation of the local distortions or corrections.
/// The the local distortions or corrections can be used to calculate the global distortions/corrections.
///
template <typename DataT = double>
class DistCorrInterpolator
{
  using RegularGrid = o2::tpc::RegularGrid3D<DataT>;
  using DataContainer = o2::tpc::DataContainer3D<DataT>;
  using TriCubic = o2::tpc::TriCubicInterpolator<DataT>;

 public:
  /// constructor
  /// \param dataDistCorrdR container for the data of the distortions dR
  /// \param dataDistCorrdZ container for the data of the distortions dZ
  /// \param dataDistCorrdRPhi container for the data of the distortions dPhi
  /// \param gridProperties properties of the grid
  /// \param side side of the tpc
  DistCorrInterpolator(const DataContainer& dataDistCorrdR, const DataContainer& dataDistCorrdZ, const DataContainer& dataDistCorrdRPhi, const RegularGrid& gridProperties, const o2::tpc::Side side) : mDataDistCorrdR{dataDistCorrdR}, mDataDistCorrdZ{dataDistCorrdZ}, mDataDistCorrdRPhi{dataDistCorrdRPhi}, mGridProperties{gridProperties}, mSide{side} {};

  /// \param r r coordinate
  /// \param phi phi coordinate
  /// \param z z coordinate
  /// \return returns the function value for the local distortion or correction dR for given coordinate
  DataT evaldR(const DataT z, const DataT r, const DataT phi) const { return interpolatorDistCorrdR(z, r, phi); }

  /// \param r r coordinate
  /// \param phi phi coordinate
  /// \param z z coordinate
  /// \return returns the function value for the local distortion or correction dZ for given coordinate
  DataT evaldZ(const DataT z, const DataT r, const DataT phi) const { return interpolatorDistCorrdZ(z, r, phi); }

  /// \param r r coordinate
  /// \param phi phi coordinate
  /// \param z z coordinate
  /// \return returns the function value for the local distortion or correction dRPhi for given coordinate
  DataT evaldRPhi(const DataT z, const DataT r, const DataT phi) const { return interpolatorDistCorrdRPhi(z, r, phi); }

  o2::tpc::Side getSide() const { return mSide; }

  static constexpr unsigned int getID() { return ID; }

 private:
  const DataContainer& mDataDistCorrdR{};                                  ///< adress to the data container of the grid
  const DataContainer& mDataDistCorrdZ{};                                  ///< adress to the data container of the grid
  const DataContainer& mDataDistCorrdRPhi{};                               ///< adress to the data container of the grid
  const RegularGrid& mGridProperties{};                                    ///< properties of the regular grid
  const o2::tpc::Side mSide{};                                             ///< side of the TPC.
  TriCubic interpolatorDistCorrdR{mDataDistCorrdR, mGridProperties};       ///< TriCubic interpolator of distortion or correction dR
  TriCubic interpolatorDistCorrdZ{mDataDistCorrdZ, mGridProperties};       ///< TriCubic interpolator of distortion or correction dZ
  TriCubic interpolatorDistCorrdRPhi{mDataDistCorrdRPhi, mGridProperties}; ///< TriCubic interpolator of distortion or correction dRPhi
  static constexpr unsigned int ID = 2;                                    ///< needed to distinguish between the different classes
};

} // namespace tpc
} // namespace o2

#endif
