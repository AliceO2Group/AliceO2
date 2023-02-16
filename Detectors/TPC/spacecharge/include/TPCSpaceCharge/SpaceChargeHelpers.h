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
#include "TFormula.h"

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
  DataT evalFieldR(DataT z, DataT r, DataT phi) const { return mErFunc(z, r, phi); }

  /// \param r r coordinate
  /// \param phi phi coordinate
  /// \param z z coordinate
  /// \return returns the function value for electric field Ez for given coordinate
  DataT evalFieldZ(DataT z, DataT r, DataT phi) const { return mEzFunc(z, r, phi); }

  /// \param r r coordinate
  /// \param phi phi coordinate
  /// \param z z coordinate
  /// \return returns the function value for electric field Ephi for given coordinate
  DataT evalFieldPhi(DataT z, DataT r, DataT phi) const { return mEphiFunc(z, r, phi); }

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

/// struct for containing simple analytical distortions (as function of local coordinates) and the resulting corrections
template <typename DataT = double>
class AnalyticalDistCorr
{
 public:
  DataT getDistortionsLX(const DataT lx, const DataT ly, const DataT lz, const Side side) const { return mDlXFormula.Eval(lx, ly, lz, side); };
  DataT getCorrectionsLX(const DataT lx, const DataT ly, const DataT lz, const Side side) const { return mClXFormula.Eval(lx, ly, lz, side); };
  DataT getDistortionsLY(const DataT lx, const DataT ly, const DataT lz, const Side side) const { return mDlYFormula.Eval(lx, ly, lz, side); };
  DataT getCorrectionsLY(const DataT lx, const DataT ly, const DataT lz, const Side side) const { return mClYFormula.Eval(lx, ly, lz, side); };
  DataT getDistortionsLZ(const DataT lx, const DataT ly, const DataT lz, const Side side) const { return mDlZFormula.Eval(lx, ly, lz, side); };
  DataT getCorrectionsLZ(const DataT lx, const DataT ly, const DataT lz, const Side side) const { return mClZFormula.Eval(lx, ly, lz, side); };

  /// set default analytical formulas for distortions and corrections
  void initDefault()
  {
    mDlXFormula = TFormula{"mDlX", "(254.5 - x) / 50"};                              ///< analytical distortions in lx as a function of lx
    mClXFormula = TFormula{"mClX", "(x * 50 - 254.5) / 49 - x"};                     ///< analytical corrections in lx as a function of lx
    mDlYFormula = TFormula{"mDlY", "2 + 0.01 * x"};                                  ///< analytical distortions in ly as a function of lx
    mClYFormula = TFormula{"mClY", "-(2 + 0.01 * (x + (x * 50 - 254.5) / 49 - x))"}; ///< analytical correction in ly as a function of lx
    mDlZFormula = TFormula{"mDlZ", "z / 50"};                                        ///< analytical correction in lz as a function of lz
    mClZFormula = TFormula{"mClZ", "z * 50 / 51 - z"};                               ///< analytical correction in lz as a function of lz
  }

  /// check if all formulas are valid
  bool isValid() const { return mDlXFormula.IsValid() && mClXFormula.IsValid() && mDlYFormula.IsValid() && mClYFormula.IsValid() && mDlZFormula.IsValid() && mClZFormula.IsValid(); }

  ///  const DataT dlX = (TPCParameters<DataT>::OFCRADIUS - lx) / 50; // (171 -> 0) / 50 = 3.42 cn
  ///  return dlX;
  TFormula mDlXFormula{}; ///< analytical distortions in lx as a function of lx

  ///  analytical correction in lx as a function of lx
  ///  distorted point: lx_2 = lx_1 + mDlX(lx_1)
  ///                   lx_2 = lx_1 + (TPCParameters<DataT>::OFCRADIUS - lx) / 50);
  ///                   lx_2 * 50 = lx_1 * 50 + TPCParameters<DataT>::OFCRADIUS - lx
  ///                   lx_2 * 50 - TPCParameters<DataT>::OFCRADIUS = lx_1 * 49
  ///                   lx_2 * 50 - TPCParameters<DataT>::OFCRADIUS / 49 = lx_1
  ///  correction: dCorrX = lx_2 - lx_1
  ///
  ///  const DataT lx2 = (lx * 50 - TPCParameters<DataT>::OFCRADIUS) / 49;
  ///  return lx2 - lx;
  TFormula mClXFormula{}; ///< analytical corrections in lx as a function of lx

  /// const DataT dlY = 2 + 0.01 * lx;
  /// return dlY;
  TFormula mDlYFormula{}; ///< analytical distortions in ly as a function of lx

  /// const DataT dlX_1 = lx + mClX(lx, 0, 0, Side::A); // corrected point (original point without distortion)
  /// return -mDlY(dlX_1, 0, 0, Side::A);               // distortion at original point
  TFormula mClYFormula{}; ///< analytical correction in ly as a function of lx

  /// const DataT dlZ = lz / 50;
  /// return dlZ;
  TFormula mDlZFormula{}; ///< analytical correction in lz as a function of lz

  ///  lz_2 = lz_1 + mDlZ(lz_1)
  ///  lz_2 = lz_1 + lz_1 / 50
  ///  lz_2 = lz_1 * (1 + 1/50)
  ///  lz_2 / (1 + 1/50) = lz_1
  ///  lz_2 * 50 / 51 = lz_1
  ///
  ///  const DataT lz2 = lz * 50 / 51.;
  ///  const DataT diffZ = lz2 - lz;
  ///  return diffZ;
  TFormula mClZFormula{}; ///< analytical correction in lz as a function of lz

  ClassDefNV(AnalyticalDistCorr, 1);
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
  NumericalFields(const DataContainer& dataEr, const DataContainer& dataEz, const DataContainer& dataEphi, const RegularGrid& gridProperties, const o2::tpc::Side side)
    : mInterpolatorEr{dataEr, gridProperties}, mInterpolatorEz{dataEz, gridProperties}, mInterpolatorEphi{dataEphi, gridProperties}, mSide{side} {};

  /// \param r r coordinate
  /// \param phi phi coordinate
  /// \param z z coordinate
  /// \return returns the function value for electric field Er for given coordinate
  DataT evalFieldR(DataT z, DataT r, DataT phi) const { return mInterpolatorEr(z, r, phi); }

  /// \param r r coordinate
  /// \param phi phi coordinate
  /// \param z z coordinate
  /// \return returns the function value for electric field Ez for given coordinate
  DataT evalFieldZ(DataT z, DataT r, DataT phi) const { return mInterpolatorEz(z, r, phi); }

  /// \param r r coordinate
  /// \param phi phi coordinate
  /// \param z z coordinate
  /// \return returns the function value for electric field Ephi for given coordinate
  DataT evalFieldPhi(DataT z, DataT r, DataT phi) const { return mInterpolatorEphi(z, r, phi); }

  o2::tpc::Side getSide() const { return mSide; }

  static constexpr unsigned int getID() { return ID; }

 private:
  o2::tpc::Side mSide{};                ///< side of the TPC
  TriCubic mInterpolatorEr{};           ///< TriCubic interpolator of the electric field Er
  TriCubic mInterpolatorEz{};           ///< TriCubic interpolator of the electric field Ez
  TriCubic mInterpolatorEphi{};         ///< TriCubic interpolator of the electric field Ephi
  static constexpr unsigned int ID = 1; ///< needed to distinguish between the different classes
};

///
/// B Field obtained from fit to chebychev polynomials (ToDo: add other BField settings)
///
class BField
{
  using DataT = double;

 public:
  /// \param r r coordinate
  /// \param phi phi coordinate
  /// \param z z coordinate
  /// \return returns the function value for B field Br for given coordinate
  DataT evalFieldR(DataT z, DataT r, DataT phi) const
  {
    const double rphiz[]{r, phi, z};
    return getBR(rphiz);
  }

  /// \param r r coordinate
  /// \param phi phi coordinate
  /// \param z z coordinate
  /// \return returns the function value for B field Bz for given coordinate
  DataT evalFieldZ(DataT z, DataT r, DataT phi) const
  {
    const double rphiz[]{r, phi, z};
    return getBZ(rphiz);
  }

  /// \param r r coordinate
  /// \param phi phi coordinate
  /// \param z z coordinate
  /// \return returns the function value for B field Bphi for given coordinate
  DataT evalFieldPhi(DataT z, DataT r, DataT phi) const
  {
    const double rphiz[]{r, phi, z};
    return getBPhi(rphiz);
  }

  static double getBR(const double rphiz[]) { return (rphiz[2] >= 0) ? getBR_A(rphiz) : getBR_C(rphiz); };
  static double getBPhi(const double rphiz[]) { return (rphiz[2] >= 0) ? getBPhi_A(rphiz) : getBPhi_C(rphiz); };
  static double getBZ(const double rphiz[]) { return (rphiz[2] >= 0) ? getBZ_A(rphiz) : getBZ_C(rphiz); };
  static double getBR_A(const double rphiz[]) { return mParamsBR_A[13] + mParamsBR_A[0] * rphiz[0] + mParamsBR_A[1] * rphiz[2] + mParamsBR_A[2] * rphiz[0] * rphiz[2] + mParamsBR_A[3] * rphiz[2] * rphiz[2] + mParamsBR_A[4] * rphiz[0] * rphiz[2] * rphiz[2] + mParamsBR_A[5] * rphiz[0] * std::cos(rphiz[1]) + mParamsBR_A[6] * rphiz[0] * std::sin(rphiz[1]) + mParamsBR_A[7] * rphiz[2] * rphiz[2] * std::cos(rphiz[1]) + mParamsBR_A[8] * rphiz[2] * rphiz[2] * std::sin(rphiz[1]) + mParamsBR_A[9] * rphiz[0] * rphiz[2] * rphiz[2] * std::cos(rphiz[1]) + mParamsBR_A[10] * rphiz[0] * rphiz[2] * rphiz[2] * std::sin(rphiz[1]) + mParamsBR_A[11] * rphiz[0] * rphiz[2] * std::cos(rphiz[1]) + mParamsBR_A[12] * rphiz[0] * rphiz[2] * std::sin(rphiz[1]); };
  static double getBR_C(const double rphiz[]) { return mParamsBR_C[13] + mParamsBR_C[0] * rphiz[0] + mParamsBR_C[1] * rphiz[2] + mParamsBR_C[2] * rphiz[0] * rphiz[2] + mParamsBR_C[3] * rphiz[2] * rphiz[2] + mParamsBR_C[4] * rphiz[0] * rphiz[2] * rphiz[2] + mParamsBR_C[5] * rphiz[0] * std::cos(rphiz[1]) + mParamsBR_C[6] * rphiz[0] * std::sin(rphiz[1]) + mParamsBR_C[7] * rphiz[2] * rphiz[2] * std::cos(rphiz[1]) + mParamsBR_C[8] * rphiz[2] * rphiz[2] * std::sin(rphiz[1]) + mParamsBR_C[9] * rphiz[0] * rphiz[2] * rphiz[2] * std::cos(rphiz[1]) + mParamsBR_C[10] * rphiz[0] * rphiz[2] * rphiz[2] * std::sin(rphiz[1]) + mParamsBR_C[11] * rphiz[0] * rphiz[2] * std::cos(rphiz[1]) + mParamsBR_C[12] * rphiz[0] * rphiz[2] * std::sin(rphiz[1]); };
  static double getBPhi_A(const double rphiz[]) { return mParamsBPhi_A[13] + mParamsBPhi_A[0] * rphiz[0] + mParamsBPhi_A[1] * rphiz[2] + mParamsBPhi_A[2] * rphiz[0] * rphiz[2] + mParamsBPhi_A[3] * rphiz[2] * rphiz[2] + mParamsBPhi_A[4] * rphiz[0] * rphiz[2] * rphiz[2] + mParamsBPhi_A[5] * rphiz[0] * std::cos(rphiz[1]) + mParamsBPhi_A[6] * rphiz[0] * std::sin(rphiz[1]) + mParamsBPhi_A[7] * rphiz[2] * rphiz[2] * std::cos(rphiz[1]) + mParamsBPhi_A[8] * rphiz[2] * rphiz[2] * std::sin(rphiz[1]) + mParamsBPhi_A[9] * rphiz[0] * rphiz[2] * rphiz[2] * std::cos(rphiz[1]) + mParamsBPhi_A[10] * rphiz[0] * rphiz[2] * rphiz[2] * std::sin(rphiz[1]) + mParamsBPhi_A[11] * rphiz[0] * rphiz[2] * std::cos(rphiz[1]) + mParamsBPhi_A[12] * rphiz[0] * rphiz[2] * std::sin(rphiz[1]); };
  static double getBPhi_C(const double rphiz[]) { return mParamsBPhi_C[13] + mParamsBPhi_C[0] * rphiz[0] + mParamsBPhi_C[1] * rphiz[2] + mParamsBPhi_C[2] * rphiz[0] * rphiz[2] + mParamsBPhi_C[3] * rphiz[2] * rphiz[2] + mParamsBPhi_C[4] * rphiz[0] * rphiz[2] * rphiz[2] + mParamsBPhi_C[5] * rphiz[0] * std::cos(rphiz[1]) + mParamsBPhi_C[6] * rphiz[0] * std::sin(rphiz[1]) + mParamsBPhi_C[7] * rphiz[2] * rphiz[2] * std::cos(rphiz[1]) + mParamsBPhi_C[8] * rphiz[2] * rphiz[2] * std::sin(rphiz[1]) + mParamsBPhi_C[9] * rphiz[0] * rphiz[2] * rphiz[2] * std::cos(rphiz[1]) + mParamsBPhi_C[10] * rphiz[0] * rphiz[2] * rphiz[2] * std::sin(rphiz[1]) + mParamsBPhi_C[11] * rphiz[0] * rphiz[2] * std::cos(rphiz[1]) + mParamsBPhi_C[12] * rphiz[0] * rphiz[2] * std::sin(rphiz[1]); };
  static double getBZ_A(const double rphiz[]) { return mParamsBZ_A[6] + mParamsBZ_A[0] * rphiz[0] + mParamsBZ_A[1] * rphiz[2] + mParamsBZ_A[2] * rphiz[1] + mParamsBZ_A[3] * rphiz[0] * rphiz[0] + mParamsBZ_A[4] * rphiz[2] * rphiz[2] + mParamsBZ_A[5] * rphiz[1] * rphiz[1]; };
  static double getBZ_C(const double rphiz[]) { return mParamsBZ_C[6] + mParamsBZ_C[0] * rphiz[0] + mParamsBZ_C[1] * rphiz[2] + mParamsBZ_C[2] * rphiz[1] + mParamsBZ_C[3] * rphiz[0] * rphiz[0] + mParamsBZ_C[4] * rphiz[2] * rphiz[2] + mParamsBZ_C[5] * rphiz[1] * rphiz[1]; };

  static constexpr double mParamsBR_A[]{-2.735308458415022e-06, 3.332307825230892e-05, -1.6122043674923547e-06, -3.651355880554624e-07, 1.279249264081895e-09, 8.022905486012087e-06, -9.860444359905876e-07, 3.731008518454023e-08, -1.621170030862478e-07, -2.993099518447553e-10, 9.188552543587662e-10, 3.694763794980658e-08, -2.4521918555825965e-07, -0.0011251001320472243};            ///< parameters for B_r A side
  static constexpr double mParamsBR_C[]{-9.56934067157109e-06, 2.925896354411999e-05, -1.590504175365935e-06, 3.2678506747823123e-07, -1.155443633847809e-09, 8.047221940176635e-06, -1.524233769981198e-06, -2.058042110382768e-07, 1.7666032683026417e-07, 8.66636087440012e-10, -9.704495551802566e-10, 3.212813408161466e-08, -2.4861803070141444e-07, 0.0008591129655999633};              ///< parameters for B_r C side
  static constexpr double mParamsBPhi_A[]{2.4816698646856386e-07, -3.3769029760269716e-07, -1.2683802228879448e-09, 2.3512494546822587e-09, -4.424558185666922e-13, -7.894179812888077e-07, -3.839830209758884e-06, -1.7904399279931762e-07, -4.412987384727642e-08, 1.0387899089797522e-09, 3.3464750104626054e-10, -2.2404404898678082e-07, -5.148774856850897e-08, -1.1983526589792469e-05}; ///< parameters for B_phi A side
  static constexpr double mParamsBPhi_C[]{5.043186514423357e-07, 1.8108880196737116e-07, -1.3759428693116512e-09, 3.5765707078538657e-09, -2.0523476064320596e-11, -6.579691696988604e-07, -3.0122693118808835e-06, 1.9271170150103e-07, 1.753682204150865e-07, -1.0480263051890858e-09, -4.509685788998614e-10, -2.2662983377275664e-07, -3.321254466726585e-08, -9.824193801152964e-05};      ///< parameters for B_phi C side
  static constexpr double mParamsBZ_A[]{-8.491591204045067e-05, 1.5584623849211725e-05, -0.0020520451709635274, -5.867431435165632e-07, 1.4724704039112152e-06, -0.00022130669269254145, -4.997232421100266};                                                                                                                                                                                   ///< parameters for B_z A side
  static constexpr double mParamsBZ_C[]{-9.422282497783464e-05, 9.827032671750074e-06, -0.002219129216064967, -5.520605034115637e-07, 1.4618205680952e-06, -0.0012705559037709936, -4.993429241196326};                                                                                                                                                                                         ///< parameters for B_z A side
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
  DistCorrInterpolator(const DataContainer& dataDistCorrdR, const DataContainer& dataDistCorrdZ, const DataContainer& dataDistCorrdRPhi, const RegularGrid& gridProperties, const o2::tpc::Side side)
    : interpolatorDistCorrdR{dataDistCorrdR, gridProperties}, interpolatorDistCorrdZ{dataDistCorrdZ, gridProperties}, interpolatorDistCorrdRPhi{dataDistCorrdRPhi, gridProperties}, mSide{side} {};

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
  o2::tpc::Side mSide{};                ///< side of the TPC.
  TriCubic interpolatorDistCorrdR{};    ///< TriCubic interpolator of distortion or correction dR
  TriCubic interpolatorDistCorrdZ{};    ///< TriCubic interpolator of distortion or correction dZ
  TriCubic interpolatorDistCorrdRPhi{}; ///< TriCubic interpolator of distortion or correction dRPhi
  static constexpr unsigned int ID = 2; ///< needed to distinguish between the different classes
};

} // namespace tpc
} // namespace o2

#endif
