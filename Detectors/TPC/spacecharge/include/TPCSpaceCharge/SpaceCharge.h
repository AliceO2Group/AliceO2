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

/// \file  SpaceCharge.h
/// \brief This class contains the algorithms for calculation the distortions and corrections
///
/// \author  Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
///          Rifki Sadikin <rifki.sadikin@cern.ch> (original code in AliRoot in AliTPCSpaceCharge3DCalc.h)
/// \date Aug 21, 2020

#ifndef ALICEO2_TPC_SPACECHARGE_H_
#define ALICEO2_TPC_SPACECHARGE_H_

#include "TPCSpaceCharge/TriCubic.h"
#include "TPCSpaceCharge/SpaceChargeHelpers.h"
#include "TPCSpaceCharge/PoissonSolverHelpers.h"
#include "TPCSpaceCharge/RegularGrid3D.h"
#include "TPCSpaceCharge/DataContainer3D.h"
#include "TPCSpaceCharge/SpaceChargeParameter.h"
#include "DataFormatsTPC/Defs.h"
#include "CommonUtils/DebugStreamer.h"

class TH3;
class TH3D;
class TH3F;
class TH2F;

namespace o2
{
namespace tpc
{

template <class T>
class CalDet;

/// \class SpaceCharge
/// this class provides the algorithms for calculating the global distortions and corrections from the space charge density.
/// The calculations can be done by a realistic space charge histogram as an input or by an analytical formula.
/// An example of of the usage can be found in 'macro/calculateDistortionsCorrections.C'

/// \tparam DataT the data type which is used during the calculations
/// \tparam Nz number of vertices in z direction
/// \tparam Nr number of vertices in r direction
/// \tparam Nphi number of vertices in phi direction
template <typename DataT = double>
class SpaceCharge
{
  using RegularGrid = RegularGrid3D<DataT>;
  using DataContainer = DataContainer3D<DataT>;
  using GridProp = GridProperties<DataT>;
  using TriCubic = TriCubicInterpolator<DataT>;
  using TH3DataT = std::conditional_t<std::is_same<DataT, double>::value, TH3D, TH3F>; // datatype for TH3 (TH3F for DataT==float and TH3D for DataT==double)

 public:
  /// default constructor
  /// grid granularity has to set before constructing an object using the static function setGrid(nZVertices, nRVertices, nPhiVertices)!
  /// \param omegaTau \omega \tau value
  /// \param t1 value for t1
  /// \param t2 value for t2
  SpaceCharge(const DataT omegaTau = 0.32f, const DataT t1 = 1, const DataT t2 = 1);

  /// \param nZVertices number of vertices of the grid in z direction
  /// \param nRVertices number of vertices of the grid in z direction
  /// \param nPhiVertices number of vertices of the grid in z direction
  static void setGrid(const unsigned short nZVertices, const unsigned short nRVertices, const unsigned short nPhiVertices);

  /// Enumerator for setting the space-charge distortion mode
  enum class SCDistortionType : int {
    SCDistortionsConstant = 0, // space-charge distortions constant over time
    SCDistortionsRealistic = 1 // realistic evolution of space-charge distortions over time
  };

  /// numerical integration strategys
  enum class IntegrationStrategy { Trapezoidal = 0,     ///< trapezoidal integration (https://en.wikipedia.org/wiki/Trapezoidal_rule). straight electron drift line assumed: z0->z1, r0->r0, phi0->phi0
                                   Simpson = 1,         ///< simpon integration. see: https://en.wikipedia.org/wiki/Simpson%27s_rule. straight electron drift line assumed: z0->z1, r0->r0, phi0->phi0
                                   Root = 2,            ///< Root integration. straight electron drift line assumed: z0->z1, r0->r0, phi0->phi0
                                   SimpsonIterative = 3 ///< simpon integration, but using an iterative method to approximate the drift path. No straight electron drift line assumed: z0->z1, r0->r1, phi0->phi1
  };

  enum class Type {
    Distortions = 0, ///< distortions
    Corrections = 1  ///< corrections
  };

  enum class GlobalDistType {
    Standard = 0, ///< classical method (start calculation of global distortion at each voxel in the tpc and follow electron drift to readout -slow-)
    Fast = 1,     ///< interpolation of global corrections (use the global corrections to apply an iterative approach to obtain the global distortions -fast-)
    None = 2      ///< dont calculate global distortions
  };

  enum class GlobalDistCorrMethod {
    LocalDistCorr,  ///< using local dis/corr interpolator for calculation of global distortions/corrections
    ElectricalField ///< using electric field for calculation of global distortions/corrections
  };

  /// step 0: set the charge density from TH3 histogram containing the space charge density
  /// \param hisSCDensity3D histogram for the space charge density
  void fillChargeDensityFromHisto(const TH3& hisSCDensity3D);

  /// step 0: set the space charge density from std::vector<CalDet> containing the space charge density. Each entry in the object corresponds to one z slice
  /// \param calSCDensity3D histogram for the space charge density
  void fillChargeDensityFromCalDet(const std::vector<CalDet<float>>& calSCDensity3D);

  /// step 0: set the charge (number of ions) from std::vector<CalDet> containing the charge. Each entry in the object corresponds to one z slice.
  /// Normalization to the space charge is also done automatically
  /// \param calCharge3D histogram for the charge
  void fillChargeFromCalDet(const std::vector<CalDet<float>>& calCharge3D);

  /// step 0: set the charge density from TH3 histogram containing the space charge density
  /// \param fInp input file containing a histogram for the space charge density
  /// \param name the name of the space charge density histogram in the file
  void fillChargeDensityFromFile(TFile& fInp, const char* name);

  /// \param side side of the TPC
  /// \param calcVectors set to calculate also the local distortion and local correction vectors
  void calculateDistortionsCorrections(const o2::tpc::Side side, const bool calcVectors = false);

  /// step 0: this function fills the internal storage for the charge density using an analytical formula
  /// \param formulaStruct struct containing a method to evaluate the density
  void setChargeDensityFromFormula(const AnalyticalFields<DataT>& formulaStruct);

  /// step 0: this function fills the boundary of the potential using an analytical formula. The boundary is used in the PoissonSolver.
  /// \param formulaStruct struct containing a method to evaluate the potential
  void setPotentialBoundaryFromFormula(const AnalyticalFields<DataT>& formulaStruct);

  /// step 0: this function fills the potential using an analytical formula
  /// \param formulaStruct struct containing a method to evaluate the potential
  void setPotentialFromFormula(const AnalyticalFields<DataT>& formulaStruct);

  /// setting default potential (same potential for all GEM frames. The default value of 1000V are matched to distortions observed in laser data without X-Ray etc.
  /// \param side side of the TPC where the potential will be set
  /// \param deltaPotential delta potential which will be set at the GEM frames
  void setDefaultStaticDistortionsGEMFrameChargeUp(const Side side, const DataT deltaPotential = 1000);

  /// setting the boundary potential of the GEM stack along the radius
  /// \param potentialFunc potential funtion as a function of the radius
  /// \side Side of the TPC
  void setPotentialBoundaryGEMFrameAlongR(const std::function<DataT(DataT)>& potentialFunc, const Side side);

  /// setting the boundary potential of the IROC on the bottom along phi
  /// \param potentialFunc potential funtion as a function of global phi
  /// \side Side of the TPC
  void setPotentialBoundaryGEMFrameIROCBottomAlongPhi(const std::function<DataT(DataT)>& potentialFunc, const Side side) { setPotentialBoundaryGEMFrameAlongPhi(potentialFunc, GEMstack::IROCgem, true, side); }

  /// setting the boundary potential of the IROC on the top along phi
  /// \param potentialFunc potential funtion as a function of global phi
  /// \side Side of the TPC
  void setPotentialBoundaryGEMFrameIROCTopAlongPhi(const std::function<DataT(DataT)>& potentialFunc, const Side side) { setPotentialBoundaryGEMFrameAlongPhi(potentialFunc, GEMstack::IROCgem, false, side); }

  /// setting the boundary potential of the OROC1 on the bottom along phi
  /// \param potentialFunc potential funtion as a function of global phi
  /// \side Side of the TPC
  void setPotentialBoundaryGEMFrameOROC1BottomAlongPhi(const std::function<DataT(DataT)>& potentialFunc, const Side side) { setPotentialBoundaryGEMFrameAlongPhi(potentialFunc, GEMstack::OROC1gem, true, side); }

  /// setting the boundary potential of the OROC1 on the top along phi
  /// \param potentialFunc potential funtion as a function of global phi
  /// \side Side of the TPC
  void setPotentialBoundaryGEMFrameOROC1TopAlongPhi(const std::function<DataT(DataT)>& potentialFunc, const Side side) { setPotentialBoundaryGEMFrameAlongPhi(potentialFunc, GEMstack::OROC1gem, false, side); }

  /// setting the boundary potential of the OROC2 on the bottom along phi
  /// \param potentialFunc potential funtion as a function of global phi
  /// \side Side of the TPC
  void setPotentialBoundaryGEMFrameOROC2BottomAlongPhi(const std::function<DataT(DataT)>& potentialFunc, const Side side) { setPotentialBoundaryGEMFrameAlongPhi(potentialFunc, GEMstack::OROC2gem, true, side); }

  /// setting the boundary potential of the OROC2 on the top along phi
  /// \param potentialFunc potential funtion as a function of global phi
  /// \side Side of the TPC
  void setPotentialBoundaryGEMFrameOROC2TopAlongPhi(const std::function<DataT(DataT)>& potentialFunc, const Side side) { setPotentialBoundaryGEMFrameAlongPhi(potentialFunc, GEMstack::OROC2gem, false, side); }

  /// setting the boundary potential of the OROC3 on the bottom along phi
  /// \param potentialFunc potential funtion as a function of global phi
  /// \side Side of the TPC
  void setPotentialBoundaryGEMFrameOROC3BottomAlongPhi(const std::function<DataT(DataT)>& potentialFunc, const Side side) { setPotentialBoundaryGEMFrameAlongPhi(potentialFunc, GEMstack::OROC3gem, true, side); }

  /// setting the boundary potential of the OROC3 on the top along phi
  /// \param potentialFunc potential funtion as a function of global phi
  /// \side Side of the TPC
  void setPotentialBoundaryGEMFrameOROC3TopAlongPhi(const std::function<DataT(DataT)>& potentialFunc, const Side side) { setPotentialBoundaryGEMFrameAlongPhi(potentialFunc, GEMstack::OROC3gem, false, side); }

  /// setting the boundary potential of the OROC3 on the top along phi
  /// \param potentialFunc potential funtion as a function of global phi
  /// \side Side of the TPC
  void setPotentialBoundaryGEMFrameOROC3ToOFCPhi(const std::function<DataT(DataT)>& potentialFunc, const Side side) { setPotentialBoundaryGEMFrameAlongPhi(potentialFunc, GEMstack::OROC3gem, false, side, true); }

  /// setting the boundary potential for the inner TPC radius along r
  /// \param potentialFunc potential funtion as a function of z
  /// \side Side of the TPC
  void setPotentialBoundaryInnerRadius(const std::function<DataT(DataT)>& potentialFunc, const Side side);

  /// setting the boundary potential for the outer TPC radius along r
  /// \param potentialFunc potential funtion as a function of z
  /// \side Side of the TPC
  void setPotentialBoundaryOuterRadius(const std::function<DataT(DataT)>& potentialFunc, const Side side);

  /// step 1: use the O2TPCPoissonSolver class to numerically calculate the potential with set space charge density and boundary conditions from potential
  /// \param side side of the TPC
  /// \param stoppingConvergence stopping criterion used in the poisson solver
  /// \param symmetry use symmetry or not in the poisson solver
  void poissonSolver(const Side side, const DataT stoppingConvergence = 1e-6, const int symmetry = 0);

  /// step 2: calculate numerically the electric field from the potential
  /// \param side side of the TPC
  void calcEField(const Side side);

  /// step 2a: set the electric field from an analytical formula
  /// \param formulaStruct struct containing a method to evaluate the electric fields
  void setEFieldFromFormula(const AnalyticalFields<DataT>& formulaStruct);

  /// scale the space charge density by a scaling factor: sc_new = sc * scalingFactor
  /// \param scalingFactor factor to scale the space-charge density
  /// \param side side for which the space-charge density will be scaled
  void scaleChargeDensity(const DataT scalingFactor, const Side side) { mDensity[side] *= scalingFactor; }

  /// step 3: calculate the local distortions and corrections with an electric field
  /// \param type calculate local corrections or local distortions: type = o2::tpc::SpaceCharge<>::Type::Distortions or o2::tpc::SpaceCharge<>::Type::Corrections
  /// \param formulaStruct struct containing a method to evaluate the electric field Er, Ez, Ephi (analytical formula or by TriCubic interpolator)
  template <typename ElectricFields = AnalyticalFields<DataT>>
  void calcLocalDistortionsCorrections(const Type type, const ElectricFields& formulaStruct);

  /// step 3b: calculate the local distortion and correction vectors with an electric field
  /// \param formulaStruct struct containing a method to evaluate the electric field Er, Ez, Ephi (analytical formula or by TriCubic interpolator)
  template <typename ElectricFields = AnalyticalFields<DataT>>
  void calcLocalDistortionCorrectionVector(const ElectricFields& formulaStruct);

  /// step 3b: calculate the local distortions and corrections with the local distortion/correction vectors using Runge Kutta 4.
  /// calcLocalDistortionCorrectionVector() has to be called before this function
  /// \param type calculate local corrections or local distortions: type = o2::tpc::SpaceCharge<>::Type::Distortions or o2::tpc::SpaceCharge<>::Type::Corrections
  /// \param side side of the TPC
  template <typename ElectricFields = AnalyticalFields<DataT>>
  void calcLocalDistortionsCorrectionsRK4(const Type type, const Side side);

  /// step 4: calculate global corrections by using the electric field or the local corrections
  /// \param formulaStruct struct containing a method to evaluate the electric field Er, Ez, Ephi or the local corrections
  template <typename Fields = AnalyticalFields<DataT>>
  void calcGlobalCorrections(const Fields& formulaStruct);

  /// step 5: calculate global distortions by using the electric field or the local distortions (SLOW)
  /// \param formulaStruct struct containing a method to evaluate the electric field Er, Ez, Ephi or the local distortions
  /// \param maxIterations maximum steps which are are performed to reach the central electrode (in general this is not necessary, but in case of problems this value aborts the calculation)
  template <typename Fields = AnalyticalFields<DataT>>
  void calcGlobalDistortions(const Fields& formulaStruct, const int maxIterations = 3 * sSteps * mParamGrid.NZVertices);

  void init();

  /// step 5: calculate global distortions using the global corrections (FAST)
  /// \param globCorr interpolator for global corrections
  /// \param maxIter maximum iterations per global distortion
  /// \param approachZ when the difference between the desired z coordinate and the position of the global correction is deltaZ, approach the desired z coordinate by deltaZ * \p approachZ.
  /// \param approachR when the difference between the desired r coordinate and the position of the global correction is deltaR, approach the desired r coordinate by deltaR * \p approachR.
  /// \param approachPhi when the difference between the desired phi coordinate and the position of the global correction is deltaPhi, approach the desired phi coordinate by deltaPhi * \p approachPhi.
  /// \param diffCorr if the absolute differences from the interpolated values for the global corrections from the last iteration compared to the current iteration is smaller than this value, set converged to true for current global distortion
  void calcGlobalDistWithGlobalCorrIterative(const DistCorrInterpolator<DataT>& globCorr, const int maxIter = 100, const DataT approachZ = 0.5, const DataT approachR = 0.5, const DataT approachPhi = 0.5, const DataT diffCorr = 1e-6);

  /// \return returns number of vertices in z direction
  unsigned short getNZVertices() { return mParamGrid.NZVertices; }

  /// \return returns number of vertices in r direction
  unsigned short getNRVertices() { return mParamGrid.NRVertices; }

  /// \return returns number of vertices in phi direction
  unsigned short getNPhiVertices() { return mParamGrid.NPhiVertices; }

  /// get the space charge density for given coordinate
  /// \param z global z coordinate
  /// \param r global r coordinate
  /// \param phi global phi coordinate
  DataT getDensityCyl(const DataT z, const DataT r, const DataT phi, const Side side) const;

  /// get the potential for given coordinate
  /// \param z global z coordinate
  /// \param r global r coordinate
  /// \param phi global phi coordinate
  DataT getPotentialCyl(const DataT z, const DataT r, const DataT phi, const Side side) const;

  /// get the electric field for given coordinate
  /// \param z global z coordinate
  /// \param r global r coordinate
  /// \param phi global phi coordinate
  /// \param eZ returns correction in z direction
  /// \param eR returns correction in r direction
  /// \param ePhi returns correction in phi direction
  void getElectricFieldsCyl(const DataT z, const DataT r, const DataT phi, const Side side, DataT& eZ, DataT& eR, DataT& ePhi) const;

  /// get the local correction for given coordinate
  /// \param z global z coordinate
  /// \param r global r coordinate
  /// \param phi global phi coordinate
  /// \param lcorrZ returns local correction in z direction
  /// \param lcorrR returns local correction in r direction
  /// \param lcorrRPhi returns local correction in rphi direction
  void getLocalCorrectionsCyl(const DataT z, const DataT r, const DataT phi, const Side side, DataT& lcorrZ, DataT& lcorrR, DataT& lcorrRPhi) const;

  /// get the local correction for given coordinates
  /// \param z global z coordinates
  /// \param r global r coordinates
  /// \param phi global phi coordinates
  /// \param lcorrZ returns local corrections in z direction
  /// \param lcorrR returns local corrections in r direction
  /// \param lcorrRPhi returns local corrections in rphi direction
  void getLocalCorrectionsCyl(const std::vector<DataT>& z, const std::vector<DataT>& r, const std::vector<DataT>& phi, const Side side, std::vector<DataT>& lcorrZ, std::vector<DataT>& lcorrR, std::vector<DataT>& lcorrRPhi) const;

  /// get the global correction for given coordinate
  /// \param z global z coordinate
  /// \param r global r coordinate
  /// \param phi global phi coordinate
  /// \param corrZ returns correction in z direction
  /// \param corrR returns correction in r direction
  /// \param corrRPhi returns correction in rphi direction
  void getCorrectionsCyl(const DataT z, const DataT r, const DataT phi, const Side side, DataT& corrZ, DataT& corrR, DataT& corrRPhi) const;

  /// get the global correction for given coordinates
  /// \param z global z coordinates
  /// \param r global r coordinates
  /// \param phi global phi coordinates
  /// \param corrZ returns corrections in z direction
  /// \param corrR returns corrections in r direction
  /// \param corrRPhi returns corrections in rphi direction
  void getCorrectionsCyl(const std::vector<DataT>& z, const std::vector<DataT>& r, const std::vector<DataT>& phi, const Side side, std::vector<DataT>& corrZ, std::vector<DataT>& corrR, std::vector<DataT>& corrRPhi) const;

  /// get the global corrections for given coordinate
  /// \param x global x coordinate
  /// \param y global y coordinate
  /// \param z global z coordinate
  /// \param corrX returns corrections in x direction
  /// \param corrY returns corrections in y direction
  /// \param corrZ returns corrections in z direction
  void getCorrections(const DataT x, const DataT y, const DataT z, const Side side, DataT& corrX, DataT& corrY, DataT& corrZ) const;

  /// get the analytical global corrections for given coordinate
  /// \param x global x coordinate
  /// \param y global y coordinate
  /// \param z global z coordinate
  /// \param corrX returns corrections in x direction
  /// \param corrY returns corrections in y direction
  /// \param corrZ returns corrections in z direction
  void getCorrectionsAnalytical(const DataT x, const DataT y, const DataT z, const Side side, DataT& corrX, DataT& corrY, DataT& corrZ) const { getDistortionsCorrectionsAnalytical(x, y, z, side, corrX, corrY, corrZ, false); }

  /// get the local distortions for given coordinate
  /// \param z global z coordinate
  /// \param r global r coordinate
  /// \param phi global phi coordinate
  /// \param ldistZ returns local distortion in z direction
  /// \param ldistR returns local distortion in r direction
  /// \param ldistRPhi returns local distortion in rphi direction
  void getLocalDistortionsCyl(const DataT z, const DataT r, const DataT phi, const Side side, DataT& ldistZ, DataT& ldistR, DataT& ldistRPhi) const;

  /// get the local distortions for given coordinates
  /// \param z global z coordinates
  /// \param r global r coordinates
  /// \param phi global phi coordinates
  /// \param ldistZ returns local distortions in z direction
  /// \param ldistR returns local distortions in r direction
  /// \param ldistRPhi returns local distortions in rphi direction
  void getLocalDistortionsCyl(const std::vector<DataT>& z, const std::vector<DataT>& r, const std::vector<DataT>& phi, const Side side, std::vector<DataT>& ldistZ, std::vector<DataT>& ldistR, std::vector<DataT>& ldistRPhi) const;

  /// get the local distortion vector for given coordinates
  /// \param z global z coordinate
  /// \param r global r coordinate
  /// \param phi global phi coordinate
  /// \param lvecdistZ returns local distortion vector in z direction
  /// \param lvecdistR returns local distortion vector in r direction
  /// \param lvecdistRPhi returns local distortion vector in rphi direction
  void getLocalDistortionVectorCyl(const DataT z, const DataT r, const DataT phi, const Side side, DataT& lvecdistZ, DataT& lvecdistR, DataT& lvecdistRPhi) const;

  /// get the local distortion vector for given coordinate
  /// \param z global z coordinates
  /// \param r global r coordinates
  /// \param phi global phi coordinates
  /// \param lvecdistZ returns local distortion vectors in z direction
  /// \param lvecdistR returns local distortion vectors in r direction
  /// \param lvecdistRPhi returns local distortion vectors in rphi direction
  void getLocalDistortionVectorCyl(const std::vector<DataT>& z, const std::vector<DataT>& r, const std::vector<DataT>& phi, const Side side, std::vector<DataT>& lvecdistZ, std::vector<DataT>& lvecdistR, std::vector<DataT>& lvecdistRPhi) const;

  /// get the local correction vector for given coordinate
  /// \param z global z coordinate
  /// \param r global r coordinate
  /// \param phi global phi coordinate
  /// \param ldistZ returns local correction vector in z direction
  /// \param ldistR returns local correction vector in r direction
  /// \param ldistRPhi returns local correction vector in rphi direction
  void getLocalCorrectionVectorCyl(const DataT z, const DataT r, const DataT phi, const Side side, DataT& lveccorrZ, DataT& lveccorrR, DataT& lveccorrRPhi) const;

  /// get the local correction vector for given coordinate
  /// \param z global z coordinates
  /// \param r global r coordinates
  /// \param phi global phi coordinates
  /// \param ldistZ returns local correction vectors in z direction
  /// \param ldistR returns local correction vectors in r direction
  /// \param ldistRPhi returns local correction vectors in rphi direction
  void getLocalCorrectionVectorCyl(const std::vector<DataT>& z, const std::vector<DataT>& r, const std::vector<DataT>& phi, const Side side, std::vector<DataT>& lveccorrZ, std::vector<DataT>& lveccorrR, std::vector<DataT>& lveccorrRPhi) const;

  /// get the global distortions for given coordinate
  /// \param z global z coordinate
  /// \param r global r coordinate
  /// \param phi global phi coordinate
  /// \param distZ returns distortion in z direction
  /// \param distR returns distortion in r direction
  /// \param distRPhi returns distortion in rphi direction
  void getDistortionsCyl(const DataT z, const DataT r, const DataT phi, const Side side, DataT& distZ, DataT& distR, DataT& distRPhi) const;

  /// get the global distortions for given coordinate
  /// \param z global z coordinates
  /// \param r global r coordinates
  /// \param phi global phi coordinates
  /// \param distZ returns distortions in z direction
  /// \param distR returns distortions in r direction
  /// \param distRPhi returns distortions in rphi direction
  void getDistortionsCyl(const std::vector<DataT>& z, const std::vector<DataT>& r, const std::vector<DataT>& phi, const Side side, std::vector<DataT>& distZ, std::vector<DataT>& distR, std::vector<DataT>& distRPhi) const;

  /// get the global distortions for given coordinate
  /// \param x global x coordinate
  /// \param y global y coordinate
  /// \param z global z coordinate
  /// \param distX returns distortion in x direction
  /// \param distY returns distortion in y direction
  /// \param distZ returns distortion in z direction
  void getDistortions(const DataT x, const DataT y, const DataT z, const Side side, DataT& distX, DataT& distY, DataT& distZ) const;

  /// get the global distortions for given coordinate for a possible analytical formula if it was specified
  /// \param x global x coordinate
  /// \param y global y coordinate
  /// \param z global z coordinate
  /// \param distX returns distortion in x direction
  /// \param distY returns distortion in y direction
  /// \param distZ returns distortion in z direction
  void getDistortionsAnalytical(const DataT x, const DataT y, const DataT z, const Side side, DataT& distX, DataT& distY, DataT& distZ) const { getDistortionsCorrectionsAnalytical(x, y, z, side, distX, distY, distZ, true); }

  /// set distortions and corrections by an analytical formula
  void setDistortionsCorrectionsAnalytical(const AnalyticalDistCorr<DataT>& formula) { mAnaDistCorr = formula; }

  /// \return returns analytical distortions/corrections
  const auto& getDistortionsCorrectionsAnalytical() const& { return mAnaDistCorr; }

  /// setting usage of the analytical formula for the distortions and corrections
  void setUseAnalyticalDistCorr(const bool useAna) { mUseAnaDistCorr = useAna; }

  /// \return returns if the analytical formula will be used in the distortElectron() and getCorrections() function
  bool getUseAnalyticalDistCorr() const { return mUseAnaDistCorr; }

  /// convert x and y coordinates from cartesian to the radius in polar coordinates
  static DataT getRadiusFromCartesian(const DataT x, const DataT y) { return std::sqrt(x * x + y * y); }

  /// convert x and y coordinates from cartesian to phi in polar coordinates
  static DataT getPhiFromCartesian(const DataT x, const DataT y) { return std::atan2(y, x); }

  /// convert radius and phi coordinates from polar coordinates to x cartesian coordinates
  static DataT getXFromPolar(const DataT r, const DataT phi) { return r * std::cos(phi); }

  /// convert radius and phi coordinates from polar coordinates to y cartesian coordinate
  static DataT getYFromPolar(const DataT r, const DataT phi) { return r * std::sin(phi); }

  /// Correct electron position using correction lookup tables
  /// \param point 3D coordinates of the electron
  void correctElectron(GlobalPosition3D& point);

  /// Distort electron position using distortion lookup tables
  /// \param point 3D coordinates of the electron
  void distortElectron(GlobalPosition3D& point) const;

  /// set the distortions directly from a look up table
  /// \param distdZ distortions in z direction
  /// \param distdR distortions in r direction
  /// \param distdRPhi distortions in rphi direction
  /// \param side side of the TPC
  void setDistortionLookupTables(const DataContainer& distdZ, const DataContainer& distdR, const DataContainer& distdRPhi, const Side side);

  /// set the density, potential, electric fields, local distortions/corrections, global distortions/corrections from a file. Missing objects in the file are ignored.
  /// \file file containing the stored values for the density, potential, electric fields, local distortions/corrections, global distortions/corrections
  /// \param side side of the TPC
  template <typename DataTIn = DataT>
  void setFromFile(TFile& file, const Side side);

  /// Get grid spacing in r direction
  DataT getGridSpacingR(const Side side) const { return mGrid3D[side].getSpacingR(); }

  /// Get grid spacing in z direction
  DataT getGridSpacingZ(const Side side) const { return mGrid3D[side].getSpacingZ(); }

  /// Get grid spacing in phi direction
  DataT getGridSpacingPhi(const Side side) const { return mGrid3D[side].getSpacingPhi(); }

  /// Get constant electric field
  static constexpr DataT getEzField(const Side side) { return getSign(side) * (TPCParameters<DataT>::cathodev - TPCParameters<DataT>::vg1t) / TPCParameters<DataT>::TPCZ0; }

  /// Get inner radius of tpc
  DataT getRMin(const Side side) const { return mGrid3D[side].getGridMinR(); }

  /// Get min z position which is used during the calaculations
  DataT getZMin(const Side side) const { return mGrid3D[side].getGridMinZ(); }

  /// Get min phi
  DataT getPhiMin(const Side side) const { return mGrid3D[side].getGridMinPhi(); }

  /// Get max r
  DataT getRMax(const Side side) const { return mGrid3D[side].getGridMaxR(); };

  /// Get max z
  DataT getZMax(const Side side) const { return mGrid3D[side].getGridMaxZ(); }

  /// Get max phi
  DataT getPhiMax(const Side side) const { return mGrid3D[side].getGridMaxPhi(); }

  // get side of TPC for z coordinate TODO rewrite this
  static Side getSide(const DataT z) { return ((z >= 0) ? Side::A : Side::C); }

  /// Get the grid object
  const RegularGrid& getGrid3D(const Side side) const { return mGrid3D[side]; }

  /// Get struct containing interpolators for the electrical fields
  /// \param side side of the TPC
  NumericalFields<DataT> getElectricFieldsInterpolator(const Side side) const;

  /// Get struct containing interpolators for local distortions dR, dZ, dPhi
  /// \param side side of the TPC
  DistCorrInterpolator<DataT> getLocalDistInterpolator(const Side side) const;

  /// Get struct containing interpolators for local corrections dR, dZ, dPhi
  /// \param side side of the TPC
  DistCorrInterpolator<DataT> getLocalCorrInterpolator(const Side side) const;

  /// Get struct containing interpolators for global distortions dR, dZ, dPhi
  /// \param side side of the TPC
  DistCorrInterpolator<DataT> getGlobalDistInterpolator(const Side side) const;

  /// Get struct containing interpolators for global corrections dR, dZ, dPhi
  /// \param side side of the TPC
  DistCorrInterpolator<DataT> getGlobalCorrInterpolator(const Side side) const;

  /// \param iz vertex in z dimension
  /// \param ir vertex in r dimension
  /// \param iphi vertex in phi dimension
  /// \param side side of the TPC
  /// \return returns local distortion dR for given vertex
  DataT getLocalDistR(const size_t iz, const size_t ir, const size_t iphi, const Side side) const { return mLocalDistdR[side](iz, ir, iphi); }

  /// \param iz vertex in z dimension
  /// \param ir vertex in r dimension
  /// \param iphi vertex in phi dimension
  /// \param side side of the TPC
  /// \return returns local distortion dZ for given vertex
  DataT getLocalDistZ(const size_t iz, const size_t ir, const size_t iphi, const Side side) const { return mLocalDistdZ[side](iz, ir, iphi); }

  /// \param iz vertex in z dimension
  /// \param ir vertex in r dimension
  /// \param iphi vertex in phi dimension
  /// \param side side of the TPC
  /// \return returns local distortion dRPhi for given vertex
  DataT getLocalDistRPhi(const size_t iz, const size_t ir, const size_t iphi, const Side side) const { return mLocalDistdRPhi[side](iz, ir, iphi); }

  /// \param iz vertex in z dimension
  /// \param ir vertex in r dimension
  /// \param iphi vertex in phi dimension
  /// \param side side of the TPC
  /// \return returns local distortion vector dR for given vertex
  DataT getLocalVecDistR(const size_t iz, const size_t ir, const size_t iphi, const Side side) const { return mLocalVecDistdR[side](iz, ir, iphi); }

  /// \param iz vertex in z dimension
  /// \param ir vertex in r dimension
  /// \param iphi vertex in phi dimension
  /// \param side side of the TPC
  /// \return returns local distortion vector dZ for given vertex
  DataT getLocalVecDistZ(const size_t iz, const size_t ir, const size_t iphi, const Side side) const { return mLocalVecDistdZ[side](iz, ir, iphi); }

  /// \param iz vertex in z dimension
  /// \param ir vertex in r dimension
  /// \param iphi vertex in phi dimension
  /// \param side side of the TPC
  /// \return returns local distortion vector dRPhi for given vertex
  DataT getLocalVecDistRPhi(const size_t iz, const size_t ir, const size_t iphi, const Side side) const { return mLocalVecDistdRPhi[side](iz, ir, iphi); }

  /// \param iz vertex in z dimension
  /// \param ir vertex in r dimension
  /// \param iphi vertex in phi dimension
  /// \param side side of the TPC
  /// \return returns local correction dR for given vertex
  DataT getLocalCorrR(const size_t iz, const size_t ir, const size_t iphi, const Side side) const { return mLocalCorrdR[side](iz, ir, iphi); }

  /// \param iz vertex in z dimension
  /// \param ir vertex in r dimension
  /// \param iphi vertex in phi dimension
  /// \param side side of the TPC
  /// \return returns local correction dZ for given vertex
  DataT getLocalCorrZ(const size_t iz, const size_t ir, const size_t iphi, const Side side) const { return mLocalCorrdZ[side](iz, ir, iphi); }

  /// \param iz vertex in z dimension
  /// \param ir vertex in r dimension
  /// \param iphi vertex in phi dimension
  /// \param side side of the TPC
  /// \return returns local correction dRPhi for given vertex
  DataT getLocalCorrRPhi(const size_t iz, const size_t ir, const size_t iphi, const Side side) const { return mLocalCorrdRPhi[side](iz, ir, iphi); }

  /// \param iz vertex in z dimension
  /// \param ir vertex in r dimension
  /// \param iphi vertex in phi dimension
  /// \param side side of the TPC
  /// \return returns local correction vector dR for given vertex
  DataT getLocalVecCorrR(const size_t iz, const size_t ir, const size_t iphi, const Side side) const { return -mLocalVecDistdR[side](iz, ir, iphi); }

  /// \param iz vertex in z dimension
  /// \param ir vertex in r dimension
  /// \param iphi vertex in phi dimension
  /// \param side side of the TPC
  /// \return returns local correction vector dZ for given vertex
  DataT getLocalVecCorrZ(const size_t iz, const size_t ir, const size_t iphi, const Side side) const { return -mLocalVecDistdZ[side](iz, ir, iphi); }

  /// \param iz vertex in z dimension
  /// \param ir vertex in r dimension
  /// \param iphi vertex in phi dimension
  /// \param side side of the TPC
  /// \return returns local correction vector dRPhi for given vertex
  DataT getLocalVecCorrRPhi(const size_t iz, const size_t ir, const size_t iphi, const Side side) const { return -mLocalVecDistdRPhi[side](iz, ir, iphi); }

  /// Get global distortion dR for vertex
  /// \param iz vertex in z dimension
  /// \param ir vertex in r dimension
  /// \param iphi vertex in phi dimension
  /// \param side side of the TPC
  DataT getGlobalDistR(const size_t iz, const size_t ir, const size_t iphi, const Side side) const { return mGlobalDistdR[side](iz, ir, iphi); }

  /// Get global distortion dZ for vertex
  /// \param iz vertex in z dimension
  /// \param ir vertex in r dimension
  /// \param iphi vertex in phi dimension
  /// \param side side of the TPC
  DataT getGlobalDistZ(const size_t iz, const size_t ir, const size_t iphi, const Side side) const { return mGlobalDistdZ[side](iz, ir, iphi); }

  /// Get global distortion dRPhi for vertex
  /// \param iz vertex in z dimension
  /// \param ir vertex in r dimension
  /// \param iphi vertex in phi dimension
  /// \param side side of the TPC
  DataT getGlobalDistRPhi(const size_t iz, const size_t ir, const size_t iphi, const Side side) const { return mGlobalDistdRPhi[side](iz, ir, iphi); }

  /// Get global correction dR for vertex
  /// \param iz vertex in z dimension
  /// \param ir vertex in r dimension
  /// \param iphi vertex in phi dimension
  /// \param side side of the TPC
  DataT getGlobalCorrR(const size_t iz, const size_t ir, const size_t iphi, const Side side) const { return mGlobalCorrdR[side](iz, ir, iphi); }

  /// Get global correction dZ for vertex
  /// \param iz vertex in z dimension
  /// \param ir vertex in r dimension
  /// \param iphi vertex in phi dimension
  /// \param side side of the TPC
  DataT getGlobalCorrZ(const size_t iz, const size_t ir, const size_t iphi, const Side side) const { return mGlobalCorrdZ[side](iz, ir, iphi); }

  /// Get global correction dRPhi for vertex
  /// \param iz vertex in z dimension
  /// \param ir vertex in r dimension
  /// \param iphi vertex in phi dimension
  /// \param side side of the TPC
  DataT getGlobalCorrRPhi(const size_t iz, const size_t ir, const size_t iphi, const Side side) const { return mGlobalCorrdRPhi[side](iz, ir, iphi); }

  /// Get global electric Field Er for vertex
  /// \param iz vertex in z dimension
  /// \param ir vertex in r dimension
  /// \param iphi vertex in phi dimension
  /// \param side side of the TPC
  DataT getEr(const size_t iz, const size_t ir, const size_t iphi, const Side side) const { return mElectricFieldEr[side](iz, ir, iphi); }

  /// Get global electric Field Ez for vertex
  /// \param iz vertex in z dimension
  /// \param ir vertex in r dimension
  /// \param iphi vertex in phi dimension
  /// \param side side of the TPC
  DataT getEz(const size_t iz, const size_t ir, const size_t iphi, const Side side) const { return mElectricFieldEz[side](iz, ir, iphi); }

  /// Get global electric Field Ephi for vertex
  /// \param iz vertex in z dimension
  /// \param ir vertex in r dimension
  /// \param iphi vertex in phi dimension
  /// \param side side of the TPC
  DataT getEphi(const size_t iz, const size_t ir, const size_t iphi, const Side side) const { return mElectricFieldEphi[side](iz, ir, iphi); }

  /// Get density for vertex
  /// \param iz vertex in z dimension
  /// \param ir vertex in r dimension
  /// \param iphi vertex in phi dimension
  /// \param side side of the TPC
  DataT getDensity(const size_t iz, const size_t ir, const size_t iphi, const Side side) const { return mDensity[side](iz, ir, iphi); }

  /// Get potential for vertex
  /// \param iz vertex in z dimension
  /// \param ir vertex in r dimension
  /// \param iphi vertex in phi dimension
  /// \param side side of the TPC
  DataT getPotential(const size_t iz, const size_t ir, const size_t iphi, const Side side) const { return mPotential[side](iz, ir, iphi); }

  /// Get the step width which is used for the calculation of the correction/distortions in units of the z-bin
  static int getStepWidth() { return 1 / sSteps; }

  /// Get phi vertex position for index in phi direction
  /// \param indexPhi index in phi direction
  DataT getPhiVertex(const size_t indexPhi, const Side side) const { return mGrid3D[side].getPhiVertex(indexPhi); }

  /// Get r vertex position for index in r direction
  /// \param indexR index in r direction
  DataT getRVertex(const size_t indexR, const Side side) const { return mGrid3D[side].getRVertex(indexR); }

  /// Get z vertex position for index in z direction
  /// \param indexZ index in z direction
  DataT getZVertex(const size_t indexZ, const Side side) const { return mGrid3D[side].getZVertex(indexZ); }

  /// \param omegaTau \omega \tau value
  /// \param t1 value for t1 see: ???
  /// \param t2 value for t2 see: ???
  void setOmegaTauT1T2(const DataT omegaTau = 0.32f, const DataT t1 = 1, const DataT t2 = 1)
  {
    const DataT wt0 = t2 * omegaTau;
    mC0 = 1 / (1 + wt0 * wt0);
    const DataT wt1 = t1 * omegaTau;
    mC1 = wt1 / (1 + wt1 * wt1);
  };

  /// \param c0 coefficient C0 (compare Jim Thomas's notes for definitions)
  /// \param c1 coefficient C1 (compare Jim Thomas's notes for definitions)
  void setC0C1(const DataT c0, const DataT c1)
  {
    mC0 = c0;
    mC1 = c1;
  }

  /// set number of steps used for calculation of distortions/corrections per z bin
  /// \param nSteps number of steps per z bin
  static void setNStep(const int nSteps) { sSteps = nSteps; }

  static int getNStep() { return sSteps; }

  /// get the number of threads used for some of the calculations
  static int getNThreads() { return sNThreads; }

  /// set the number of threads used for some of the calculations
  static void setNThreads(const int nThreads) { sNThreads = nThreads; }

  /// set which kind of numerical integration is used for calcution of the integrals int Er/Ez dz, int Ephi/Ez dz, int Ez dz
  /// \param strategy numerical integration strategy. see enum IntegrationStrategy for the different types
  static void setNumericalIntegrationStrategy(const IntegrationStrategy strategy) { sNumericalIntegrationStrategy = strategy; }
  static IntegrationStrategy getNumericalIntegrationStrategy() { return sNumericalIntegrationStrategy; }

  static void setGlobalDistType(const GlobalDistType globalDistType) { sGlobalDistType = globalDistType; }
  static GlobalDistType getGlobalDistType() { return sGlobalDistType; }

  static void setGlobalDistCorrMethod(const GlobalDistCorrMethod globalDistCorrMethod) { sGlobalDistCorrCalcMethod = globalDistCorrMethod; }
  static GlobalDistCorrMethod getGlobalDistCorrMethod() { return sGlobalDistCorrCalcMethod; }

  static void setSimpsonNIteratives(const int nIter) { sSimpsonNIteratives = nIter; }
  static int getSimpsonNIteratives() { return sSimpsonNIteratives; }

  /// Set the space-charge distortions model
  /// \param distortionType distortion type (constant or realistic)
  static void setSCDistortionType(SCDistortionType distortionType) { sSCDistortionType = distortionType; }
  /// Get the space-charge distortions model
  static SCDistortionType getSCDistortionType() { return sSCDistortionType; }

  void setUseInitialSCDensity(const bool useInitialSCDensity) { mUseInitialSCDensity = useInitialSCDensity; }

  /// write all fields etc to a file
  /// \param outf output file
  /// \side side of the TPC
  template <typename DataTOut = DataT>
  void dumpToFile(TFile& outf, const Side side) const;

  /// dump sc density, potential, electric fields, global distortions/corrections to tree
  /// \param outFileName name of the output file
  /// \side side of the TPC
  /// \param nZPoints number of vertices of the output in z
  /// \param nRPoints number of vertices of the output in r
  /// \param nPhiPoints number of vertices of the output in phi
  void dumpToTree(const char* outFileName, const Side side, const int nZPoints = 50, const int nRPoints = 150, const int nPhiPoints = 180) const;

  /// write electric fields to root file
  /// \param outf output file where the electrical fields will be written to
  /// \side side of the TPC
  template <typename DataTOut = DataT>
  int dumpElectricFields(TFile& outf, const Side side) const;

  /// set electric field from root file
  /// \param inpf input file where the electrical fields are stored
  /// \side side of the TPC
  template <typename DataTIn = DataT>
  void setElectricFieldsFromFile(TFile& inpf, const Side side);

  /// write potential to root file
  /// \param outf output file where the potential will be written to
  /// \side side of the TPC
  template <typename DataTOut = DataT>
  int dumpPotential(TFile& outf, const Side side) const
  {
    return mPotential[side].template writeToFile<DataTOut>(outf, Form("potential_side%s", getSideName(side).data()));
  }

  /// set potential from root file
  /// \param inpf input file where the potential is stored
  /// \side side of the TPC
  template <typename DataTIn = DataT>
  void setPotentialFromFile(TFile& inpf, const Side side)
  {
    mPotential[side].template initFromFile<DataTIn>(inpf, Form("potential_side%s", getSideName(side).data()));
  }

  /// set the potential directly
  /// \param potential potential which will be set
  /// \param iz vertex in z dimension
  /// \param ir vertex in r dimension
  /// \param iphi vertex in phi dimension
  /// \side side of the TPC
  void fillPotential(const DataT potential, const size_t iz, const size_t ir, const size_t iphi, const Side side) { mPotential[side](iz, ir, iphi) = potential; }

  /// write density to root file
  /// \param outf output file where the charge density will be written to
  /// \side side of the TPC
  template <typename DataTOut = DataT>
  int dumpDensity(TFile& outf, const Side side) const
  {
    return mDensity[side].template writeToFile<DataTOut>(outf, Form("density_side%s", getSideName(side).data()));
  }

  /// set density from root file
  /// \param inpf input file where the charge density is stored
  /// \side side of the TPC
  template <typename DataTIn = DataT>
  void setDensityFromFile(TFile& inpf, const Side side)
  {
    mDensity[side].template initFromFile<DataTIn>(inpf, Form("density_side%s", getSideName(side).data()));
    setDensityFilled(side);
  }

  /// set the space charge density directly
  /// \param density space charege density which will be set
  /// \param iz vertex in z dimension
  /// \param ir vertex in r dimension
  /// \param iphi vertex in phi dimension
  /// \side side of the TPC
  void fillDensity(const DataT density, const size_t iz, const size_t ir, const size_t iphi, const Side side) { mDensity[side](iz, ir, iphi) = density; }

  /// set the status of the density as filled
  /// \side side of the TPC
  void setDensityFilled(const Side side) { mIsChargeSet[side] = true; }

  /// write global distortions to root file
  /// \param outf output file where the global distortions will be written to
  /// \side side of the TPC
  template <typename DataTOut = DataT>
  int dumpGlobalDistortions(TFile& outf, const Side side) const;

  /// write analytical corrections and distortions to file
  /// \param outf output file where the analytical corrections and distortions will be written to
  int dumpAnalyticalCorrectionsDistortions(TFile& outf) const;

  /// set analytical corrections and distortions from file
  /// \param inpf input file where the analytical corrections and distortions are stored
  void setAnalyticalCorrectionsDistortionsFromFile(TFile& inpf);

  /// set global distortions from root file
  /// \param inpf input file where the global distortions are stored
  /// \side side of the TPC
  template <typename DataTIn = DataT>
  void setGlobalDistortionsFromFile(TFile& inpf, const Side side);

  /// write global corrections to root file
  /// \param outf output file where the global corrections will be written to
  /// \side side of the TPC
  template <typename DataTOut = DataT>
  int dumpGlobalCorrections(TFile& outf, const Side side) const;

  /// set global corrections from root file
  /// \param inpf input file where the global corrections are stored
  /// \side side of the TPC
  template <typename DataTIn = DataT>
  void setGlobalCorrectionsFromFile(TFile& inpf, const Side side);

  /// write local corrections to root file
  /// \param outf output file where the local corrections will be written to
  /// \side side of the TPC
  template <typename DataTOut = DataT>
  int dumpLocalCorrections(TFile& outf, const Side side) const;

  /// set local corrections from root file
  /// \param inpf input file where the local corrections are stored
  /// \side side of the TPC
  template <typename DataTIn = DataT>
  void setLocalCorrectionsFromFile(TFile& inpf, const Side side);

  /// write local distortions to root file
  /// \param outf output file where the local distortions will be written to
  /// \side side of the TPC
  template <typename DataTOut = DataT>
  int dumpLocalDistortions(TFile& outf, const Side side) const;

  /// write local distortion vector to root file
  /// \param outf output file where the local distortions will be written to
  /// \side side of the TPC
  template <typename DataTOut = DataT>
  int dumpLocalDistCorrVectors(TFile& outf, const Side side) const;

  /// set local distortions from root file
  /// \param inpf input file where the local distortions are stored
  /// \side side of the TPC
  template <typename DataTIn = DataT>
  void setLocalDistortionsFromFile(TFile& inpf, const Side side);

  /// set local distortion vector from root file
  /// \param inpf input file where the local distortions are stored
  /// \side side of the TPC
  template <typename DataTIn = DataT>
  void setLocalDistCorrVectorsFromFile(TFile& inpf, const Side side);

  /// set z coordinate between min z max z
  /// \param posZ z position which will be regulated if needed
  DataT regulateZ(const DataT posZ, const Side side) const { return mGrid3D[side].clampToGrid(posZ, 0); }

  /// set r coordinate between 'RMIN - 4 * GRIDSPACINGR' and 'RMAX + 2 * GRIDSPACINGR'. the r coordinate is not clamped to RMIN and RMAX to ensure correct interpolation at the borders of the grid.
  DataT regulateR(const DataT posR, const Side side) const;

  /// set phi coordinate between min phi max phi
  DataT regulatePhi(const DataT posPhi, const Side side) const { return mGrid3D[side].clampToGridCircular(posPhi, 2); }

  /// rebin the input space charge density histogram to desired binning
  /// \param hOrig original histogram
  /// \param nBinsZNew number of z bins of rebinned histogram
  /// \param nBinsRNew number of r bins of rebinned histogram
  /// \param nBinsPhiNew number of phi bins of rebinned histogram
  static TH3DataT rebinDensityHisto(const TH3& hOrig, const unsigned short nBinsZNew, const unsigned short nBinsRNew, const unsigned short nBinsPhiNew);

  /// function to calculate the drift paths of the electron whose starting position is delivered. Electric fields must be set!
  /// \param elePos global position of the start position of the electron
  /// \param nSamplingPoints number of output points of the electron drift path
  /// \param outFile name of the output debug file (if empty no file is created)
  /// \return returns the input electron a vector of 3D-points describing the drift path of the electron
  std::vector<std::pair<std::vector<o2::math_utils::Point3D<float>>, std::array<DataT, 3>>> calculateElectronDriftPath(const std::vector<GlobalPosition3D>& elePos, const int nSamplingPoints, const std::string_view outFile = "electron_tracks.root") const;

  /// \param inpFile input file containing the electron tracks tree, which is the output file of calculateElectronDriftPath()
  /// \param hBorder histogram which defines the borders for the drawing and also the axis titles
  /// \param type setting dimensions: type=0: radius vs z, type=0: radius vs phi
  /// \param gifSpeed speed of the output gif file (fastest is 2, slowest is 99)
  /// \param maxsamplingpoints maximum number of frames which will be drawn (higher number increases processing time)
  /// \param outName name of the output file
  static void makeElectronDriftPathGif(const char* inpFile, TH2F& hBorder, const int type = 0, const int gifSpeed = 2, const int maxsamplingpoints = 100, const char* outName = "electron_drift_path");

  /// normalize a histogram containing the charge to the space charge density
  /// \param histoIonsPhiRZ histogram which will be normalized to the sapce charge density
  static void normalizeHistoQVEps0(TH3& histoIonsPhiRZ);

  /// \return returns max threads
  static int getOMPMaxThreads();

  /// write output of streamer to file
  void flushStreamer() { mStreamer.flush(); }

 private:
  inline static auto& mParamGrid = ParameterSpaceCharge::Instance();                                      ///<! parameters of the grid on which the calculations are performed
  inline static int sNThreads{getOMPMaxThreads()};                                                        ///<! number of threads which are used during the calculations
  inline static IntegrationStrategy sNumericalIntegrationStrategy{IntegrationStrategy::SimpsonIterative}; ///<! numerical integration strategy of integration of the E-Field: 0: trapezoidal, 1: Simpson, 2: Root (only for analytical formula case)
  inline static int sSimpsonNIteratives{3};                                                               ///<! number of iterations which are performed in the iterative simpson calculation of distortions/corrections
  inline static int sSteps{1};                                                                            ///<! during the calculation of the corrections/distortions it is assumed that the electron drifts on a line from deltaZ = z0 -> z1. The value sets the deltaZ width: 1: deltaZ=zBin/1, 5: deltaZ=zBin/5
  inline static GlobalDistType sGlobalDistType{GlobalDistType::Fast};                                     ///<! setting for global distortions: 0: standard method,      1: interpolation of global corrections
  inline static GlobalDistCorrMethod sGlobalDistCorrCalcMethod{GlobalDistCorrMethod::LocalDistCorr};      ///<! setting for  global distortions/corrections: 0: using electric field, 1: using local dis/corr interpolator
  inline static SCDistortionType sSCDistortionType{SCDistortionType::SCDistortionsConstant};              ///<! Type of space-charge distortions

  DataT mC0 = 0;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ///< coefficient C0 (compare Jim Thomas's notes for definitions)
  DataT mC1 = 0;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ///< coefficient C1 (compare Jim Thomas's notes for definitions)
  static constexpr int FNSIDES = SIDES;                                                                                                                                                                                                                                                                                                                                                                                                                                                                 ///< number of sides of the TPC
  bool mIsEfieldSet[FNSIDES]{};                                                                                                                                                                                                                                                                                                                                                                                                                                                                         ///< flag if E-fields are set
  bool mIsLocalCorrSet[FNSIDES]{};                                                                                                                                                                                                                                                                                                                                                                                                                                                                      ///< flag if local corrections are set
  bool mIsLocalDistSet[FNSIDES]{};                                                                                                                                                                                                                                                                                                                                                                                                                                                                      ///< flag if local distortions are set
  bool mIsLocalVecDistSet[FNSIDES]{};                                                                                                                                                                                                                                                                                                                                                                                                                                                                   ///< flag if local distortions vectors are set
  bool mIsGlobalCorrSet[FNSIDES]{};                                                                                                                                                                                                                                                                                                                                                                                                                                                                     ///< flag if global corrections are set
  bool mIsGlobalDistSet[FNSIDES]{};                                                                                                                                                                                                                                                                                                                                                                                                                                                                     ///< flag if global distortions are set
  bool mIsChargeSet[FNSIDES]{};                                                                                                                                                                                                                                                                                                                                                                                                                                                                         ///< flag if the charge is set
  bool mUseInitialSCDensity{false};                                                                                                                                                                                                                                                                                                                                                                                                                                                                     ///< Flag for the use of an initial space-charge density at the beginning of the simulation
  bool mInitLookUpTables{false};                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ///< Flag to indicate if lookup tables have been calculated
  const RegularGrid mGrid3D[FNSIDES]{{GridProp::ZMIN, GridProp::RMIN, GridProp::PHIMIN, getSign(Side::A) * GridProp::getGridSpacingZ(mParamGrid.NZVertices), GridProp::getGridSpacingR(mParamGrid.NRVertices), GridProp::getGridSpacingPhi(mParamGrid.NPhiVertices)}, {GridProp::ZMIN, GridProp::RMIN, GridProp::PHIMIN, getSign(Side::C) * GridProp::getGridSpacingZ(mParamGrid.NZVertices), GridProp::getGridSpacingR(mParamGrid.NRVertices), GridProp::getGridSpacingPhi(mParamGrid.NPhiVertices)}}; ///<! grid properties
  DataContainer mLocalDistdR[FNSIDES]{DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices), DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices)};                                                                                                                                                                                                                                                                                      ///< data storage for local distortions dR
  DataContainer mLocalDistdZ[FNSIDES]{DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices), DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices)};                                                                                                                                                                                                                                                                                      ///< data storage for local distortions dZ
  DataContainer mLocalDistdRPhi[FNSIDES]{DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices), DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices)};                                                                                                                                                                                                                                                                                   ///< data storage for local distortions dRPhi
  DataContainer mLocalVecDistdR[FNSIDES]{DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices), DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices)};                                                                                                                                                                                                                                                                                   ///< data storage for local distortions vector dR
  DataContainer mLocalVecDistdZ[FNSIDES]{DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices), DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices)};                                                                                                                                                                                                                                                                                   ///< data storage for local distortions vector dZ
  DataContainer mLocalVecDistdRPhi[FNSIDES]{DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices), DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices)};                                                                                                                                                                                                                                                                                ///< data storage for local distortions vector dRPhi
  DataContainer mLocalCorrdR[FNSIDES]{DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices), DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices)};                                                                                                                                                                                                                                                                                      ///< data storage for local corrections dR
  DataContainer mLocalCorrdZ[FNSIDES]{DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices), DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices)};                                                                                                                                                                                                                                                                                      ///< data storage for local corrections dZ
  DataContainer mLocalCorrdRPhi[FNSIDES]{DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices), DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices)};                                                                                                                                                                                                                                                                                   ///< data storage for local corrections dRPhi
  DataContainer mGlobalDistdR[FNSIDES]{DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices), DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices)};                                                                                                                                                                                                                                                                                     ///< data storage for global distortions dR
  DataContainer mGlobalDistdZ[FNSIDES]{DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices), DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices)};                                                                                                                                                                                                                                                                                     ///< data storage for global distortions dZ
  DataContainer mGlobalDistdRPhi[FNSIDES]{DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices), DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices)};                                                                                                                                                                                                                                                                                  ///< data storage for global distortions dRPhi
  DataContainer mGlobalCorrdR[FNSIDES]{DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices), DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices)};                                                                                                                                                                                                                                                                                     ///< data storage for global corrections dR
  DataContainer mGlobalCorrdZ[FNSIDES]{DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices), DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices)};                                                                                                                                                                                                                                                                                     ///< data storage for global corrections dZ
  DataContainer mGlobalCorrdRPhi[FNSIDES]{DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices), DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices)};                                                                                                                                                                                                                                                                                  ///< data storage for global corrections dRPhi
  DataContainer mDensity[FNSIDES]{DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices), DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices)};                                                                                                                                                                                                                                                                                          ///< data storage for space charge density
  DataContainer mPotential[FNSIDES]{DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices), DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices)};                                                                                                                                                                                                                                                                                        ///< data storage for the potential
  DataContainer mElectricFieldEr[FNSIDES]{DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices), DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices)};                                                                                                                                                                                                                                                                                  ///< data storage for the electric field Er
  DataContainer mElectricFieldEz[FNSIDES]{DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices), DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices)};                                                                                                                                                                                                                                                                                  ///< data storage for the electric field Ez
  DataContainer mElectricFieldEphi[FNSIDES]{DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices), DataContainer(mParamGrid.NZVertices, mParamGrid.NRVertices, mParamGrid.NPhiVertices)};                                                                                                                                                                                                                                                                                ///< data storage for the electric field Ephi
  TriCubic mInterpolatorPotential[FNSIDES]{{mPotential[Side::A], mGrid3D[Side::A]}, {mPotential[Side::C], mGrid3D[Side::C]}};                                                                                                                                                                                                                                                                                                                                                                           ///<! interpolator for the potenial
  TriCubic mInterpolatorDensity[FNSIDES]{{mDensity[Side::A], mGrid3D[Side::A]}, {mDensity[Side::C], mGrid3D[Side::C]}};                                                                                                                                                                                                                                                                                                                                                                                 ///<! interpolator for the charge
  DistCorrInterpolator<DataT> mInterpolatorGlobalCorr[FNSIDES]{{mGlobalCorrdR[Side::A], mGlobalCorrdZ[Side::A], mGlobalCorrdRPhi[Side::A], mGrid3D[Side::A], Side::A}, {mGlobalCorrdR[Side::C], mGlobalCorrdZ[Side::C], mGlobalCorrdRPhi[Side::C], mGrid3D[Side::C], Side::C}};                                                                                                                                                                                                                         ///<! interpolator for the global corrections
  DistCorrInterpolator<DataT> mInterpolatorLocalCorr[FNSIDES]{{mLocalCorrdR[Side::A], mLocalCorrdZ[Side::A], mLocalCorrdRPhi[Side::A], mGrid3D[Side::A], Side::A}, {mLocalCorrdR[Side::C], mLocalCorrdZ[Side::C], mLocalCorrdRPhi[Side::C], mGrid3D[Side::C], Side::C}};                                                                                                                                                                                                                                ///<! interpolator for the local corrections
  DistCorrInterpolator<DataT> mInterpolatorGlobalDist[FNSIDES]{{mGlobalDistdR[Side::A], mGlobalDistdZ[Side::A], mGlobalDistdRPhi[Side::A], mGrid3D[Side::A], Side::A}, {mGlobalDistdR[Side::C], mGlobalDistdZ[Side::C], mGlobalDistdRPhi[Side::C], mGrid3D[Side::C], Side::C}};                                                                                                                                                                                                                         ///<! interpolator for the global distortions
  DistCorrInterpolator<DataT> mInterpolatorLocalDist[FNSIDES]{{mLocalDistdR[Side::A], mLocalDistdZ[Side::A], mLocalDistdRPhi[Side::A], mGrid3D[Side::A], Side::A}, {mLocalDistdR[Side::C], mLocalDistdZ[Side::C], mLocalDistdRPhi[Side::C], mGrid3D[Side::C], Side::C}};                                                                                                                                                                                                                                ///<! interpolator for the local distortions
  DistCorrInterpolator<DataT> mInterpolatorLocalVecDist[FNSIDES]{{mLocalVecDistdR[Side::A], mLocalVecDistdZ[Side::A], mLocalVecDistdRPhi[Side::A], mGrid3D[Side::A], Side::A}, {mLocalVecDistdR[Side::C], mLocalVecDistdZ[Side::C], mLocalVecDistdRPhi[Side::C], mGrid3D[Side::C], Side::C}};                                                                                                                                                                                                           ///<! interpolator for the local distortion vectors
  NumericalFields<DataT> mInterpolatorEField[FNSIDES]{{mElectricFieldEr[Side::A], mElectricFieldEz[Side::A], mElectricFieldEphi[Side::A], mGrid3D[Side::A], Side::A}, {mElectricFieldEr[Side::C], mElectricFieldEz[Side::C], mElectricFieldEphi[Side::C], mGrid3D[Side::C], Side::C}};                                                                                                                                                                                                                  ///<! interpolator for the electric fields
  AnalyticalDistCorr<DataT> mAnaDistCorr;
  bool mUseAnaDistCorr{false};        ///< flag if analytical distortions will be used in the distortElectron() and getCorrections() function
  o2::utils::DebugStreamer mStreamer; ///<! debug streamer for distortions

  /// check if the addition of two values are close to zero.
  /// This avoids errors during the integration of the electric fields when the sum of the nominal electric with the electric field from the space charge is close to 0 (usually this is not the case!).
  bool isCloseToZero(const DataT valA, const DataT valB) const { return std::abs(valA + valB) < static_cast<DataT>(0.01); }

  static int getSign(const Side side) { return side == Side::C ? -1 : 1; }

  /// get inverse spacing in z direction
  DataT getInvSpacingZ(const Side side) const { return mGrid3D[side].getInvSpacingZ(); }

  /// get inverse spacing in r direction
  DataT getInvSpacingR(const Side side) const { return mGrid3D[side].getInvSpacingR(); }

  /// get inverse spacing in phi direction
  DataT getInvSpacingPhi(const Side side) const { return mGrid3D[side].getInvSpacingPhi(); }

  /// \return returns minimum r coordinate up to distortions and corrections are being calculated
  DataT getRMinSim(const Side side) const { return getRMin(side) - 4 * getGridSpacingR(side); }

  /// \return returns maximum r coordinate up to distortions and corrections are being calculated
  DataT getRMaxSim(const Side side) const { return getRMax(side) + 2 * getGridSpacingR(side); }

  std::string getSideName(const Side side) const { return side == Side::A ? "A" : "C"; }

  /// calculate distortions or corrections analytical with electric fields
  template <typename Fields = AnalyticalFields<DataT>>
  void calcDistCorr(const DataT p1r, const DataT p1phi, const DataT p1z, const DataT p2z, DataT& ddR, DataT& ddPhi, DataT& ddZ, const Fields& formulaStruct, const bool localDistCorr) const;

  /// calculate distortions/corrections using the formulas proposed in https://edms.cern.ch/ui/file/1108138/1/ALICE-INT-2010-016.pdf page 7
  void langevinCylindrical(DataT& ddR, DataT& ddPhi, DataT& ddZ, const DataT radius, const DataT localIntErOverEz, const DataT localIntEPhiOverEz, const DataT localIntDeltaEz) const;

  /// integrate electrical fields using root integration method
  template <typename Fields = AnalyticalFields<DataT>>
  void integrateEFieldsRoot(const DataT p1r, const DataT p1phi, const DataT p1z, const DataT p2z, DataT& localIntErOverEz, DataT& localIntEPhiOverEz, DataT& localIntDeltaEz, const Fields& formulaStruct) const;

  /// integrate electrical fields using trapezoidal integration method
  template <typename Fields = AnalyticalFields<DataT>>
  void integrateEFieldsTrapezoidal(const DataT p1r, const DataT p1phi, const DataT p1z, const DataT p2z, DataT& localIntErOverEz, DataT& localIntEPhiOverEz, DataT& localIntDeltaEz, const Fields& formulaStruct) const;

  /// integrate electrical fields using simpson integration method
  template <typename Fields = AnalyticalFields<DataT>>
  void integrateEFieldsSimpson(const DataT p1r, const DataT p1phi, const DataT p1z, const DataT p2z, DataT& localIntErOverEz, DataT& localIntEPhiOverEz, DataT& localIntDeltaEz, const Fields& formulaStruct) const;

  /// integrate electrical fields using simpson integration method with non straight drift of electrons
  template <typename Fields = AnalyticalFields<DataT>>
  void integrateEFieldsSimpsonIterative(const DataT p1r, const DataT p2r, const DataT p1phi, const DataT p2phi, const DataT p1z, const DataT p2z, DataT& localIntErOverEz, DataT& localIntEPhiOverEz, DataT& localIntDeltaEz, const Fields& formulaStruct) const;

  /// calculate distortions/corrections using analytical electric fields
  void processGlobalDistCorr(const DataT radius, const DataT phi, const DataT z0Tmp, const DataT z1Tmp, DataT& ddR, DataT& ddPhi, DataT& ddZ, const AnalyticalFields<DataT>& formulaStruct) const { calcDistCorr(radius, phi, z0Tmp, z1Tmp, ddR, ddPhi, ddZ, formulaStruct, false); }

  /// calculate distortions/corrections using electric fields from tricubic interpolator
  void processGlobalDistCorr(const DataT radius, const DataT phi, const DataT z0Tmp, const DataT z1Tmp, DataT& ddR, DataT& ddPhi, DataT& ddZ, const NumericalFields<DataT>& formulaStruct) const { calcDistCorr(radius, phi, z0Tmp, z1Tmp, ddR, ddPhi, ddZ, formulaStruct, false); }

  /// calculate distortions/corrections by interpolation of local distortions/corrections
  void processGlobalDistCorr(const DataT radius, const DataT phi, const DataT z0Tmp, [[maybe_unused]] const DataT z1Tmp, DataT& ddR, DataT& ddPhi, DataT& ddZ, const DistCorrInterpolator<DataT>& localDistCorr) const
  {
    ddR = localDistCorr.evaldR(z0Tmp, radius, phi);
    ddZ = localDistCorr.evaldZ(z0Tmp, radius, phi);
    ddPhi = localDistCorr.evaldRPhi(z0Tmp, radius, phi) / radius;
  }

  /// dump the created electron tracks with calculateElectronDriftPath function to a tree
  void dumpElectronTracksToTree(const std::vector<std::pair<std::vector<o2::math_utils::Point3D<float>>, std::array<DataT, 3>>>& electronTracks, const int nSamplingPoints, const char* outFile) const;

  /// \return returns nearest phi vertex for given phi position
  size_t getNearestPhiVertex(const DataT phi, const Side side) const { return std::round(phi / getGridSpacingPhi(side)); }

  /// \return returns nearest r vertex for given radius position
  size_t getNearestRVertex(const DataT r, const Side side) const { return std::round((r - getRMin(side)) / getGridSpacingR(side)); }

  /// \return returns number of bins in phi direction for the gap between sectors and for the GEM frame
  size_t getPhiBinsGapFrame(const Side side) const;

  /// \return setting the boundary potential for given GEM stack
  void setPotentialBoundaryGEMFrameAlongPhi(const std::function<DataT(DataT)>& potentialFunc, const GEMstack stack, const bool bottom, const Side side, const bool outerFrame = false);

  void getDistortionsCorrectionsAnalytical(const DataT x, const DataT y, const DataT z, const Side side, DataT& distX, DataT& distY, DataT& distZ, const bool dist) const;

  ClassDefNV(SpaceCharge, 3);
};

///
/// ========================================================================================================
///                                Inline implementations of some methods
/// ========================================================================================================
///

template <typename DataT>
template <typename Fields>
void SpaceCharge<DataT>::integrateEFieldsTrapezoidal(const DataT p1r, const DataT p1phi, const DataT p1z, const DataT p2z, DataT& localIntErOverEz, DataT& localIntEPhiOverEz, DataT& localIntDeltaEz, const Fields& formulaStruct) const
{
  //========trapezoidal rule see: https://en.wikipedia.org/wiki/Trapezoidal_rule ==============
  const DataT fielder0 = formulaStruct.evalEr(p1z, p1r, p1phi);
  const DataT fieldez0 = formulaStruct.evalEz(p1z, p1r, p1phi);
  const DataT fieldephi0 = formulaStruct.evalEphi(p1z, p1r, p1phi);

  const DataT fielder1 = formulaStruct.evalEr(p2z, p1r, p1phi);
  const DataT fieldez1 = formulaStruct.evalEz(p2z, p1r, p1phi);
  const DataT fieldephi1 = formulaStruct.evalEphi(p2z, p1r, p1phi);

  const DataT ezField = getEzField(formulaStruct.getSide());
  const DataT eZ0 = isCloseToZero(ezField, fieldez0) ? 0 : 1. / (ezField + fieldez0);
  const DataT eZ1 = isCloseToZero(ezField, fieldez1) ? 0 : 1. / (ezField + fieldez1);

  const DataT deltaX = 0.5 * (p2z - p1z);
  localIntErOverEz = deltaX * (fielder0 * eZ0 + fielder1 * eZ1);
  localIntEPhiOverEz = deltaX * (fieldephi0 * eZ0 + fieldephi1 * eZ1);
  localIntDeltaEz = getSign(formulaStruct.getSide()) * deltaX * (fieldez0 + fieldez1);
}

template <typename DataT>
template <typename Fields>
void SpaceCharge<DataT>::integrateEFieldsSimpson(const DataT p1r, const DataT p1phi, const DataT p1z, const DataT p2z, DataT& localIntErOverEz, DataT& localIntEPhiOverEz, DataT& localIntDeltaEz, const Fields& formulaStruct) const
{
  //==========simpsons rule see: https://en.wikipedia.org/wiki/Simpson%27s_rule =============================
  const DataT fielder0 = formulaStruct.evalEr(p1z, p1r, p1phi);
  const DataT fieldez0 = formulaStruct.evalEz(p1z, p1r, p1phi);
  const DataT fieldephi0 = formulaStruct.evalEphi(p1z, p1r, p1phi);

  const DataT fielder1 = formulaStruct.evalEr(p2z, p1r, p1phi);
  const DataT fieldez1 = formulaStruct.evalEz(p2z, p1r, p1phi);
  const DataT fieldephi1 = formulaStruct.evalEphi(p2z, p1r, p1phi);

  const DataT deltaX = p2z - p1z;
  const DataT ezField = getEzField(formulaStruct.getSide());
  const DataT xk2N = (p2z - static_cast<DataT>(0.5) * deltaX);
  const DataT ezField2 = formulaStruct.evalEz(xk2N, p1r, p1phi);
  const DataT ezField2Denominator = isCloseToZero(ezField, ezField2) ? 0 : 1. / (ezField + ezField2);
  const DataT fieldSum2ErOverEz = formulaStruct.evalEr(xk2N, p1r, p1phi) * ezField2Denominator;
  const DataT fieldSum2EphiOverEz = formulaStruct.evalEphi(xk2N, p1r, p1phi) * ezField2Denominator;

  const DataT eZ0 = isCloseToZero(ezField, fieldez0) ? 0 : 1. / (ezField + fieldez0);
  const DataT eZ1 = isCloseToZero(ezField, fieldez1) ? 0 : 1. / (ezField + fieldez1);

  const DataT deltaXSimpsonSixth = deltaX / 6.;
  localIntErOverEz = deltaXSimpsonSixth * (4. * fieldSum2ErOverEz + fielder0 * eZ0 + fielder1 * eZ1);
  localIntEPhiOverEz = deltaXSimpsonSixth * (4. * fieldSum2EphiOverEz + fieldephi0 * eZ0 + fieldephi1 * eZ1);
  localIntDeltaEz = getSign(formulaStruct.getSide()) * deltaXSimpsonSixth * (4. * ezField2 + fieldez0 + fieldez1);
}

template <typename DataT>
template <typename Fields>
void SpaceCharge<DataT>::integrateEFieldsSimpsonIterative(const DataT p1r, const DataT p2r, const DataT p1phi, const DataT p2phi, const DataT p1z, const DataT p2z, DataT& localIntErOverEz, DataT& localIntEPhiOverEz, DataT& localIntDeltaEz, const Fields& formulaStruct) const
{
  //==========simpsons rule see: https://en.wikipedia.org/wiki/Simpson%27s_rule =============================
  const Side side = formulaStruct.getSide();
  const DataT ezField = getEzField(side);
  const DataT p2phiSave = regulatePhi(p2phi, side);

  const DataT fielder0 = formulaStruct.evalEr(p1z, p1r, p1phi);
  const DataT fieldez0 = formulaStruct.evalEz(p1z, p1r, p1phi);
  const DataT fieldephi0 = formulaStruct.evalEphi(p1z, p1r, p1phi);

  const DataT fielder1 = formulaStruct.evalEr(p2z, p2r, p2phiSave);
  const DataT fieldez1 = formulaStruct.evalEz(p2z, p2r, p2phiSave);
  const DataT fieldephi1 = formulaStruct.evalEphi(p2z, p2r, p2phiSave);

  const DataT eZ0Inv = isCloseToZero(ezField, fieldez0) ? 0 : 1. / (ezField + fieldez0);
  const DataT eZ1Inv = isCloseToZero(ezField, fieldez1) ? 0 : 1. / (ezField + fieldez1);

  const DataT pHalfZ = 0.5 * (p1z + p2z);                              // dont needs to be regulated since p1z and p2z are already regulated
  const DataT pHalfPhiSave = regulatePhi(0.5 * (p1phi + p2phi), side); // needs to be regulated since p2phi is not regulated
  const DataT pHalfR = 0.5 * (p1r + p2r);

  const DataT ezField2 = formulaStruct.evalEz(pHalfZ, pHalfR, pHalfPhiSave);
  const DataT eZHalfInv = (isCloseToZero(ezField, ezField2) | isCloseToZero(ezField, fieldez0) | isCloseToZero(ezField, fieldez1)) ? 0 : 1. / (ezField + ezField2);
  const DataT fieldSum2ErOverEz = formulaStruct.evalEr(pHalfZ, pHalfR, pHalfPhiSave);
  const DataT fieldSum2EphiOverEz = formulaStruct.evalEphi(pHalfZ, pHalfR, pHalfPhiSave);

  const DataT deltaXSimpsonSixth = (p2z - p1z) / 6;
  localIntErOverEz = deltaXSimpsonSixth * (4 * fieldSum2ErOverEz * eZHalfInv + fielder0 * eZ0Inv + fielder1 * eZ1Inv);
  localIntEPhiOverEz = deltaXSimpsonSixth * (4 * fieldSum2EphiOverEz * eZHalfInv + fieldephi0 * eZ0Inv + fieldephi1 * eZ1Inv);
  localIntDeltaEz = getSign(side) * deltaXSimpsonSixth * (4 * ezField2 + fieldez0 + fieldez1);
}

template <typename DataT>
template <typename Fields>
void SpaceCharge<DataT>::calcDistCorr(const DataT p1r, const DataT p1phi, const DataT p1z, const DataT p2z, DataT& ddR, DataT& ddPhi, DataT& ddZ, const Fields& formulaStruct, const bool localDistCorr) const
{
  // see: https://edms.cern.ch/ui/file/1108138/1/ALICE-INT-2010-016.pdf
  // needed for calculation of distortions/corrections
  DataT localIntErOverEz = 0;   // integral_p1z^p2z Er/Ez dz
  DataT localIntEPhiOverEz = 0; // integral_p1z^p2z Ephi/Ez dz
  DataT localIntDeltaEz = 0;    // integral_p1z^p2z Ez dz

  // there are differentnumerical integration strategys implements. for details see each function.
  switch (sNumericalIntegrationStrategy) {
    case IntegrationStrategy::SimpsonIterative:                                                         // iterative simpson integration (should be more precise at least for the analytical E-Field case but takes alot more time than normal simpson integration)
      for (int i = 0; i < sSimpsonNIteratives; ++i) {                                                   // TODO define a convergence criterion to abort the algorithm earlier for speed up.
        const DataT tmpZ = localDistCorr ? (p2z + ddZ) : regulateZ(p2z + ddZ, formulaStruct.getSide()); // dont regulate for local distortions/corrections! (to get same result as using electric field at last/first bin)
        integrateEFieldsSimpsonIterative(p1r, p1r + ddR, p1phi, p1phi + ddPhi, p1z, tmpZ, localIntErOverEz, localIntEPhiOverEz, localIntDeltaEz, formulaStruct);
        langevinCylindrical(ddR, ddPhi, ddZ, (p1r + 0.5 * ddR), localIntErOverEz, localIntEPhiOverEz, localIntDeltaEz); // using the mean radius '(p1r + 0.5 * ddR)' for calculation of distortions/corections
      }
      break;
    case IntegrationStrategy::Simpson: // simpson integration
      integrateEFieldsSimpson(p1r, p1phi, p1z, p2z, localIntErOverEz, localIntEPhiOverEz, localIntDeltaEz, formulaStruct);
      langevinCylindrical(ddR, ddPhi, ddZ, p1r, localIntErOverEz, localIntEPhiOverEz, localIntDeltaEz);
      break;
    case IntegrationStrategy::Trapezoidal: // trapezoidal integration (fastest)
      integrateEFieldsTrapezoidal(p1r, p1phi, p1z, p2z, localIntErOverEz, localIntEPhiOverEz, localIntDeltaEz, formulaStruct);
      langevinCylindrical(ddR, ddPhi, ddZ, p1r, localIntErOverEz, localIntEPhiOverEz, localIntDeltaEz);
      break;
    case IntegrationStrategy::Root: // using integration implemented in ROOT (slow)
      integrateEFieldsRoot(p1r, p1phi, p1z, p2z, localIntErOverEz, localIntEPhiOverEz, localIntDeltaEz, formulaStruct);
      langevinCylindrical(ddR, ddPhi, ddZ, p1r, localIntErOverEz, localIntEPhiOverEz, localIntDeltaEz);
      break;
    default:
      integrateEFieldsSimpson(p1r, p1phi, p1z, p2z, localIntErOverEz, localIntEPhiOverEz, localIntDeltaEz, formulaStruct);
      langevinCylindrical(ddR, ddPhi, ddZ, p1r, localIntErOverEz, localIntEPhiOverEz, localIntDeltaEz);
  }
}

template <typename DataT>
void SpaceCharge<DataT>::langevinCylindrical(DataT& ddR, DataT& ddPhi, DataT& ddZ, const DataT radius, const DataT localIntErOverEz, const DataT localIntEPhiOverEz, const DataT localIntDeltaEz) const
{
  // calculated distortions/correction with the formula described in https://edms.cern.ch/ui/file/1108138/1/ALICE-INT-2010-016.pdf page 7.
  ddR = mC0 * localIntErOverEz + mC1 * localIntEPhiOverEz;
  ddPhi = (mC0 * localIntEPhiOverEz - mC1 * localIntErOverEz) / radius;
  ddZ = -localIntDeltaEz * TPCParameters<DataT>::DVDE;
}

template <typename DataT>
template <typename DataTOut>
int SpaceCharge<DataT>::dumpElectricFields(TFile& outf, const Side side) const
{
  if (!mIsEfieldSet[side]) {
    LOGP(info, "============== E-Fields are not set! returning ==============\n");
    return 0;
  }
  const std::string sideName = getSideName(side);
  const int er = mElectricFieldEr[side].template writeToFile<DataTOut>(outf, fmt::format("fieldEr_side{}", sideName).data());
  const int ez = mElectricFieldEz[side].template writeToFile<DataTOut>(outf, fmt::format("fieldEz_side{}", sideName).data());
  const int ephi = mElectricFieldEphi[side].template writeToFile<DataTOut>(outf, fmt::format("fieldEphi_side{}", sideName).data());
  return er + ez + ephi;
}

template <typename DataT>
template <typename DataTIn>
void SpaceCharge<DataT>::setElectricFieldsFromFile(TFile& inpf, const Side side)
{
  const std::string sideName = getSideName(side);
  mElectricFieldEr[side].template initFromFile<DataTIn>(inpf, fmt::format("fieldEr_side{}", sideName).data());
  mElectricFieldEz[side].template initFromFile<DataTIn>(inpf, fmt::format("fieldEz_side{}", sideName).data());
  mElectricFieldEphi[side].template initFromFile<DataTIn>(inpf, fmt::format("fieldEphi_side{}", sideName).data());
  mIsEfieldSet[side] = true;
}

template <typename DataT>
template <typename DataTOut>
int SpaceCharge<DataT>::dumpGlobalDistortions(TFile& outf, const Side side) const
{
  if (!mIsGlobalDistSet[side]) {
    LOGP(info, "============== global distortions are not set! returning ==============\n");
    return 0;
  }
  const std::string sideName = getSideName(side);
  const int er = mGlobalDistdR[side].template writeToFile<DataTOut>(outf, fmt::format("distR_side{}", sideName).data());
  const int ez = mGlobalDistdZ[side].template writeToFile<DataTOut>(outf, fmt::format("distZ_side{}", sideName).data());
  const int ephi = mGlobalDistdRPhi[side].template writeToFile<DataTOut>(outf, fmt::format("distRphi_side{}", sideName).data());
  return er + ez + ephi;
}

template <typename DataT>
template <typename DataTIn>
void SpaceCharge<DataT>::setGlobalDistortionsFromFile(TFile& inpf, const Side side)
{
  mIsGlobalDistSet[side] = true;
  const std::string sideName = getSideName(side);
  mGlobalDistdR[side].template initFromFile<DataTIn>(inpf, fmt::format("distR_side{}", sideName).data());
  mGlobalDistdZ[side].template initFromFile<DataTIn>(inpf, fmt::format("distZ_side{}", sideName).data());
  mGlobalDistdRPhi[side].template initFromFile<DataTIn>(inpf, fmt::format("distRphi_side{}", sideName).data());
}

template <typename DataT>
template <typename DataTOut>
int SpaceCharge<DataT>::dumpGlobalCorrections(TFile& outf, const Side side) const
{
  if (!mIsGlobalCorrSet[side]) {
    LOGP(info, "============== global corrections are not set! returning ==============\n");
    return 0;
  }
  const std::string sideName = getSideName(side);
  const int er = mGlobalCorrdR[side].template writeToFile<DataTOut>(outf, fmt::format("corrR_side{}", sideName).data());
  const int ez = mGlobalCorrdZ[side].template writeToFile<DataTOut>(outf, fmt::format("corrZ_side{}", sideName).data());
  const int ephi = mGlobalCorrdRPhi[side].template writeToFile<DataTOut>(outf, fmt::format("corrRPhi_side{}", sideName).data());
  return er + ez + ephi;
}

template <typename DataT>
template <typename DataTIn>
void SpaceCharge<DataT>::setGlobalCorrectionsFromFile(TFile& inpf, const Side side)
{
  mIsGlobalCorrSet[side] = true;
  const std::string sideName = getSideName(side);
  mGlobalCorrdR[side].template initFromFile<DataTIn>(inpf, fmt::format("corrR_side{}", sideName).data());
  mGlobalCorrdZ[side].template initFromFile<DataTIn>(inpf, fmt::format("corrZ_side{}", sideName).data());
  mGlobalCorrdRPhi[side].template initFromFile<DataTIn>(inpf, fmt::format("corrRPhi_side{}", sideName).data());
}

template <typename DataT>
template <typename DataTOut>
int SpaceCharge<DataT>::dumpLocalCorrections(TFile& outf, const Side side) const
{
  if (!mIsLocalCorrSet[side]) {
    LOGP(info, "============== local corrections are not set! returning ==============\n");
    return 0;
  }
  const std::string sideName = getSideName(side);
  const int lCorrdR = mLocalCorrdR[side].template writeToFile<DataTOut>(outf, fmt::format("lcorrR_side{}", sideName).data());
  const int lCorrdZ = mLocalCorrdZ[side].template writeToFile<DataTOut>(outf, fmt::format("lcorrZ_side{}", sideName).data());
  const int lCorrdRPhi = mLocalCorrdRPhi[side].template writeToFile<DataTOut>(outf, fmt::format("lcorrRPhi_side{}", sideName).data());
  return lCorrdR + lCorrdZ + lCorrdRPhi;
}

template <typename DataT>
template <typename DataTIn>
void SpaceCharge<DataT>::setLocalCorrectionsFromFile(TFile& inpf, const Side side)
{
  const std::string sideName = getSideName(side);
  const bool lCorrdR = mLocalCorrdR[side].template initFromFile<DataTIn>(inpf, fmt::format("lcorrR_side{}", sideName).data());
  const bool lCorrdZ = mLocalCorrdZ[side].template initFromFile<DataTIn>(inpf, fmt::format("lcorrZ_side{}", sideName).data());
  const bool lCorrdRPhi = mLocalCorrdRPhi[side].template initFromFile<DataTIn>(inpf, fmt::format("lcorrRPhi_side{}", sideName).data());
  if (lCorrdR && lCorrdZ && lCorrdRPhi) {
    mIsLocalCorrSet[side] = true;
  } else {
    mIsLocalCorrSet[side] = false;
  }
}

template <typename DataT>
template <typename DataTOut>
int SpaceCharge<DataT>::dumpLocalDistortions(TFile& outf, const Side side) const
{
  if (!mIsLocalDistSet[side]) {
    LOGP(info, "============== local distortions are not set! returning ==============\n");
    return 0;
  }
  const std::string sideName = getSideName(side);
  const int lDistdR = mLocalDistdR[side].template writeToFile<DataTOut>(outf, fmt::format("ldistR_side{}", sideName).data());
  const int lDistdZ = mLocalDistdZ[side].template writeToFile<DataTOut>(outf, fmt::format("ldistZ_side{}", sideName).data());
  const int lDistdRPhi = mLocalDistdRPhi[side].template writeToFile<DataTOut>(outf, fmt::format("ldistRPhi_side{}", sideName).data());
  return lDistdR + lDistdZ + lDistdRPhi;
}

template <typename DataT>
template <typename DataTOut>
int SpaceCharge<DataT>::dumpLocalDistCorrVectors(TFile& outf, const Side side) const
{
  if (!mIsLocalVecDistSet[side]) {
    LOGP(info, "============== local distortion vectors are not set! returning ==============\n");
    return 0;
  }
  const std::string sideName = getSideName(side);
  const int lVecDistdR = mLocalVecDistdR[side].template writeToFile<DataTOut>(outf, fmt::format("lvecdistR_side{}", sideName).data());
  const int lVecDistdZ = mLocalVecDistdZ[side].template writeToFile<DataTOut>(outf, fmt::format("lvecdistZ_side{}", sideName).data());
  const int lVecDistdRPhi = mLocalVecDistdRPhi[side].template writeToFile<DataTOut>(outf, fmt::format("lvecdistRPhi_side{}", sideName).data());
  return lVecDistdR + lVecDistdZ + lVecDistdRPhi;
}

template <typename DataT>
template <typename DataTIn>
void SpaceCharge<DataT>::setLocalDistortionsFromFile(TFile& inpf, const Side side)
{
  const std::string sideName = getSideName(side);
  const bool lDistdR = mLocalDistdR[side].template initFromFile<DataTIn>(inpf, fmt::format("ldistR_side{}", sideName).data());
  const bool lDistdZ = mLocalDistdZ[side].template initFromFile<DataTIn>(inpf, fmt::format("ldistZ_side{}", sideName).data());
  const bool lDistdRPhi = mLocalDistdRPhi[side].template initFromFile<DataTIn>(inpf, fmt::format("ldistRPhi_side{}", sideName).data());

  if (lDistdR && lDistdZ && lDistdRPhi) {
    mIsLocalDistSet[side] = true;
  } else {
    mIsLocalDistSet[side] = false;
  }
}

template <typename DataT>
template <typename DataTIn>
void SpaceCharge<DataT>::setLocalDistCorrVectorsFromFile(TFile& inpf, const Side side)
{
  const std::string sideName = getSideName(side);
  const bool lVecDistdR = mLocalVecDistdR[side].template initFromFile<DataTIn>(inpf, fmt::format("lvecdistR_side{}", sideName).data());
  const bool lVecDistdZ = mLocalVecDistdZ[side].template initFromFile<DataTIn>(inpf, fmt::format("lvecdistZ_side{}", sideName).data());
  const bool lVecDistdRPhi = mLocalVecDistdRPhi[side].template initFromFile<DataTIn>(inpf, fmt::format("lvecdistRPhi_side{}", sideName).data());

  if (lVecDistdR && lVecDistdZ && lVecDistdRPhi) {
    mIsLocalVecDistSet[side] = true;
  } else {
    mIsLocalVecDistSet[side] = false;
  }
}

template <typename DataT>
template <typename DataTOut>
void SpaceCharge<DataT>::dumpToFile(TFile& file, const Side side) const
{
  dumpElectricFields<DataTOut>(file, side);
  dumpPotential<DataTOut>(file, side);
  dumpDensity<DataTOut>(file, side);
  dumpGlobalDistortions<DataTOut>(file, side);
  dumpGlobalCorrections<DataTOut>(file, side);
  dumpLocalCorrections<DataTOut>(file, side);
  dumpLocalDistortions<DataTOut>(file, side);
  dumpLocalDistCorrVectors<DataTOut>(file, side);
}

template <typename DataT>
template <typename DataTIn>
void SpaceCharge<DataT>::setFromFile(TFile& file, const Side side)
{
  setDensityFromFile<DataTIn>(file, side);
  setPotentialFromFile<DataTIn>(file, side);
  setElectricFieldsFromFile<DataTIn>(file, side);
  setLocalDistortionsFromFile<DataTIn>(file, side);
  setLocalCorrectionsFromFile<DataTIn>(file, side);
  setGlobalDistortionsFromFile<DataTIn>(file, side);
  setGlobalCorrectionsFromFile<DataTIn>(file, side);
  setLocalDistCorrVectorsFromFile<DataTIn>(file, side);
}

} // namespace tpc
} // namespace o2

#endif
