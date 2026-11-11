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

/// \file PoissonSolver.h
/// \brief This class provides implementation of Poisson equation
/// solver by MultiGrid Method
/// Original version of this class can be found in AliTPCPoissonSolver.h
///
///
/// \author  Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
///          Rifki Sadikin <rifki.sadikin@cern.ch> (original code in AliRoot in AliTPCPoissonSolver.h)
/// \date Aug 21, 2020

#ifndef ALICEO2_TPC_POISSONSOLVER_H_
#define ALICEO2_TPC_POISSONSOLVER_H_

#include "TPCSpaceCharge/RegularGrid3D.h"
#include "CommonConstants/MathConstants.h"
#include "TPCSpaceCharge/SpaceChargeParameter.h"
#include <vector>
#include <type_traits>

namespace o2
{
namespace tpc
{

template <typename DataT>
class Vector3D;

template <typename DataT>
class DataContainer3D;

/// \class PoissonSolver
/// The PoissonSolver class represents methods to solve the poisson equation.
/// Original version with more methods can be found in AliTPCPoissonSolver.
/// Implemented: full multi grid (F cycle of V cycles) in 3D with red-black Gauss-Seidel relaxation, full weighting grid transfer and periodic boundary in phi.
/// The phi direction is coarsened only if it is not weakly coupled (MGParameters::phiCoarseningThreshold). On the levels where phi is strongly coupled (fine grid in phi)
/// line relaxation in phi is used instead of the point relaxation (MGParameters::phiLineRelaxThreshold).

/// \tparam DataT the type of data which is used during the calculations
template <typename DataT = double>
class PoissonSolver
{
 public:
  using RegularGrid = RegularGrid3D<DataT>;
  using DataContainer = DataContainer3D<DataT>;
  using Vector = Vector3D<DataT>;

  /// default constructor
  PoissonSolver(const RegularGrid& gridProperties) : mGrid3D{gridProperties} {};

  /// Provides poisson solver in Cylindrical 3D (TPC geometry)
  ///
  /// Geometric full multi grid (F cycle of V cycles) with red-black Gauss-Seidel relaxation, full weighting grid transfer and periodic boundary in phi.
  /// The parameters are set in MGParameters
  ///
  /// \param matricesV potential in 3D
  /// \param matricesCharge charge density in 3D
  ///
  /// \pre Charge density distribution in **matricesCharge** is known and boundary values for **matricesV** are set
  /// \post Numerical solution for potential distribution is calculated and stored in each rod at **matricesV**
  void poissonSolver3D(DataContainer& matricesV, const DataContainer& matricesCharge);

  DataT getSpacingZ() const { return mGrid3D.getSpacingZ(); }
  DataT getSpacingR() const { return mGrid3D.getSpacingR(); }
  DataT getSpacingPhi() const { return mGrid3D.getSpacingPhi(); }

  /// set the stopping criterion of the multi grid V cycles. For MGParameters::relativeConvergence (default): maximum relative change sqrt(sum dV^2 / sum V^2) of the potential between two V cycles
  static void setConvergenceError(const DataT error) { sConvergenceError = error; }

  static DataT getConvergenceError() { return sConvergenceError; }

  /// default stopping criterion: relative change of 1e-8 for double, 1e-6 for float (the relative change of float stays at ~1e-7 because of the rounding)
  static constexpr DataT getDefaultConvergenceError() { return std::is_same_v<DataT, float> ? DataT(1e-6) : DataT(1e-8); }

  /// get the number of threads used for some of the calculations
  static int getNThreads() { return sNThreads; }

  /// set the number of threads used for some of the calculations
  static void setNThreads(int nThreads) { sNThreads = nThreads; }

 private:
  const RegularGrid& mGrid3D{};                                      ///< grid properties
  const ParamSpaceCharge mParamGrid{mGrid3D.getParamSC()};           ///< parameters of the grid on which the calculations are performed
  inline static DataT sConvergenceError{getDefaultConvergenceError()}; ///< stopping criterion of the V cycles (see MGParameters::relativeConvergence)
  static constexpr DataT INVTWOPI = 1. / o2::constants::math::TwoPI; ///< inverse of 2*pi
  inline static int sNThreads{4};                                    ///< number of threads used for the relaxation, residue, restriction, interpolation and convergence calculation on the larger multi grid levels (set by SpaceCharge::poissonSolver() to SpaceCharge::getNThreads())
  static constexpr long MINPOINTSPARALLEL{20000};                    ///< minimum number of grid points of a multi grid level for which the calculations are performed in parallel

  /// \returns inverse grid size in phi (either 1/2Pi or NSECTORSPERSIDE/2Pi)
  static DataT getGridSizePhiInv();

  /// \return returns the change of the potential between two V cycles (stopping criterion of the multi grid):
  /// MGParameters::relativeConvergence: sqrt(sum dV^2 / sum V^2), otherwise the maximum of the sum of dV^2 of the phi slices
  ///
  /// \param matricesCurrentV current potential (numerical solution)
  /// \param prevArrayV content from matricesCurrentV from previous iteration
  DataT getConvergenceError(const Vector& matricesCurrentV, const Vector& prevArrayV) const;

  /// grid of one level of the multi grid
  struct MGLevel {
    int nr{};            ///< number of vertices in r
    int nz{};            ///< number of vertices in z
    int nphi{};          ///< number of vertices in phi
    int rzFactor{};      ///< grid spacing in r and z in units of the grid spacing of the finest grid
    bool lineRelaxPhi{}; ///< use line relaxation in phi (phi strongly coupled, see MGParameters::phiLineRelaxThreshold) instead of point relaxation
  };

  /// factorisation of the periodic tridiagonal systems of the line relaxation in phi of one level (Thomas algorithm and Sherman-Morrison correction).
  /// The matrix depends only on the radial index i (and on the level): storage of the vectors with n = nphi entries per i: index = i + nr * m
  struct PhiLineFactorisation {
    std::vector<DataT> cPrime{};     ///< modified super diagonal of the Thomas algorithm
    std::vector<DataT> invDen{};     ///< inverse of the modified diagonal of the Thomas algorithm
    std::vector<DataT> zSM{};        ///< solution of T z = u for the Sherman-Morrison correction
    std::vector<DataT> factDenInv{}; ///< 1 / (1 + z_0 + b/a * z_{n-1}) for each i
    std::vector<DataT> bOverA{};     ///< b / a for each i
    std::vector<DataT> bCoeff{};     ///< b (coupling in phi) for each i
  };

  /// set the coefficients of the relaxation for given level
  /// \return returns h^2 with h the grid spacing in r of the level
  DataT setLevelCoefficients(const MGLevel& level, const DataT ratioZ, std::vector<DataT>& coefficient1, std::vector<DataT>& coefficient2, std::vector<DataT>& coefficient3, std::vector<DataT>& coefficient4) const;

  /// \return returns the factorisation of the periodic tridiagonal systems of the line relaxation in phi for given level and coefficients of the level
  PhiLineFactorisation factorisePhiLines(const MGLevel& level, const std::vector<DataT>& coefficient3, const std::vector<DataT>& coefficient4) const;

  /// relaxation of the given level: line relaxation in phi (MGLevel::lineRelaxPhi) with the factorisation phiLineFact of the level or point relaxation (relax3D())
  void relaxLevel(const MGLevel& level, const PhiLineFactorisation& phiLineFact, Vector& matricesCurrentV, const Vector& matricesCurrentCharge, const DataT h2, const DataT tempRatioZ,
                  const std::vector<DataT>& coefficient1, const std::vector<DataT>& coefficient2, const std::vector<DataT>& coefficient3, const std::vector<DataT>& coefficient4) const;

  /// red-black Gauss-Seidel line relaxation in phi direction: the potential of all phi slices on one line (r, z) is solved exactly (periodic tridiagonal system)
  /// for given potential on the neighbouring lines. The lines with (ir + iz) even and odd are relaxed one after the other.
  /// Robust if the phi direction is strongly coupled compared to r and z (fine grid in phi), where the point relaxation smooths the error badly.
  /// \param phiLineFact factorisation of the periodic tridiagonal systems of the level (see factorisePhiLines())
  void relaxLinePhi(Vector& matricesCurrentV, const Vector& matricesCurrentCharge, const int tnRRow, const int tnZColumn, const int iPhi, const DataT h2, const DataT tempRatioZ,
                    const std::vector<DataT>& coefficient1, const std::vector<DataT>& coefficient2, const PhiLineFactorisation& phiLineFact) const;

  /// \returns the grids of the multi grid levels. From one level to the next the r and z direction are coarsened and the phi direction is coarsened (if the number of slices is even)
  /// only if it is not weakly coupled compared to r and z (see MGParameters::phiCoarseningThreshold). Line relaxation in phi is used on the levels where phi is strongly coupled (see MGParameters::phiLineRelaxThreshold)
  /// \param nLevelsRZ number of levels for the coarsening in r and z
  std::vector<MGLevel> getMultiGridLevels(const int nLevelsRZ) const;

  /// \return returns the relative coupling (h_{r,z} / (radius * h_{phi}))^2 in phi direction compared to r and z for given grid and radius
  DataT getRelCouplingPhi(const MGLevel& level, const DataT radius) const;

  /// restriction of the fine level to the coarse level (full weighting)
  void restrictLevel(Vector& coarse, const Vector& fine, const MGLevel& levelCoarse, const MGLevel& levelFine) const;

  /// restriction of the boundary values of the fine level to the coarse level
  void restrictBoundaryLevel(Vector& coarse, const Vector& fine, const MGLevel& levelCoarse, const MGLevel& levelFine) const;

  /// interpolation of the coarse level to the fine level: set (Add = false) or add (Add = true) the interpolated values
  template <bool Add>
  void interpolateLevel(Vector& fine, const Vector& coarse, const MGLevel& levelFine, const MGLevel& levelCoarse) const;

  /// get the indices of the neighbouring phi slices for given phi slice (periodic boundary in phi)
  /// \param m phi slice
  /// \param nPhi number of phi slices
  /// \param mp1 index of the slice m + 1 (output)
  /// \param mm1 index of the slice m - 1 (output)
  static void getPhiNeighbours(const int m, const int nPhi, int& mp1, int& mm1);

  /// 3D - Solve Poisson's Equation in 3D in all direction by MultiGrid
  ///
  ///    NOTE: In order for this algorithm to work, the number of nRRow and nZColumn must be a power of 2 plus one.
  ///    The number of nRRow and Z Column can be different.
  ///
  ///    R Row       ==  2**M + 1
  ///    Z Column   ==  2**N + 1
  ///    Phi Slices  ==  Arbitrary but greater than 3
  ///
  ///    Solving: \f$  \nabla^{2}V(r,\phi,z) = - f(r,\phi,z) \f$
  ///
  ///  Algorithm for MultiGrid Full Cycle (FMG)
  /// - Restrict the charge density and the boundary potential to the coarser grids (see getMultiGridLevels())
  /// - Relax on the coarsest grid
  /// - Do from coarsest to finest
  ///   - Interpolate potential from coarse -> fine
  ///   - Do V-Cycles from the current level to the coarsest level until converged (see getConvergenceError())
  ///
  /// \param matricesV potential in 3D matrix (input: boundary potential, output: solution)
  /// \param matricesCharge charge density in 3D matrix
  void poissonMultiGrid3D(DataContainer& matricesV, const DataContainer& matricesCharge);

  /// Restrict2D
  ///
  ///    Grid transfer operator, restrict from fine -> coarse grid
  ///    provide full-half weighting
  ///
  ///    \[ \frac{1}{16}\left( \begin{array}{ccc}
  ///      1 & 2 & 1 \\
  ///      2 & 4 & 2 \\
  ///      1 & 2 & 1 \end{array} \right) \]
  ///
  /// \param matricesCurrentCharge coarse grid (2h)
  /// \param residue fine grid (h)
  /// \param tnRRow number of vertices in r direction of TPC
  /// \param tnZColumn number of vertices in z direction of TPC
  /// \param iphi phi vertex
  void restrict2D(Vector& matricesCurrentCharge, const Vector& residue, const int tnRRow, const int tnZColumn, const int iphi) const;

  /// Restriction in 3D
  ///
  /// Restriction is a map from fine grid (h) to coarse grid (2h)
  ///
  /// In case of 3D
  /// Full weighting:
  /// \f[ (R u)_{i,j,k} = \frac{1}{2} u_{2i,2j,2k} + \frac{1}{4} S_{1} + \frac{1}{8} S_{2} + \frac{1}{16} S_{3}\f]
  ///
  ///
  /// Restriction in all direction r-phi-z
  /// restriction in phi only if oldPhi == 2*newPhi
  /// \param matricesCurrentCharge coarser grid 2h
  /// \param residue fine grid h
  /// \param tnRRow number of grid in Nr (in r-direction) for coarser grid should be 2^N + 1, finer grid in 2^{N+1} + 1
  /// \param tnZColumn number of grid in Nz (in z-direction) for coarser grid should be  2^M + 1, finer grid in 2^{M+1} + 1
  /// \param newPhiSlice number of Nphi (in phi-direction) for coarser grid
  /// \param oldPhiSlice number of Nphi (in phi-direction) for finer grid
  void restrict3D(Vector& matricesCurrentCharge, const Vector& residue, const int tnRRow, const int tnZColumn, const int newPhiSlice, const int oldPhiSlice) const;

  /// Restrict Boundary in 3D
  ///
  /// Pass boundary information to coarse grid
  ///
  /// \param matricesCurrentCharge coarser grid 2h
  /// \param residue fine grid h
  /// \param tnRRow number of grid in Nr (in r-direction) for coarser grid should be 2^N + 1, finer grid in 2^{N+1} + 1
  /// \param tnZColumn number of grid in Nz (in z-direction) for coarser grid should be  2^M + 1, finer grid in 2^{M+1} + 1
  /// \param newPhiSlice number of Nphi (in phi-direction) for coarser grid
  /// \param oldPhiSlice number of Nphi (in phi-direction) for finer grid
  void restrictBoundary3D(Vector& matricesCurrentCharge, const Vector& residue, const int tnRRow, const int tnZColumn, const int newPhiSlice, const int oldPhiSlice) const;

  /// Relaxation operation for multiGrid
  ///   relaxation used 7 stencil in cylindrical coordinate
  ///
  /// Using the following equations
  /// \f$ U_{i,j,k} = (1 + \frac{1}{r_{i}h_{r}}) U_{i+1,j,k}  + (1 - \frac{1}{r_{i}h_{r}}) U_{i+1,j,k}  \f$
  ///
  /// \param matricesCurrentV potential in 3D (matrices of matrix)
  /// \param matricesCurrentCharge charge in 3D
  /// \param tnRRow number of grid in in r-direction for coarser grid should be 2^N + 1, finer grid in 2^{N+1} + 1
  /// \param tnZColumn number of grid in in z-direction for coarser grid should be  2^M + 1, finer grid in 2^{M+1} + 1
  /// \param iPhi phi vertex
  /// \param h2 \f$  h_{r}^{2} \f$
  /// \param tempRatioZ ration between grid size in z-direction and r-direction
  /// \param coefficient1 coefficients for \f$  V_{x+1,y,z} \f$
  /// \param coefficient2 coefficients for \f$  V_{x-1,y,z} \f$
  /// \param coefficient3 coefficients for z
  /// \param coefficient4 coefficients for f(r,\phi,z)
  void relax3D(Vector& matricesCurrentV, const Vector& matricesCurrentCharge, const int tnRRow, const int tnZColumn, const int iPhi, const DataT h2, const DataT tempRatioZ,
               const std::vector<DataT>& coefficient1, const std::vector<DataT>& coefficient2, const std::vector<DataT>& coefficient3, const std::vector<DataT>& coefficient4) const;

  /// Interpolation/prolongation from the coarse grid (2h) to the fine grid (h): set (Add = false) or add (Add = true) the interpolated values
  /// Interpolation in all directions r-phi-z, in phi only if newPhiSlice == 2 * oldPhiSlice
  /// \param matricesCurrentV fine grid h
  /// \param matricesCurrentVC coarse grid 2h
  /// \param tnRRow number of vertices in r-direction of the fine grid
  /// \param tnZColumn number of vertices in z-direction of the fine grid
  /// \param newPhiSlice number of vertices in phi-direction of the fine grid
  /// \param oldPhiSlice number of vertices in phi-direction of the coarse grid
  template <bool Add>
  void interpolate3D(Vector& matricesCurrentV, const Vector& matricesCurrentVC, const int tnRRow, const int tnZColumn, const int newPhiSlice, const int oldPhiSlice) const;

  /// Interpolation/prolongation in r and z from the coarse grid (2h) to the fine grid (h) for one phi slice: set (Add = false) or add (Add = true) the interpolated values
  /// \param iphi phi slice
  template <bool Add>
  void interpolate2D(Vector& matricesCurrentV, const Vector& matricesCurrentVC, const int tnRRow, const int tnZColumn, const int iphi) const;

  /// VCycle 3D, V Cycle in multiGrid, fine-->coarsest-->fine, propagating the residue to correct initial guess of V
  ///
  ///    NOTE: In order for this algorithm to work, the number of nRRow and nZColumn must be a power of 2 plus one.
  ///    The number of nRRow and Z Column can be different.
  ///
  ///    R Row       ==  2**M + 1
  ///    Z Column    ==  2**N + 1
  ///    Phi Slice  ==  Arbitrary but greater than 3
  ///
  ///    DeltaPhi in Radians
  ///
  /// \param gridFrom finest level of grid
  /// \param gridTo coarsest level of grid
  /// \param nPre number of smoothing before coarsening
  /// \param nPost number of smoothing after coarsening
  /// \param ratioZ ratio between square of grid r and grid z
  /// \param levels grids of the multi grid levels
  /// \param phiLineFact factorisation of the line relaxation in phi of the levels (empty for the levels without line relaxation)
  /// \param tvArrayV vector of V potential in different grids
  /// \param tvCharge vector of charge distribution in different grids
  /// \param tvResidue vector of residue calculation in different grids
  /// \param coefficient1 coefficient for relaxation (r direction)
  /// \param coefficient2 coefficient for relaxation (r direction)
  /// \param coefficient3 coefficient for relaxation (ratio r/z)
  /// \param coefficient4 coefficient for relaxation (ratio for grid_r)
  /// \param inverseCoefficient4 coefficient for relaxation (inverse coefficient4)
  void vCycle3D(const int gridFrom, const int gridTo, const int nPre, const int nPost, const DataT ratioZ, const std::vector<MGLevel>& levels, const std::vector<PhiLineFactorisation>& phiLineFact,
                std::vector<Vector>& tvArrayV, std::vector<Vector>& tvCharge, std::vector<Vector>& tvResidue, std::vector<DataT>& coefficient1, std::vector<DataT>& coefficient2,
                std::vector<DataT>& coefficient3, std::vector<DataT>& coefficient4, std::vector<DataT>& inverseCoefficient4) const;

  /// Residue3D
  ///
  ///    Compute residue from V(.) where V(.) is numerical potential and f(.).
  ///    residue used 7 stencil in cylindrical coordinate
  ///
  /// Using the following equations
  /// \f$ U_{i,j,k} = (1 + \frac{1}{r_{i}h_{r}}) U_{i+1,j,k}  + (1 - \frac{1}{r_{i}h_{r}}) U_{i+1,j,k}  \f$
  ///
  /// \param residue residue in 3D (matrices of matrix)
  /// \param matricesCurrentV potential in 3D (matrices of matrix)
  /// \param matricesCurrentCharge charge in 3D
  /// \param tnRRow number of vertices in the r direction of TPC
  /// \param tnZColumn number of vertices in z direction of TPC
  /// \param tnPhi number of vertices in phi direction of TPC
  /// \param ih2 \f$ 1/ h_{r}^{2} \f$
  /// \param tempRatioZ ration between grid size in z-direction and r-direction
  /// \param coefficient1 coefficient for \f$  V_{x+1,y,z} \f$
  /// \param coefficient2 coefficient for \f$  V_{x-1,y,z} \f$
  /// \param coefficient3 coefficient for z
  /// \param inverseCoefficient4 inverse coefficient for f(r,\phi,z)
  void residue3D(Vector& residue, const Vector& matricesCurrentV, const Vector& matricesCurrentCharge, const int tnRRow, const int tnZColumn, const int tnPhi, const DataT ih2, const DataT tempRatioZ,
                 const std::vector<DataT>& coefficient1, const std::vector<DataT>& coefficient2, const std::vector<DataT>& coefficient3, const std::vector<DataT>& inverseCoefficient4) const;

  ///    Boundary transfer  restrict from fine -> coarse grid
  ///
  /// \param matricesCurrentCharge coarse grid (2h)
  /// \param residue fine grid  (h)
  /// \param tnRRow number of vertices in the r direction of TPC
  /// \param tnZColumn number of vertices in z direction of TPC
  /// \param phiCoarse phi slice of the coarse grid
  /// \param phiFine phi slice of the fine grid
  void restrictBoundary2D(Vector& matricesCurrentCharge, const Vector& residue, const int tnRRow, const int tnZColumn, const int phiCoarse, const int phiFine) const;

  // calculate coefficients
  void calcCoefficients(unsigned int from, unsigned int to, const DataT h, const DataT tempRatioZ, const DataT tempRatioPhi, std::vector<DataT>& coefficient1,
                        std::vector<DataT>& coefficient2, std::vector<DataT>& coefficient3, std::vector<DataT>& coefficient4) const;

  /// Helper function to check if the integer is equal to a power of two
  /// \param i the number
  /// \return 1 if it is a power of two, else 0
  bool isPowerOfTwo(const int i) const
  {
    return ((i > 0) && !(i & (i - 1)));
  };
};

} // namespace tpc
} // namespace o2

#endif
