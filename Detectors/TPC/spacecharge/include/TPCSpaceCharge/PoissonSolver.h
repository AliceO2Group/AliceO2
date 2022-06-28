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
/// Following methods are implemented: poissonSolver3D, poissonSolver3D2D, poissonSolver2D

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
  /// Strategy based on parameter settings (mMgParameters)provided
  /// * Cascaded multi grid with S.O.R
  /// * Geometric MultiGrid
  ///   * Cycles: V, W, Full
  ///   * Relaxation: Jacobi, Weighted-Jacobi, Gauss-Seidel
  ///   * Grid transfer operators: Full, Half
  /// * Spectral Methods (TODO)
  ///
  /// \param matricesV potential in 3D
  /// \param matricesCharge charge density in 3D (side effect)
  /// \param symmetry symmetry or not
  ///
  /// \pre Charge density distribution in **matricesCharge** is known and boundary values for **matricesV** are set
  /// \post Numerical solution for potential distribution is calculated and stored in each rod at **matricesV**
  void poissonSolver3D(DataContainer& matricesV, const DataContainer& matricesCharge, const int symmetry);

  /// Provides poisson solver in 2D
  ///
  /// Based on the strategy (multi grid)
  ///
  /// \param matricesV potential in matrix
  /// \param matricesCharge charge density in matrix (side effect
  void poissonSolver2D(DataContainer& matricesV, const DataContainer& matricesCharge);

  DataT getSpacingZ() const { return mGrid3D.getSpacingZ(); }
  DataT getSpacingR() const { return mGrid3D.getSpacingR(); }
  DataT getSpacingPhi() const { return mGrid3D.getSpacingPhi(); }

  static void setConvergenceError(const DataT error) { sConvergenceError = error; }

  static DataT getConvergenceError() { return sConvergenceError; }

  /// get the number of threads used for some of the calculations
  static int getNThreads() { return sNThreads; }

  /// set the number of threads used for some of the calculations
  static void setNThreads(int nThreads) { sNThreads = nThreads; }

 private:
  inline static auto& mParamGrid = ParameterSpaceCharge::Instance(); ///< parameters of the grid on which the calculations are performed
  const RegularGrid& mGrid3D{};                                      ///< grid properties
  inline static DataT sConvergenceError{1e-6};                       ///< Error tolerated
  static constexpr DataT INVTWOPI = 1. / o2::constants::math::TwoPI; ///< inverse of 2*pi
  inline static int sNThreads{4};                                    ///< number of threads which are used during some of the calculations (increasing this number has no big impact)

  /// Relative error calculation: comparison with exact solution
  ///
  /// \param matricesCurrentV current potential (numerical solution)
  /// \param prevArrayV content from matricesCurrentV from previous iteration
  DataT getConvergenceError(const Vector& matricesCurrentV, Vector& prevArrayV) const;

  /// 3D - Solve Poisson's Equation in 3D by MultiGrid with constant phi slices
  ///
  ///    NOTE: In order for this algorithm to work, the number of Nr and Nz must be a power of 2 plus one.
  ///    The number of Nr and Z Column can be different.
  ///
  ///    R Row       ==  2**M + 1
  ///    Z Column  ==  2**N + 1
  ///    Phi Slice  ==  Arbitrary but greater than 3
  ///
  ///    Solving: \f$  \nabla^{2}V(r,\phi,z) = - f(r,\phi,z) \f$
  ///
  /// Algorithm for MultiGrid Full Cycle (FMG)
  /// - Relax on the coarsest grid
  /// - Do from coarsest to finest
  ///     - Interpolate potential from coarse -> fine
  ///   - Do V-Cycle to the current coarse level to the coarsest
  ///   - Stop if converged
  ///
  /// DeltaPhi in Radians
  /// \param matricesV potential in 3D matrix \f$ V(r,\phi,z) \f$
  /// \param matricesCharge charge density in 3D matrix (side effect) \f$ - f(r,\phi,z) \f$
  /// \param symmetry symmetry (TODO for symmetry = 1)
  //
  ///    SYMMETRY = 0 if no phi symmetries, and no phi boundary condition
  ///    = 1 if we have reflection symmetry at the boundaries (eg. sector symmetry or half sector symmetries).
  void poissonMultiGrid3D2D(DataContainer& matricesV, const DataContainer& matricesCharge, const int symmetry);

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
  /// - Relax on the coarsest grid
  /// - Do from coarsest to finest
  ///     - Interpolate potential from coarse -> fine
  ///   - Do V-Cycle to the current coarse level to the coarsest
  ///   - Stop if converged
  ///
  /// \param matricesV potential in 3D matrix
  /// \param matricesCharge charge density in 3D matrix (side effect)
  /// \param symmetry symmetry or not: symmetry = 0 if no phi symmetries, and no phi boundary condition.
  /// symmetry = 1 if we have reflection symmetry at the boundaries (eg. sector symmetry or half sector symmetries).
  void poissonMultiGrid3D(DataContainer& matricesV, const DataContainer& matricesCharge, const int symmetry);

  /// Solve Poisson's Equation by MultiGrid Technique in 2D (assuming cylindrical symmetry)
  ///
  /// NOTE: In order for this algorithm to work, the number of nRRow and nZColumn must be a power of 2 plus one.
  /// So nRRow == 2**M + 1 and nZColumn == 2**N + 1.  The number of nRRow and nZColumn can be different.
  ///
  /// \param matricesV potential in matrix
  /// \param matricesCharge charge density in matrix (side effect
  /// \param iPhi phi vertex
  void poissonMultiGrid2D(DataContainer& matricesV, const DataContainer& matricesCharge, const int iPhi = 0);

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
  /// \param symmetry is the cylinder has symmetry
  /// \param h2 \f$  h_{r}^{2} \f$
  /// \param tempRatioZ ration between grid size in z-direction and r-direction
  /// \param coefficient1 coefficients for \f$  V_{x+1,y,z} \f$
  /// \param coefficient2 coefficients for \f$  V_{x-1,y,z} \f$
  /// \param coefficient3 coefficients for z
  /// \param coefficient4 coefficients for f(r,\phi,z)
  void relax3D(Vector& matricesCurrentV, const Vector& matricesCurrentCharge, const int tnRRow, const int tnZColumn, const int iPhi, const int symmetry, const DataT h2, const DataT tempRatioZ,
               const std::vector<DataT>& coefficient1, const std::vector<DataT>& coefficient2, const std::vector<DataT>& coefficient3, const std::vector<DataT>& coefficient4) const;

  /// Relax2D
  ///
  ///    Relaxation operation for multiGrid
  ///    relaxation used 5 stencil in cylindrical coordinate
  ///
  /// Using the following equations
  /// \f$ U_{i,j,k} = (1 + \frac{1}{r_{i}h_{r}}) U_{i+1,j,k}  + (1 - \frac{1}{r_{i}h_{r}}) U_{i+1,j,k}  \f$
  ///
  /// \param matricesCurrentV potential in 3D (matrices of matrix)
  /// \param matricesCurrentCharge charge in 3D
  /// \param tnRRow number of vertices in r direction of TPC
  /// \param tnZColumn number of vertices in z direction of TPC
  /// \param h2 \f$  h_{r}^{2} \f$
  /// \param tempFourth coefficient for h
  /// \param tempRatio ratio between grid size in z-direction and r-direction
  /// \param coefficient1 coefficient for \f$  V_{x+1,y,z} \f$
  /// \param coefficient2 coefficient for \f$  V_{x-1,y,z} \f$
  void relax2D(Vector& matricesCurrentV, const Vector& matricesCurrentCharge, const int tnRRow, const int tnZColumn, const DataT h2, const DataT tempFourth, const DataT tempRatio,
               std::vector<DataT>& coefficient1, std::vector<DataT>& coefficient2);

  /// Interpolation/Prolongation in 2D
  ///
  /// Interpolation is a map from coarse grid (h) to fine grid (2h)
  ///
  /// In case of 2D
  /// Full weighting:
  /// \f[ (R u)_{i,j,k} = \frac{1}{2} u_{2i,2j,2k} + \frac{1}{4} S_{1} + \frac{1}{8} S_{2} + \frac{1}{16} S_{3}\f]
  ///
  ///
  /// Restriction in all direction r-phi-z
  /// \param matricesCurrentV finer grid h
  /// \param matricesCurrentVC coarse grid 2h
  /// \param tnRRow number of grid in r-direction for coarser grid should be 2^N + 1, finer grid in 2^{N+1} + 1
  /// \param tnZColumn number of grid in z-direction for coarser grid should be  2^M + 1, finer grid in 2^{M+1} + 1
  /// \param iphi phi vertex
  void interp2D(Vector& matricesCurrentV, const Vector& matricesCurrentVC, const int tnRRow, const int tnZColumn, const int iphi) const;

  /// Interpolation/Prolongation in 3D
  ///
  /// Interpolation is a map from coarse grid (h) to fine grid (2h)
  ///
  /// In case of 3D
  /// Full weighting:
  /// \f[ (R u)_{i,j,k} = \frac{1}{2} u_{2i,2j,2k} + \frac{1}{4} S_{1} + \frac{1}{8} S_{2} + \frac{1}{16} S_{3}\f]
  ///
  ///
  /// Restriction in all direction r-phi-z
  /// restriction in phi only if oldPhi == 2*newPhi
  /// \param matricesCurrentV finer grid h
  /// \param matricesCurrentVC coarse grid 2h
  /// \param tnRRow number of vertices in r-direction for coarser grid should be 2^N + 1, finer grid in 2^{N+1} + 1
  /// \param tnZColumn number of vertices in z-direction for coarser grid should be  2^M + 1, finer grid in 2^{M+1} + 1
  /// \param newPhiSlice number of vertices in phi-direction for coarser grid
  /// \param oldPhiSlice number of vertices in phi-direction for finer grid
  void interp3D(Vector& matricesCurrentV, const Vector& matricesCurrentVC, const int tnRRow, const int tnZColumn, const int newPhiSlice, const int oldPhiSlice) const;

  /// Prolongation with Addition for 3D
  ///
  /// Interpolation with addition from coarse level (2h) -->  fine level (h)
  ///
  /// Interpolation in all direction r-phi-z
  /// Interpolation in phi only if oldPhi == 2*newPhi
  /// \param matricesCurrentV fine grid h
  /// \param matricesCurrentVC coarse grid 2h
  /// \param tnRRow number of vertices in r-direction for coarser grid should be 2^N + 1, finer grid in 2^{N+1} + 1
  /// \param tnZColumn number of vertices in z-direction for coarser grid should be  2^M + 1, finer grid in 2^{M+1} + 1a
  /// \param newPhiSlice number of vertices in phi-direction for coarser grid
  /// \param oldPhiSlice number of vertices in phi-direction for finer grid
  void addInterp3D(Vector& matricesCurrentV, const Vector& matricesCurrentVC, const int tnRRow, const int tnZColumn, const int newPhiSlice, const int oldPhiSlice) const;

  /// Prolongation with Addition for 2D
  ///
  /// Interpolation with addition from coarse level (2h) -->  fine level (h)
  ///
  /// Interpolation in all direction r-phi-z
  /// \param matricesCurrentV fine grid h
  /// \param matricesCurrentVC coarse grid 2h
  /// \param tnRRow number of vertices in r-direction for coarser grid should be 2^N + 1, finer grid in 2^{N+1} + 1
  /// \param tnZColumn number of vertices in z-direction for coarser grid should be  2^M + 1, finer grid in 2^{M+1} + 1a
  /// \param tnPhi phi vertices
  void addInterp2D(Vector& matricesCurrentV, const Vector& matricesCurrentVC, const int tnRRow, const int tnZColumn, const int tnPhi) const;

  /// VCycle 3D2D, V Cycle 3D in multiGrid with constant Nphi
  /// fine-->coarsest-->fine, propagating the residue to correct initial guess of V
  ///
  /// Algorithm:
  ///
  ///    NOTE: In order for this algorithm to work, the number of Nr and Nz must be a power of 2 plus one.
  ///    The number of Nr and Z Column can be different.
  ///
  ///    R Row       ==  2**M + 1
  ///    Z Column    ==  2**N + 1
  ///    Phi Slice  ==  Arbitrary but greater than 3
  ///
  ///    DeltaPhi in Radians
  /// \param gridFrom finest level of grid
  /// \param gridTo coarsest level of grid
  /// \param nPre number of smoothing before coarsening
  /// \param nPost number of smoothing after coarsening
  /// \param ratioZ ratio between square of grid r and grid z (OPTION,  recalculate)
  /// \param ratioPhi ratio between square of grid r and grid phi (OPTION,  recalculate)
  /// \param tvArrayV vector of V potential in different grids
  /// \param tvCharge vector of charge distribution in different grids
  /// \param tvResidue vector of residue calculation in different grids
  /// \param coefficient1 coefficient for relaxation (r direction)
  /// \param coefficient2 coefficient for relaxation (r direction)
  /// \param coefficient3 coefficient for relaxation (ratio r/z)
  /// \param coefficient4 coefficient for relaxation (ratio for grid_r)
  /// \param inverseCoefficient4 coefficient for relaxation (inverse coefficient4)
  void vCycle3D2D(const int symmetry, const int gridFrom, const int gridTo, const int nPre, const int nPost, const DataT ratioZ, const DataT ratioPhi, std::vector<Vector>& tvArrayV,
                  std::vector<Vector>& tvCharge, std::vector<Vector>& tvResidue, std::vector<DataT>& coefficient1, std::vector<DataT>& coefficient2, std::vector<DataT>& coefficient3,
                  std::vector<DataT>& coefficient4, std::vector<DataT>& inverseCoefficient4) const;

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
  /// \param symmetry symmetry or not
  /// \param gridFrom finest level of grid
  /// \param gridTo coarsest level of grid
  /// \param nPre number of smoothing before coarsening
  /// \param nPost number of smoothing after coarsening
  /// \param ratioZ ratio between square of grid r and grid z (OPTION,  recalculate)
  /// \param tvArrayV vector of V potential in different grids
  /// \param tvCharge vector of charge distribution in different grids
  /// \param tvResidue vector of residue calculation in different grids
  /// \param coefficient1 coefficient for relaxation (r direction)
  /// \param coefficient2 coefficient for relaxation (r direction)
  /// \param coefficient3 coefficient for relaxation (ratio r/z)
  /// \param coefficient4 coefficient for relaxation (ratio for grid_r)
  /// \param inverseCoefficient4 coefficient for relaxation (inverse coefficient4)
  void vCycle3D(const int symmetry, const int gridFrom, const int gridTo, const int nPre, const int nPost, const DataT ratioZ, std::vector<Vector>& tvArrayV, std::vector<Vector>& tvCharge,
                std::vector<Vector>& tvResidue, std::vector<DataT>& coefficient1, std::vector<DataT>& coefficient2, std::vector<DataT>& coefficient3,
                std::vector<DataT>& coefficient4, std::vector<DataT>& inverseCoefficient4) const;

  /// V-Cycle 2D
  ///
  /// Implementation non-recursive V-cycle for 2D
  ///
  /// Algorithms:
  ///
  /// \param gridFrom finest level of grid
  /// \param gridTo coarsest level of grid
  /// \param nPre number of smoothing before coarsening
  /// \param nPost number of smoothing after coarsening
  /// \param gridSizeR grid size in r direction (OPTION,  recalculate)
  /// \param ratio ratio between square of grid r and grid z (OPTION,  recalculate)
  /// \param tvArrayV vector of V potential in different grids
  /// \param tvCharge vector of charge distribution in different grids
  /// \param tvResidue vector of residue calculation in different grids
  void vCycle2D(const int gridFrom, const int gridTo, const int nPre, const int nPost, const DataT gridSizeR, const DataT ratio, std::vector<Vector>& tvArrayV,
                std::vector<Vector>& tvCharge, std::vector<Vector>& tvResidue);

  /// W-Cycle 2D
  ///
  /// Implementation non-recursive W-cycle for 2D
  ///
  /// Algorithms:
  ///
  /// \param gridFrom finest level of grid
  /// \param gridTo coarsest level of grid
  /// \param gamma number of iterations at coarsest level
  /// \param nPre number of smoothing before coarsening
  /// \param nPost number of smoothing after coarsening
  /// \param gridSizeR grid size in r direction (OPTION,  recalculate)
  /// \param ratio ratio between square of grid r and grid z (OPTION,  recalculate)
  /// \param tvArrayV vector of V potential in different grids
  /// \param tvCharge vector of charge distribution in different grids
  /// \param tvResidue vector of residue calculation in different grids
  void wCycle2D(const int gridFrom, const int gridTo, const int gamma, const int nPre, const int nPost, const DataT gridSizeR, const DataT ratio,
                std::vector<Vector>& tvArrayV, std::vector<Vector>& tvCharge, std::vector<Vector>& tvResidue);

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
  /// \param symmetry if the cylinder has symmetry
  /// \param ih2 \f$ 1/ h_{r}^{2} \f$
  /// \param tempRatioZ ration between grid size in z-direction and r-direction
  /// \param coefficient1 coefficient for \f$  V_{x+1,y,z} \f$
  /// \param coefficient2 coefficient for \f$  V_{x-1,y,z} \f$
  /// \param coefficient3 coefficient for z
  /// \param inverseCoefficient4 inverse coefficient for f(r,\phi,z)
  void residue3D(Vector& residue, const Vector& matricesCurrentV, const Vector& matricesCurrentCharge, const int tnRRow, const int tnZColumn, const int tnPhi, const int symmetry, const DataT ih2, const DataT tempRatioZ,
                 const std::vector<DataT>& coefficient1, const std::vector<DataT>& coefficient2, const std::vector<DataT>& coefficient3, const std::vector<DataT>& inverseCoefficient4) const;

  /// Residue2D
  ///
  ///    Compute residue from V(.) where V(.) is numerical potential and f(.).
  ///    residue used 5 stencil in cylindrical coordinate
  ///
  /// Using the following equations
  /// \f$ U_{i,j,k} = (1 + \frac{1}{r_{i}h_{r}}) U_{i+1,j,k}  + (1 - \frac{1}{r_{i}h_{r}}) U_{i+1,j,k}  \f$
  ///
  /// \param residue potential in 2D
  /// \param matricesCurrentV potential in 2D
  /// \param matricesCurrentCharge charge in 2D
  /// \param nRRow number of nRRow in the r direction of TPC
  /// \param nZColumn number of nZColumn in z direction of TPC
  /// \param ih2 \f$  h_{r}^{2} \f$
  /// \param iTempFourth coefficient for h
  /// \param tempRatio ratio between grid size in z-direction and r-direction
  /// \param coefficient1 coefficient for \f$  V_{x+1,y,z} \f$
  /// \param coefficient2 coefficient for \f$  V_{x-1,y,z} \f$
  void residue2D(Vector& residue, const Vector& matricesCurrentV, const Vector& matricesCurrentCharge, const int tnRRow, const int tnZColumn, const DataT ih2, const DataT inverseTempFourth,
                 const DataT tempRatio, std::vector<DataT>& coefficient1, std::vector<DataT>& coefficient2);

  ///    Boundary transfer  restrict from fine -> coarse grid
  ///
  /// \param matricesCurrentCharge coarse grid (2h)
  /// \param residue fine grid  (h)
  /// \param tnRRow number of vertices in the r direction of TPC
  /// \param tnZColumn number of vertices in z direction of TPC
  /// \param tnPhi phi vertices
  void restrictBoundary2D(Vector& matricesCurrentCharge, const Vector& residue, const int tnRRow, const int tnZColumn, const int tnPhi) const;

  // calculate coefficients
  void calcCoefficients(unsigned int from, unsigned int to, const DataT h, const DataT tempRatioZ, const DataT tempRatioPhi, std::vector<DataT>& coefficient1,
                        std::vector<DataT>& coefficient2, std::vector<DataT>& coefficient3, std::vector<DataT>& coefficient4) const;

  // calculate coefficients for 2D poisson solver
  void calcCoefficients2D(unsigned int from, unsigned int to, const DataT h, std::vector<DataT>& coefficient1, std::vector<DataT>& coefficient2) const;

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
