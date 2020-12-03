// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file O2TPCPoissonSolverHelpers.h
/// \brief This file provides all the necessary structs which are used in the poisson solver
///
/// \author  Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
/// \date Aug 21, 2020

#ifndef ALICEO2_TPC_POISSONSOLVERHELPERS_H_
#define ALICEO2_TPC_POISSONSOLVERHELPERS_H_

#include "CommonConstants/MathConstants.h"

namespace o2
{
namespace tpc
{

///< Enumeration of Cycles Type
enum class CycleType {
  VCycle = 0, ///< V Cycle
  WCycle = 1, ///< W Cycle (TODO)
  FCycle = 2  ///< Full Cycle
};

///< Fine -> Coarse Grid transfer operator types
enum class GridTransferType {
  Half = 0, ///< Half weighting
  Full = 1, ///< Full weighting
};

///< Smoothing (Relax) operator types
enum class RelaxType {
  Jacobi = 0,         ///< Jacobi (5 Stencil 2D, 7 Stencil 3D_
  WeightedJacobi = 1, ///< (TODO)
  GaussSeidel = 2     ///< Gauss Seidel 2D (2 Color, 5 Stencil), 3D (7 Stencil)
};

struct MGParameters {                                             ///< Parameters choice for MultiGrid algorithm
  inline static bool isFull3D = true;                             ///<  TRUE: full coarsening, FALSE: semi coarsening
  inline static CycleType cycleType = CycleType::FCycle;          ///< cycleType follow  CycleType
  inline static GridTransferType gtType = GridTransferType::Full; ///< gtType grid transfer type follow GridTransferType
  inline static RelaxType relaxType = RelaxType::GaussSeidel;     ///< relaxType follow RelaxType
  inline static int nPre = 2;                                     ///< number of iteration for pre smoothing
  inline static int nPost = 2;                                    ///< number of iteration for post smoothing
  inline static int nMGCycle = 200;                               ///< number of multi grid cycle (V type)
  inline static int maxLoop = 7;                                  ///< the number of tree-deep of multi grid
  inline static int gamma = 1;                                    ///< number of iteration at coarsest level !TODO SET TO REASONABLE VALUE!
};

template <typename DataT = double>
struct TPCParameters {
  static constexpr DataT TPCZ0{249.525};                        ///< nominal G1T position
  static constexpr DataT IFCRADIUS{83.5};                       ///< Mean Radius of the Inner Field Cage ( 82.43 min,  83.70 max) (cm)
  static constexpr DataT OFCRADIUS{254.5};                      ///< Mean Radius of the Outer Field Cage (252.55 min, 256.45 max) (cm)
  static constexpr DataT ZOFFSET{0.2};                          ///< Offset from CE: calculate all distortions closer to CE as if at this point
  static constexpr DataT DVDE{0.0024};                          ///< [cm/V] drift velocity dependency on the E field (from Magboltz for NeCO2N2 at standard environment)
  static constexpr DataT EM{-1.602176487e-19 / 9.10938215e-31}; ///< charge/mass in [C/kg]
  static constexpr DataT E0{8.854187817e-12};                   ///< vacuum permittivity [A·s/(V·m)]
  inline static DataT cathodev{-103070.0};                      ///< Cathode Voltage [V] (for 400 V/cm)
  inline static DataT vg1t{-3260};                              ///< GEM 1 Top voltage. (setting with reduced ET1,2,4 = 3.5kV/cm)
};

template <typename DataT = double, size_t Nr = 129, size_t Nz = 129, size_t Nphi = 180>
struct GridProperties {
  static constexpr DataT RMIN{TPCParameters<DataT>::IFCRADIUS};                  ///< min radius
  static constexpr DataT ZMIN{0};                                                ///< min z coordinate
  static constexpr DataT PHIMIN{0};                                              ///< min phi coordinate
  static constexpr DataT RMAX{TPCParameters<DataT>::OFCRADIUS};                  ///< max radius
  static constexpr DataT ZMAX{TPCParameters<DataT>::TPCZ0};                      ///< max z coordinate
  static constexpr DataT PHIMAX{static_cast<DataT>(o2::constants::math::TwoPI)}; ///< max phi coordinate
  static constexpr DataT GRIDSPACINGR{(RMAX - RMIN) / (Nr - 1)};                 ///< grid spacing in r direction
  static constexpr DataT GRIDSPACINGZ{(ZMAX - ZMIN) / (Nz - 1)};                 ///< grid spacing in z direction
  static constexpr DataT GRIDSPACINGPHI{PHIMAX / Nphi};                          ///< grid spacing in phi direction
};

} // namespace tpc
} // namespace o2

#endif
