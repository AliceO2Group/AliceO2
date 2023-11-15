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
  inline static bool normalizeGridToOneSector = false;            ///< the grid in phi direction is squashed from 2 Pi to (2 Pi / SECTORSPERSIDE). This can used to get the potential for phi symmetric sc density or boundary potentials
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

template <typename DataT = double>
struct GEMFrameParameters {
  static constexpr DataT WIDTHFRAME{1};                             ///< width of the frame 1 cm
  static constexpr DataT LENGTHFRAMEIROCBOTTOM{29.195};             ///< length of the GEM frame on the bottom side of the IROC
  static constexpr DataT LENGTHFRAMEOROC3TOP{87.048};               ///< length of the GEM frame on the top side of the OROC3
  static constexpr DataT POSBOTTOM[]{83.65, 133.5, 169.75, 207.85}; ///< local x position of the GEM frame on the bottom side per stack
  static constexpr DataT POSTOP[]{133.3, 169.75, 207.85, 247.7};    ///< local x position of the GEM frame on the top side per stack
};

template <typename DataT = double>
struct GridProperties {
  static constexpr DataT RMIN{TPCParameters<DataT>::IFCRADIUS};                  ///< min radius
  static constexpr DataT ZMIN{0};                                                ///< min z coordinate
  static constexpr DataT PHIMIN{0};                                              ///< min phi coordinate
  static constexpr DataT RMAX{TPCParameters<DataT>::OFCRADIUS};                  ///< max radius
  static constexpr DataT ZMAX{TPCParameters<DataT>::TPCZ0};                      ///< max z coordinate
  static constexpr DataT PHIMAX{static_cast<DataT>(o2::constants::math::TwoPI)}; ///< max phi coordinate

  static constexpr DataT getRMin() { return RMIN; }
  static constexpr DataT getZMin() { return ZMIN; }
  static constexpr DataT getPhiMin() { return PHIMIN; }

  ///< \return returns grid spacing in r direction
  static constexpr DataT getGridSpacingR(const unsigned int nR) { return (RMAX - RMIN) / (nR - 1); }

  ///< \return returns grid spacing in z direction
  static constexpr DataT getGridSpacingZ(const unsigned int nZ) { return (ZMAX - ZMIN) / (nZ - 1); }

  ///< \return returns grid spacing in phi direction
  static constexpr DataT getGridSpacingPhi(const unsigned int nPhi) { return PHIMAX / nPhi; }
};

} // namespace tpc
} // namespace o2

#endif
