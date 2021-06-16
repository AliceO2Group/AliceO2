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

/// \file  SplineSpec.h
/// \brief Definition of SplineSpec class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#ifndef ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_SPLINEUTIL_H
#define ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_SPLINEUTIL_H

namespace GPUCA_NAMESPACE
{
namespace gpu
{

/// ==================================================================================================
/// Utilities for the Spline class
///
class SplineUtil
{
 public:
  /// Calculate a Spline specialization number depending on nXdim, nYdim
  ///
  static constexpr int getSpec(int nXdim, int nYdim)
  {
    // List of the Spline class specializations:
    //
    //  0 - a parent class for other specializations
    //  1 - nXdim>0, nYdim>0: both nXdim and nYdim are set at the compile time
    //  2 - at least one of the dimensions must be set during runtime
    //  3 - specialization where nYdim==1 (a small add-on on top of the other specs)

    // calculate it as one return statement to make the AliRoot compiler happy
    return (nYdim == 1) ? 3 : ((nXdim > 0 && nYdim > 0) ? 1 : 2);
    /*
    if (nYdim == 1) {
      return 3;
    }
    if (nXdim > 0 && nYdim > 0) {
      return 1;
    } else {
      return 2;
    }
    */
  }

  /// Spline1D & Spline2D specialization number depending on nYdim
  ///
  static constexpr int getSpec(int nYdim)
  {
    return getSpec(1, nYdim);
  }

  /// abs() as a constexpr method, to make the GPU compiler happy
  static constexpr int abs(int v) { return (v >= 0) ? v : -v; }

  /// class lets one to switch between constexpr int ValTrueT and int mValFalse, depending on the ConditionT
  template <bool ConditionT, int ValTrueT>
  class Switch;

  /// An expression
  /// const auto tmp = getNdim<nDimT>(int Ndim);
  /// tmp.get();
  /// returns either a constexpr integer NdimT, or an integer Ndim, depending on the (NdimT>0) value
  /// (a temporary variable tmp is needed to make the GPU compiler happy)
  ///
  template <int NdimT>
  GPUd() static Switch<(NdimT > 0), NdimT> getNdim(int Ndim)
  {
    return Switch<(NdimT > 0), NdimT>(Ndim);
  }

  /// An expression
  /// const auto tmp = getMaxNdim(int Ndim);
  /// tmp.get();
  /// returns either a constexpr integer abs(NdimT), or an integer Ndim, depending on the (NdimT!=0) value
  ///
  template <int NdimT>
  GPUd() static Switch<(NdimT != 0), abs(NdimT)> getMaxNdim(int Ndim)
  {
    return Switch<(NdimT != 0), abs(NdimT)>(Ndim);
  }
};

template <int ValTrueT>
class SplineUtil::Switch<true, ValTrueT>
{
 public:
  GPUd() Switch(int /*valFalse*/) {}
  GPUd() static constexpr int get() { return ValTrueT; }
};

template <int ValTrueT>
class SplineUtil::Switch<false, ValTrueT>
{
 public:
  GPUd() Switch(int valFalse) : mValFalse(valFalse) {}
  GPUd() int get() const { return mValFalse; }

 private:
  int mValFalse;
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
