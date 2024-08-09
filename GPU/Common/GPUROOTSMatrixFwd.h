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

/// \file GPUROOTSMatrixFwd.h
/// \author Matteo Concas

#ifndef GPUROOTSMATRIXFWD_H
#define GPUROOTSMATRIXFWD_H

// Standalone forward declarations for Svector / SMatrix / etc.
// To be used on GPU where ROOT is not available.

#include "GPUCommonDef.h"

namespace ROOT
{
namespace Math
{
template <typename T, unsigned int N>
class SVector;
template <class T, unsigned int D1, unsigned int D2, class R>
class SMatrix;
template <class T, unsigned int D>
class MatRepSym;
template <class T, unsigned int D1, unsigned int D2>
class MatRepStd;
} // namespace Math
} // namespace ROOT

namespace o2
{
namespace math_utils
{

namespace detail
{
template <typename T, unsigned int N>
class SVectorGPU;
template <class T, unsigned int D1, unsigned int D2, class R>
class SMatrixGPU;
template <class T, unsigned int D>
class MatRepSymGPU;
template <class T, unsigned int D1, unsigned int D2>
class MatRepStdGPU;
} // namespace detail

#if !defined(GPUCA_STANDALONE) && !defined(GPUCA_GPUCODE)
template <typename T, unsigned int N>
using SVector = ROOT::Math::SVector<T, N>;
template <class T, unsigned int D1, unsigned int D2, class R>
using SMatrix = ROOT::Math::SMatrix<T, D1, D2, R>;
template <class T, unsigned int D>
using MatRepSym = ROOT::Math::MatRepSym<T, D>;
template <class T, unsigned int D1, unsigned int D2 = D1>
using MatRepStd = ROOT::Math::MatRepStd<T, D1, D2>;
#else
template <typename T, unsigned int N>
using SVector = detail::SVectorGPU<T, N>;
template <class T, unsigned int D1, unsigned int D2 = D1, class R = detail::MatRepStdGPU<T, D1, D2>>
using SMatrix = detail::SMatrixGPU<T, D1, D2, R>;
template <class T, unsigned int D>
using MatRepSym = detail::MatRepSymGPU<T, D>;
template <class T, unsigned int D1, unsigned int D2 = D1>
using MatRepStd = detail::MatRepStdGPU<T, D1, D2>;

#endif

} // namespace math_utils
} // namespace o2

#endif
