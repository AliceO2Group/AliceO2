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

/// \file GPUUtilsLinkDef.h
/// \author David Rohr

#ifdef __CLING__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class o2::gpu::FlatObject + ;
#pragma link C++ class o2::gpu::gpustd::bitset < 1> + ;
#pragma link C++ class o2::gpu::gpustd::bitset < 2> + ;
#pragma link C++ class o2::gpu::gpustd::bitset < 3> + ;
#pragma link C++ class o2::gpu::gpustd::bitset < 4> + ;
#pragma link C++ class o2::gpu::gpustd::bitset < 5> + ;
#pragma link C++ class o2::gpu::gpustd::bitset < 6> + ;
#pragma link C++ class o2::gpu::gpustd::bitset < 7> + ;
#pragma link C++ class o2::gpu::gpustd::bitset < 8> + ;
#pragma link C++ class o2::gpu::gpustd::bitset < 9> + ;
#pragma link C++ class o2::gpu::gpustd::bitset < 10> + ;
#pragma link C++ class o2::gpu::gpustd::bitset < 11> + ;
#pragma link C++ class o2::gpu::gpustd::bitset < 12> + ;
#pragma link C++ class o2::gpu::gpustd::bitset < 13> + ;
#pragma link C++ class o2::gpu::gpustd::bitset < 14> + ;
#pragma link C++ class o2::gpu::gpustd::bitset < 15> + ;
#pragma link C++ class o2::gpu::gpustd::bitset < 16> + ;
#pragma link C++ class o2::gpu::gpustd::bitset < 17> + ;
#pragma link C++ class o2::gpu::gpustd::bitset < 18> + ;
#pragma link C++ class o2::gpu::gpustd::bitset < 19> + ;
#pragma link C++ class o2::gpu::gpustd::bitset < 20> + ;
#pragma link C++ class o2::gpu::gpustd::bitset < 21> + ;
#pragma link C++ class o2::gpu::gpustd::bitset < 22> + ;
#pragma link C++ class o2::gpu::gpustd::bitset < 23> + ;
#pragma link C++ class o2::gpu::gpustd::bitset < 24> + ;
#pragma link C++ class o2::gpu::gpustd::bitset < 25> + ;
#pragma link C++ class o2::gpu::gpustd::bitset < 26> + ;
#pragma link C++ class o2::gpu::gpustd::bitset < 27> + ;
#pragma link C++ class o2::gpu::gpustd::bitset < 28> + ;
#pragma link C++ class o2::gpu::gpustd::bitset < 29> + ;
#pragma link C++ class o2::gpu::gpustd::bitset < 30> + ;
#pragma link C++ class o2::gpu::gpustd::bitset < 31> + ;
#pragma link C++ class o2::gpu::gpustd::bitset < 32> + ;

#pragma link C++ class o2::gpu::Spline1DContainer < float, o2::gpu::FlatObject> + ;
#pragma link C++ class o2::gpu::Spline1DContainer < double, o2::gpu::FlatObject> + ;
#pragma link C++ class o2::gpu::Spline1D < float, 0, o2::gpu::FlatObject> + ;
#pragma link C++ class o2::gpu::Spline1D < double, 0, o2::gpu::FlatObject> + ;
#pragma link C++ class o2::gpu::Spline1DHelperOld < float> + ;
#pragma link C++ class o2::gpu::Spline1DHelperOld < double> + ;
#pragma link C++ class o2::gpu::Spline1DHelper < float> + ;
#pragma link C++ class o2::gpu::Spline1DHelper < double> + ;
#pragma link C++ class o2::gpu::Spline1DSpec < float, 0, 2> + ;
#pragma link C++ class o2::gpu::Spline1DSpec < double, 0, 2> + ;

#pragma link C++ class o2::gpu::Spline2DContainer < float, o2::gpu::FlatObject> + ;
#pragma link C++ class o2::gpu::Spline2DContainer < double, o2::gpu::FlatObject> + ;
#pragma link C++ class o2::gpu::Spline2D < float, 0, o2::gpu::FlatObject> + ;
#pragma link C++ class o2::gpu::Spline2D < double, 0, o2::gpu::FlatObject> + ;
#pragma link C++ class o2::gpu::Spline2DHelper < float> + ;
#pragma link C++ class o2::gpu::Spline2DHelper < double> + ;

#pragma link C++ class o2::gpu::SplineContainer < float> + ;
#pragma link C++ class o2::gpu::SplineContainer < double> + ;
#pragma link C++ class o2::gpu::SplineHelper < float> + ;
#pragma link C++ class o2::gpu::SplineHelper < double> + ;

#pragma link C++ class o2::gpu::Spline < float> + ;
#pragma link C++ class o2::gpu::Spline < double> + ;

#pragma link C++ struct o2::gpu::MultivariatePolynomialContainer + ;
#pragma link C++ struct o2::gpu::NDPiecewisePolynomialContainer + ;

#pragma link C++ class o2::gpu::SymMatrixSolver + ;
#pragma link C++ class o2::gpu::BandMatrixSolver < 0> + ;

#endif
