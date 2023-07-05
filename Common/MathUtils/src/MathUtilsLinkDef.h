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

#ifdef __CLING__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;
#pragma link C++ enum o2::math_utils::TransformType;

#pragma link C++ class o2::math_utils::Chebyshev3D + ;
#pragma link C++ class o2::math_utils::Chebyshev3DCalc + ;

#pragma link C++ function o2::math_utils::fit < float>;
#pragma link C++ function o2::math_utils::fit < double>;

#pragma link C++ function o2::math_utils::fitGaus < float>;
#pragma link C++ function o2::math_utils::fitGaus < double>;

#pragma link C++ function o2::math_utils::getStatisticsData < float>;
#pragma link C++ function o2::math_utils::getStatisticsData < double>;
#pragma link C++ function o2::math_utils::getStatisticsData < short>;

#pragma link C++ class o2::math_utils::Transform3D + ;
#pragma link C++ class o2::math_utils::Rotation2Df_t + ;
#pragma link C++ class o2::math_utils::Rotation2Dd_t + ;
#pragma link C++ class o2::math_utils::CachingTF1 + ;

#pragma link C++ class o2::math_utils::SymMatrixSolver + ;

#pragma link C++ class o2::math_utils::CircleXYf_t + ;
#pragma link C++ class o2::math_utils::CircleXYd_t + ;
#pragma link C++ class o2::math_utils::IntervalXYf_t + ;
#pragma link C++ class o2::math_utils::IntervalXYd_t + ;
#pragma link C++ class o2::math_utils::Bracketf_t + ;
#pragma link C++ class o2::math_utils::Bracketd_t + ;

#endif
