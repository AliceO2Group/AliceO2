// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifdef __CLING__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;
#pragma link C++ enum o2::TransformType;

#pragma link C++ class o2::math_utils::Chebyshev3D + ;
#pragma link C++ class o2::math_utils::Chebyshev3DCalc + ;

#pragma link C++ namespace o2::math_utils::math_base;

#pragma link C++ function o2::math_utils::math_base::fitGaus < float>;
#pragma link C++ function o2::math_utils::math_base::fitGaus < double>;

#pragma link C++ function o2::math_utils::math_base::getStatisticsData < float>;
#pragma link C++ function o2::math_utils::math_base::getStatisticsData < double>;
#pragma link C++ function o2::math_utils::math_base::getStatisticsData < short>;

#pragma link C++ class o2::Transform3D + ;
#pragma link C++ class o2::Rotation2D + ;
#pragma link C++ class o2::base::CachingTF1 + ;

#endif
