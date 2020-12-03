// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TPCSpacechargeLinkDef.h
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#ifdef __CLING__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ nestedclasses;
#pragma link C++ nestedtypedef;

#pragma link C++ class o2::tpc::MGParameters + ;
#pragma link C++ class o2::tpc::TPCParameters < double> + ;
#pragma link C++ class o2::tpc::AnalyticalFields < double> + ;

// 257*257*360
#pragma link C++ class o2::tpc::RegularGrid3D < double, 257, 257, 360> + ;
#pragma link C++ class o2::tpc::DataContainer3D < double, 257, 257, 360> + ;
#pragma link C++ class o2::tpc::PoissonSolver < double, 257, 257, 360> + ;
#pragma link C++ class o2::tpc::SpaceCharge < double, 257, 257, 360> + ;
#pragma link C++ class o2::tpc::NumericalFields < double, 257, 257, 360> + ;
#pragma link C++ class o2::tpc::DistCorrInterpolator < double, 257, 257, 360> + ;
#pragma link C++ class o2::tpc::GridProperties < double, 257, 257, 360> + ;

// 257*257*180
#pragma link C++ class o2::tpc::RegularGrid3D < double, 257, 257, 180> + ;
#pragma link C++ class o2::tpc::DataContainer3D < double, 257, 257, 180> + ;
#pragma link C++ class o2::tpc::PoissonSolver < double, 257, 257, 180> + ;
#pragma link C++ class o2::tpc::SpaceCharge < double, 257, 257, 180> + ;
#pragma link C++ class o2::tpc::NumericalFields < double, 257, 257, 180> + ;
#pragma link C++ class o2::tpc::DistCorrInterpolator < double, 257, 257, 180> + ;
#pragma link C++ class o2::tpc::GridProperties < double, 257, 257, 180> + ;

// 129*129*180
#pragma link C++ class o2::tpc::RegularGrid3D < double, 129, 129, 180> + ;
#pragma link C++ class o2::tpc::DataContainer3D < double, 129, 129, 180> + ;
#pragma link C++ class o2::tpc::PoissonSolver < double, 129, 129, 180> + ;
#pragma link C++ class o2::tpc::SpaceCharge < double, 129, 129, 180> + ;
#pragma link C++ class o2::tpc::NumericalFields < double, 129, 129, 180> + ;
#pragma link C++ class o2::tpc::DistCorrInterpolator < double, 129, 129, 180> + ;
#pragma link C++ class o2::tpc::GridProperties < double, 129, 129, 180> + ;

// 65*65*180
#pragma link C++ class o2::tpc::RegularGrid3D < double, 65, 65, 180> + ;
#pragma link C++ class o2::tpc::DataContainer3D < double, 65, 65, 180> + ;
#pragma link C++ class o2::tpc::PoissonSolver < double, 65, 65, 180> + ;
#pragma link C++ class o2::tpc::SpaceCharge < double, 65, 65, 180> + ;
#pragma link C++ class o2::tpc::NumericalFields < double, 65, 65, 180> + ;
#pragma link C++ class o2::tpc::DistCorrInterpolator < double, 65, 65, 180> + ;
#pragma link C++ class o2::tpc::GridProperties < double, 65, 65, 180> + ;

// 33*33*180
#pragma link C++ class o2::tpc::RegularGrid3D < double, 33, 33, 180> + ;
#pragma link C++ class o2::tpc::DataContainer3D < double, 33, 33, 180> + ;
#pragma link C++ class o2::tpc::PoissonSolver < double, 33, 33, 180> + ;
#pragma link C++ class o2::tpc::SpaceCharge < double, 33, 33, 180> + ;
#pragma link C++ class o2::tpc::NumericalFields < double, 33, 33, 180> + ;
#pragma link C++ class o2::tpc::DistCorrInterpolator < double, 33, 33, 180> + ;
#pragma link C++ class o2::tpc::GridProperties < double, 33, 33, 180> + ;

// 17*17*90
#pragma link C++ class o2::tpc::RegularGrid3D < double, 17, 17, 90> + ;
#pragma link C++ class o2::tpc::DataContainer3D < double, 17, 17, 90> + ;
#pragma link C++ class o2::tpc::PoissonSolver < double, 17, 17, 90> + ;
#pragma link C++ class o2::tpc::SpaceCharge < double, 17, 17, 90> + ;
#pragma link C++ class o2::tpc::NumericalFields < double, 17, 17, 90> + ;
#pragma link C++ class o2::tpc::DistCorrInterpolator < double, 17, 17, 90> + ;
#pragma link C++ class o2::tpc::GridProperties < double, 17, 17, 90> + ;

#endif
