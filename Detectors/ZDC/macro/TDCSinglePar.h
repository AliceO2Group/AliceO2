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

#include <limits>
#include "ZDCBase/Constants.h"
#include "ZDCReconstruction/ZDCTDCCorr.h"

// clang-format off
// TDC time correction
std::array<double,o2::zdc::ZDCTDCCorr::NParExtC*o2::zdc::NTDCChannels+1> ts_beg_c={
+3.567646e+02,+3.640414e+01,+5.862955e-01,-1.080591e-03, // ts_beg_c_0
+3.567646e+02,+3.640414e+01,+5.862955e-01,-1.080591e-03, // ts_beg_c_1
+3.537072e+02,+2.464594e+01,+5.405806e-01,-1.153500e-03, // ts_beg_c_2
+3.537072e+02,+2.464594e+01,+5.405806e-01,-1.153500e-03, // ts_beg_c_3
+3.675198e+02,+4.187785e+01,+5.160246e-01,-1.026586e-03, // ts_beg_c_4
+3.675198e+02,+4.187785e+01,+5.160246e-01,-1.026586e-03, // ts_beg_c_5
+3.606453e+02,+3.717699e+01,+5.627414e-01,-1.100238e-03, // ts_beg_c_6
+3.606453e+02,+3.717699e+01,+5.627414e-01,-1.100238e-03, // ts_beg_c_7
+3.541302e+02,+2.464913e+01,+5.349674e-01,-1.171708e-03, // ts_beg_c_8
+3.541302e+02,+2.464913e+01,+5.349674e-01,-1.171708e-03, // ts_beg_c_9
std::numeric_limits<double>::quiet_NaN() // End_of_array
};

std::array<double,o2::zdc::ZDCTDCCorr::NParMidC*o2::zdc::NTDCChannels+1> ts_mid_c={
+3.630497e+01, // ts_mid_c_0
+3.630497e+01, // ts_mid_c_1
+2.457175e+01, // ts_mid_c_2
+2.457175e+01, // ts_mid_c_3
+4.146975e+01, // ts_mid_c_4
+4.146975e+01, // ts_mid_c_5
+3.697969e+01, // ts_mid_c_6
+3.697969e+01, // ts_mid_c_7
+2.456690e+01, // ts_mid_c_8
+2.456690e+01, // ts_mid_c_9
std::numeric_limits<double>::quiet_NaN() // End_of_array
};

std::array<double,o2::zdc::ZDCTDCCorr::NParExtC*o2::zdc::NTDCChannels+1> ts_end_c={
+2.084266e+03,+3.636002e+01,+4.350060e-01,+1.469586e-03, // ts_end_c_0
+2.084266e+03,+3.636002e+01,+4.350060e-01,+1.469586e-03, // ts_end_c_1
+2.071436e+03,+2.461395e+01,+4.138076e-01,+1.481211e-03, // ts_end_c_2
+2.071436e+03,+2.461395e+01,+4.138076e-01,+1.481211e-03, // ts_end_c_3
+2.020147e+03,+4.122140e+01,+8.946281e-02,+1.675048e-03, // ts_end_c_4
+2.020147e+03,+4.122140e+01,+8.946281e-02,+1.675048e-03, // ts_end_c_5
+2.080272e+03,+3.712559e+01,+3.991075e-01,+1.500954e-03, // ts_end_c_6
+2.080272e+03,+3.712559e+01,+3.991075e-01,+1.500954e-03, // ts_end_c_7
+2.071706e+03,+2.461032e+01,+4.167563e-01,+1.471722e-03, // ts_end_c_8
+2.071706e+03,+2.461032e+01,+4.167563e-01,+1.471722e-03, // ts_end_c_9
std::numeric_limits<double>::quiet_NaN() // End_of_array
};

// TDC amplitude correction
std::array<double,o2::zdc::ZDCTDCCorr::NParExtC*o2::zdc::NTDCChannels+1> af_beg_c={
+2.424586e+02,+8.065792e-01,-6.864762e-05,+0.000000e+00, // af_beg_c_0
+2.424586e+02,+8.065792e-01,-6.864762e-05,+0.000000e+00, // af_beg_c_1
+2.616765e+02,+8.260722e-01,-7.824813e-05,+0.000000e+00, // af_beg_c_2
+2.616765e+02,+8.260722e-01,-7.824813e-05,+0.000000e+00, // af_beg_c_3
+2.563623e+02,+8.627691e-01,-1.099489e-04,+0.000000e+00, // af_beg_c_4
+2.563623e+02,+8.627691e-01,-1.099489e-04,+0.000000e+00, // af_beg_c_5
+2.469214e+02,+8.229703e-01,-7.281013e-05,+0.000000e+00, // af_beg_c_6
+2.469214e+02,+8.229703e-01,-7.281013e-05,+0.000000e+00, // af_beg_c_7
+2.594337e+02,+8.259826e-01,-7.986759e-05,+0.000000e+00, // af_beg_c_8
+2.594337e+02,+8.259826e-01,-7.986759e-05,+0.000000e+00, // af_beg_c_9
std::numeric_limits<double>::quiet_NaN() // End_of_array
};

std::array<double,o2::zdc::ZDCTDCCorr::NParMidC*o2::zdc::NTDCChannels+1> af_mid_c={
+8.066536e-01, // af_mid_c_0
+8.066536e-01, // af_mid_c_1
+8.262335e-01, // af_mid_c_2
+8.262335e-01, // af_mid_c_3
+8.631356e-01, // af_mid_c_4
+8.631356e-01, // af_mid_c_5
+8.231687e-01, // af_mid_c_6
+8.231687e-01, // af_mid_c_7
+8.261481e-01, // af_mid_c_8
+8.261481e-01, // af_mid_c_9
std::numeric_limits<double>::quiet_NaN() // End_of_array
};

std::array<double,o2::zdc::ZDCTDCCorr::NParExtC*o2::zdc::NTDCChannels+1> af_end_c={
+2.126592e+03,+8.066512e-01,+1.268630e-04,+0.000000e+00, // af_end_c_0
+2.126592e+03,+8.066512e-01,+1.268630e-04,+0.000000e+00, // af_end_c_1
+2.124078e+03,+8.261414e-01,+1.153045e-04,+0.000000e+00, // af_end_c_2
+2.124078e+03,+8.261414e-01,+1.153045e-04,+0.000000e+00, // af_end_c_3
+2.116062e+03,+8.629863e-01,+1.426419e-04,+0.000000e+00, // af_end_c_4
+2.116062e+03,+8.629863e-01,+1.426419e-04,+0.000000e+00, // af_end_c_5
+2.123255e+03,+8.232540e-01,+1.253908e-04,+0.000000e+00, // af_end_c_6
+2.123255e+03,+8.232540e-01,+1.253908e-04,+0.000000e+00, // af_end_c_7
+2.123129e+03,+8.262841e-01,+1.148693e-04,+0.000000e+00, // af_end_c_8
+2.123129e+03,+8.262841e-01,+1.148693e-04,+0.000000e+00, // af_end_c_9
std::numeric_limits<double>::quiet_NaN() // End_of_array
};
// clang-format on
