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
//
// Author: J. E. Munoz Mendez jesus.munoz@cern.ch

std::function<void(const double*, double*)> field()
{
  return [](const double* x, double* b) {
    double Rc;
    double R1;
    double R2;
    double B1;
    double B2;
    double beamStart = 500.;    //[cm]
    double tokGauss = 1. / 0.1; // conversion from Tesla to kGauss

    bool isMagAbs = true;

    // ***********************
    // LAYOUT 1
    // ***********************

    // RADIUS
    Rc = 185.; //[cm]
    R1 = 220.; //[cm]
    R2 = 290.; //[cm]

    // To set the B2
    B1 = 2.;                                    //[T]
    B2 = -Rc * Rc / ((R2 * R2 - R1 * R1) * B1); //[T]

    if ((abs(x[2]) <= beamStart) && (sqrt(x[0] * x[0] + x[1] * x[1]) < Rc)) {
      b[0] = 0.;
      b[1] = 0.;
      b[2] = B1 * tokGauss;
    } else if ((abs(x[2]) <= beamStart) &&
               (sqrt(x[0] * x[0] + x[1] * x[1]) >= Rc &&
                sqrt(x[0] * x[0] + x[1] * x[1]) < R1)) {
      b[0] = 0.;
      b[1] = 0.;
      b[2] = 0.;
    } else if ((abs(x[2]) <= beamStart) &&
               (sqrt(x[0] * x[0] + x[1] * x[1]) >= R1 &&
                sqrt(x[0] * x[0] + x[1] * x[1]) < R2)) {
      b[0] = 0.;
      b[1] = 0.;
      if (isMagAbs) {
        b[2] = B2 * tokGauss;
      } else {
        b[2] = 0.;
      }
    } else {
      b[0] = 0.;
      b[1] = 0.;
      b[2] = 0.;
    }
  };
}