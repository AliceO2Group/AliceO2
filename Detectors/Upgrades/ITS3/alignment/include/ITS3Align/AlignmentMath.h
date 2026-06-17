// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_ITS3_ALIGNMENT_MATH_H
#define O2_ITS3_ALIGNMENT_MATH_H

#include <utility>
#include <vector>

namespace o2::its3::align
{

struct TrackSlopes {
  double dydx{0.};
  double dzdx{0.};
};

std::pair<double, double> computeUV(double gloX, double gloY, double gloZ, int sensorID, double radius);
TrackSlopes computeTrackSlopes(double snp, double tgl);
std::vector<double> legendrePols(int order, double x);

// First and second derivatives dP_n/dx, d^2P_n/dx^2 for n = 0..order.
std::vector<double> legendrePolsD1(int order, double x);
std::vector<double> legendrePolsD2(int order, double x);

// Jacobian factor of the angular normalisation used by computeUV: c_phi = du/dphi = 2 / (phiBorder2 - phiBorder1).
// Needed to convert derivatives with respect to the normalised u back to derivatives with respect to the azimuth phi.
double phiScale(double radius);

} // namespace o2::its3::align

#endif
