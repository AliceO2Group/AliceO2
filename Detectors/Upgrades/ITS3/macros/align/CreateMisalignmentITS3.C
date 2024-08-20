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

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TRandom.h"
#include "TMatrixD.h"

#include "ITS3Align/MisalignmentParameters.h"
#endif

void CreateMisalignmentITS3(bool dummy = false, bool manual = false)
{
  gRandom->SetSeed(42);

  // Legendre coeff.
  constexpr int nOrder{2};
  auto getRandom = []() {
    constexpr double scale{50.e-4};
    return scale * gRandom->Uniform(-1.0, 1.0);
  };

  auto getSign = []() { return gRandom->Uniform() ? -1.0 : 1.0; };

  o2::its3::align::MisalignmentParameters params;

  if (dummy) {
    TMatrixD coeffNull(0 + 1, 0 + 1);
    for (int sensorID{0}; sensorID < 6; ++sensorID) {
      params.setLegendreCoeffX(sensorID, coeffNull);
      params.setLegendreCoeffY(sensorID, coeffNull);
      params.setLegendreCoeffZ(sensorID, coeffNull);
    }
  } else if (manual) {
    // (0,0) -> shift
    // (1,0) ->
    for (int sensorID{0}; sensorID < 6; ++sensorID) {
      constexpr double scale{20e-4};
      TMatrixD coeffNull(1, 1);

      TMatrixD coeffMinusX(1 + 1, 1 + 1);
      TMatrixD coeffPlusX(1 + 1, 1 + 1);
      coeffMinusX(1, 1) = -scale;
      coeffPlusX(1, 1) = scale;

      TMatrixD coeffMinusY(4 + 1, 4 + 1);
      TMatrixD coeffPlusY(4 + 1, 4 + 1);
      coeffMinusY(0, 0) = scale;
      coeffPlusY(0, 0) = -scale;
      coeffMinusY(4, 4) = -scale;
      coeffPlusY(4, 4) = scale;
      if (sensorID % 2 == 0) {
        params.setLegendreCoeffX(sensorID, coeffPlusX);
        params.setLegendreCoeffY(sensorID, coeffPlusY);
        params.setLegendreCoeffZ(sensorID, coeffNull);
      } else {
        params.setLegendreCoeffX(sensorID, coeffMinusX);
        params.setLegendreCoeffY(sensorID, coeffMinusY);
        params.setLegendreCoeffZ(sensorID, coeffNull);
      }
    }
  } else {
    for (int sensorID{0}; sensorID < 6; ++sensorID) {
      TMatrixD coeffX(nOrder + 1, nOrder + 1);
      TMatrixD coeffY(nOrder + 1, nOrder + 1);
      TMatrixD coeffZ(nOrder + 1, nOrder + 1);
      for (int i{0}; i <= nOrder; ++i) {
        for (int j{0}; j <= i; ++j) {
          // some random scaling as higher order parameters have higher influence
          coeffX(i, j) = getRandom() / (1.0 + i * j * 2.0);
          coeffZ(i, j) = getRandom() / (1.0 + i * j * 2.0);
          coeffY(i, j) = getRandom() / (1.0 + i * j * 2.0);
        }
      }

      params.setLegendreCoeffX(sensorID, coeffX);
      params.setLegendreCoeffY(sensorID, coeffY);
      params.setLegendreCoeffZ(sensorID, coeffZ);
    }
  }

  for (int sensorID{0}; sensorID < 6; ++sensorID) {
    params.printLegendreParams(sensorID);
  }

  params.store("misparams.root");
}
