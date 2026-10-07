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

/// \file make_geometry_csv.C
/// \brief Export ALICE 3 geometry boundaries in the R-z plane to a CSV file.
/// \author Nicola Nicassio (nicola.nicassio@cern.ch)
/// \author Rocco Liotino (rocco.liotino@cern.ch)

#include <TGeoManager.h>
#include <TMath.h>
#include <TSystem.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

/// Export ALICE 3 geometry boundary points in the R-z plane.
///
/// The macro is intended to be executed from the top-level fluence-study
/// working directory, i.e. the directory containing the .sh and .py files.
///
/// By default it reads:
///   Simulation_files/ALICE3_geometry.root
///
/// and writes:
///   geometry_points.csv
///
/// Example:
///   root -l -b -q make_geometry_csv.C
///
/// \param geomfile Input ROOT geometry file.
/// \param outfile Output CSV file written in the current working directory.
/// \param npoints Number of angular scan directions.
/// \param rmax Maximum radius stored in the CSV [cm].
/// \param zmax Maximum absolute z stored in the CSV [cm].
void make_geometry_csv(const char* geomfile = "Simulation_files/ALICE3_geometry.root",
                       const char* outfile = "geometry_points.csv",
                       int npoints = 2000,
                       double rmax = 400.,
                       double zmax = 500.)
{
  if (npoints <= 0) {
    std::cerr << "ERROR: npoints must be positive." << std::endl;
    return;
  }

  if (gSystem->AccessPathName(geomfile)) {
    std::cerr << "ERROR: geometry file not found: " << geomfile << std::endl;
    std::cerr << "Run this macro from the top-level fluence-study directory."
              << std::endl;
    return;
  }

  std::cout << "Reading geometry from: " << geomfile << std::endl;

  TGeoManager* geometry = TGeoManager::Import(geomfile);
  if (!geometry) {
    std::cerr << "ERROR: failed to load geometry from: " << geomfile << std::endl;
    return;
  }

  std::vector<double> rValues;
  std::vector<double> zValues;

  double direction[3];
  double start[3] = {0.1, 0.1, 0.1};

  const double deltaAlpha = 180. * TMath::DegToRad() / npoints;

  for (int i = 0; i < npoints; ++i) {
    const double alpha = (i + 1) * deltaAlpha;

    direction[0] = 0.;
    direction[1] = std::sin(alpha);
    direction[2] = std::cos(alpha);

    geometry->InitTrack(start, direction);

    while (!geometry->IsOutside()) {
      geometry->FindNextBoundaryAndStep(1.e20, kFALSE);

      const double* point = geometry->GetCurrentPoint();
      if (!point) {
        break;
      }

      const double z = point[2];
      const double r = std::sqrt(point[0] * point[0] + point[1] * point[1]);

      if (std::fabs(z) > zmax || r > rmax) {
        break;
      }

      rValues.push_back(r);
      zValues.push_back(z);
    }
  }

  std::ofstream output(outfile);
  if (!output.is_open()) {
    std::cerr << "ERROR: cannot create output file: " << outfile << std::endl;
    return;
  }

  output << "R,Z\n";

  for (std::size_t i = 0; i < rValues.size(); ++i) {
    output << rValues[i] << "," << zValues[i] << "\n";
  }

  output.close();

  std::cout << "Saved " << rValues.size() << " geometry points to: "
            << outfile << std::endl;
}
