// Copyright 2021-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  FV0Misaligner.C
/// \brief ROOT macro for creating an FV0 geometry alignment object. The alignment object will align both
///        detector halves in the same way. Based on ITSMisaligner.C
///
/// \author Andreas Molander andreas.molander@cern.ch, Alla Maevskaya

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include "CCDB/CcdbApi.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#include "DetectorsCommonDataFormats/AlignParam.h"
#include "FV0Base/Geometry.h"

#include <TFile.h>
#include <vector>
#include <fmt/format.h>

#endif

using AlgPar = std::array<double, 6>;

AlgPar generateMisalignment(double x, double y, double z, double psi, double theta, double phi);

void FV0Misaligner(const std::string& ccdbHost = "http://ccdb-test.cern.ch:8080", long tmin = 0, long tmax = -1,
                   double x = 0., double y = 0., double z = 0., double psi = 0., double theta = 0., double phi = 0.,
                   const std::string& objectPath = "",
                   const std::string& fileName = "FV0Alignment.root")
{
  std::vector<o2::detectors::AlignParam> params;
  AlgPar pars;
  bool glo = true;

  o2::detectors::DetID detFV0("FV0");

  pars = generateMisalignment(x, y, z, psi, theta, phi);

  for (auto& symName : {o2::fv0::Geometry::getDetectorRightSymName(), o2::fv0::Geometry::getDetectorLeftSymName()}) {
    params.emplace_back(symName.c_str(), -1, pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], glo);
  }

  if (!ccdbHost.empty()) {
    std::string path = objectPath.empty() ? o2::base::DetectorNameConf::getAlignmentPath(detFV0) : objectPath;
    LOGP(info, "Storing alignment object on {}/{}", ccdbHost, path);
    o2::ccdb::CcdbApi api;
    std::map<std::string, std::string> metadata; // can be empty
    api.init(ccdbHost.c_str());   // or http://localhost:8080 for a local installation
    // store abitrary user object in strongly typed manner
    api.storeAsTFileAny(&params, path, metadata, tmin, tmax);
  }

  if (!fileName.empty()) {
    LOGP(info, "Storing FV0 alignment in local file {}", fileName);
    TFile algFile(fileName.c_str(), "recreate");
    algFile.WriteObjectAny(&params, "std::vector<o2::detectors::AlignParam>", "alignment");
    algFile.Close();
  }
}

AlgPar generateMisalignment(double x, double y, double z, double psi, double theta, double phi)
{
  AlgPar pars;
  pars[0] = x;
  pars[1] = y;
  pars[2] = z;
  pars[3] = psi;
  pars[4] = theta;
  pars[5] = phi;
  return std::move(pars);
}
