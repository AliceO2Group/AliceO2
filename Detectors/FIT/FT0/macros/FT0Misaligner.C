#if !defined(__CLING__) || defined(__ROOTCLING__)
//#define ENABLE_UPGRADES
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "DetectorsCommonDataFormats/AlignParam.h"
#include "DetectorsBase/GeometryManager.h"
#include "CCDB/CcdbApi.h"
#include "FT0Base/Geometry.h"
#include <TRandom.h>
#include <TFile.h>
#include <vector>
#include <fmt/format.h>
#endif

using AlgPar = std::array<double, 6>;

AlgPar generateMisalignment(double x, double y, double z, double psi, double theta, double phi);

void FT0Misaligner(const std::string& ccdbHost = "http://ccdb-test.cern.ch:8080", long tmin = 0, long tmax = -1,
                   double xA = 0., double yA = 0., double zA = 0., double psiA = 0., double thetaA = 0., double phiA = 0.,
                   double xC = 0., double yC = 0., double zC = 0., double psiC = 0., double thetaC = 0., double phiC = 0.,
                   const std::string& objectPath = "",
                   const std::string& fileName = "FT0Alignment.root")
{
  std::vector<o2::detectors::AlignParam> params;
  o2::base::GeometryManager::loadGeometry("", false);
  //  auto geom = o2::ft0::Geometry::Instance();
  AlgPar pars;
  bool glo = true;

  o2::detectors::DetID detFT0("FT0");

  // FT0 detector
  //set A side
  std::string symNameA = "FT0A";
  pars = generateMisalignment(xA, yA, zA, psiA, thetaA, phiA);
  params.emplace_back(symNameA.c_str(), -1, pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], glo);
  //set C side
  std::string symNameC = "FT0C";
  pars = generateMisalignment(xC, yC, zC, psiC, thetaC, phiC);
  params.emplace_back(symNameC.c_str(), -1, pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], glo);

  if (!ccdbHost.empty()) {
    std::string path = objectPath.empty() ? o2::base::NameConf::getAlignmentPath(detFT0) : objectPath;
    LOGP(INFO, "Storing alignment object on {}/{}", ccdbHost, path);
    o2::ccdb::CcdbApi api;
    map<string, string> metadata; // can be empty
    api.init(ccdbHost.c_str());   // or http://localhost:8080 for a local installation
    // store abitrary user object in strongly typed manner
    api.storeAsTFileAny(&params, path, metadata, tmin, tmax);
  }

  if (!fileName.empty()) {
    LOGP(INFO, "Storing FT0 alignment in local file {}", fileName);
    TFile algFile(fileName.c_str(), "recreate");
    algFile.WriteObjectAny(&params, "std::vector<o2::detectors::AlignParam>", "alignment");
    algFile.Close();
  }
}
AlgPar generateMisalignment(double x, double y, double z, double psi, double theta, double phi)
{
  AlgPar pars;
  pars[0] = gRandom->Gaus(0, x);
  pars[1] = gRandom->Gaus(0, y);
  pars[2] = gRandom->Gaus(0, z);
  pars[3] = gRandom->Gaus(0, psi);
  pars[4] = gRandom->Gaus(0, theta);
  pars[5] = gRandom->Gaus(0, phi);
  return std::move(pars);
}
