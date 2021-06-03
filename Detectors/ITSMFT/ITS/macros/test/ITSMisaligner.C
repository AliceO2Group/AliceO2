#if !defined(__CLING__) || defined(__ROOTCLING__)
//#define ENABLE_UPGRADES
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "DetectorsCommonDataFormats/AlignParam.h"
#include "DetectorsBase/GeometryManager.h"
#include "CCDB/CcdbApi.h"
#include "ITSBase/GeometryTGeo.h"
#include <TRandom.h>
#include <TFile.h>
#include <vector>
#include <fmt/format.h>
#endif

using AlgPar = std::array<double, 6>;

AlgPar generateMisalignment(double x, double y, double z, double psi, double theta, double phi);

void ITSMisaligner(const std::string& ccdbHost = "http://ccdb-test.cern.ch:8080", long tmin = 0, long tmax = -1,
                   double xEnv = 0., double yEnv = 0., double zEnv = 0., double psiEnv = 0., double thetaEnv = 0., double phiEnv = 0.,
                   double xLay = 0., double yLay = 0., double zLay = 0., double psiLay = 0., double thetaLay = 0., double phiLay = 0.,
                   double xSta = 0., double ySta = 0., double zSta = 0., double psiSta = 0., double thetaSta = 0., double phiSta = 0.,
                   double xHSt = 0., double yHSt = 0., double zHSt = 0., double psiHSt = 0., double thetaHSt = 0., double phiHSt = 0.,
                   double xMod = 0., double yMod = 0., double zMod = 0., double psiMod = 0., double thetaMod = 0., double phiMod = 0.,
                   double xChp = 0., double yChp = 0., double zChp = 0., double psiChp = 0., double thetaChp = 0., double phiChp = 0.,
                   const std::string& objectPath = "",
                   const std::string& fileName = "ITSAlignment.root")
{
  std::vector<o2::detectors::AlignParam> params;
  o2::base::GeometryManager::loadGeometry("", false);
  auto geom = o2::its::GeometryTGeo::Instance();
  std::string symname;
  AlgPar pars;
  bool glo = true;
  symname = geom->composeSymNameITS();

  o2::detectors::DetID detITS("ITS");

  // ITS envelope
  pars = generateMisalignment(xEnv, yEnv, zEnv, psiEnv, thetaEnv, phiEnv);
  params.emplace_back(symname.c_str(), -1, pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], glo);

  for (int ilr = 0; ilr < geom->getNumberOfLayers(); ilr++) {
    symname = geom->composeSymNameLayer(ilr);
    pars = generateMisalignment(xLay, yLay, zLay, psiLay, thetaLay, phiLay);
    params.emplace_back(symname.c_str(), -1, pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], glo);

    for (int ist = 0; ist < geom->getNumberOfStaves(ilr); ist++) {
      symname = geom->composeSymNameStave(ilr, ist);
      pars = generateMisalignment(xSta, ySta, zSta, psiSta, thetaSta, phiSta);
      params.emplace_back(symname.c_str(), -1, pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], glo);

      for (int ihst = 0; ihst < geom->getNumberOfHalfStaves(ilr); ihst++) {
        symname = geom->composeSymNameHalfStave(ilr, ist, ihst);
        pars = generateMisalignment(xHSt, yHSt, zHSt, psiHSt, thetaHSt, phiHSt);
        params.emplace_back(symname.c_str(), -1, pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], glo);

        for (int imd = 0; imd < geom->getNumberOfModules(ilr); imd++) {
          symname = geom->composeSymNameModule(ilr, ist, ihst, imd);
          pars = generateMisalignment(xMod, yMod, zMod, psiMod, thetaMod, phiMod);
          params.emplace_back(symname.c_str(), -1, pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], glo);
        }
      }
    }
  }

  for (int ich = 0; ich < geom->getNumberOfChips(); ich++) {
    symname = o2::base::GeometryManager::getSymbolicName(detITS, ich);
    pars = generateMisalignment(xChp, yChp, zChp, psiChp, thetaChp, phiChp);
    int chID = o2::base::GeometryManager::getSensID(detITS, ich);
    params.emplace_back(symname.c_str(), chID, pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], glo);
  }

  if (!ccdbHost.empty()) {
    std::string path = objectPath.empty() ? o2::base::NameConf::getAlignmentPath(detITS) : objectPath;
    LOGP(INFO, "Storing alignment object on {}/{}", ccdbHost, path);
    o2::ccdb::CcdbApi api;
    map<string, string> metadata; // can be empty
    api.init(ccdbHost.c_str());   // or http://localhost:8080 for a local installation
    // store abitrary user object in strongly typed manner
    api.storeAsTFileAny(&params, path, metadata, tmin, tmax);
  }

  if (!fileName.empty()) {
    LOGP(INFO, "Storing ITS alignment in local file {}", fileName);
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
