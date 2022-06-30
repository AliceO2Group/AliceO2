#if !defined(__CLING__) || defined(__ROOTCLING__)

//#include "MCHGeometryTest/Helpers.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/MaterialManager.h"
#include "DetectorsCommonDataFormats/AlignParam.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"

#include "MCHGeometryMisAligner/MisAligner.h"
#include "MCHGeometryTransformer/Transformations.h"

#include "MCHGeometryTest/Helpers.h"
#include "MCHGeometryCreator/Geometry.h"
#include "CCDB/CcdbApi.h"

#include "MathUtils/Cartesian.h"
#include "Math/GenVector/Cartesian3D.h"
#include "TGLRnrCtx.h"
#include "TGLViewer.h"
#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TH2F.h"
#include "TPRegexp.h"
#include "TVirtualPad.h"

#include <iostream>
#include <fmt/format.h>

#endif

// void misAlign()
void misAlign(Double_t xcartmisaligm = 0.01, Double_t xcartmisaligw = 0.0,
              Double_t ycartmisaligm = 0.02, Double_t ycartmisaligw = 0.0,
              Double_t angmisaligm = 0.0, Double_t angmisaligw = 0.0,
              //                      TString nameCDB = "ResMisAlignCDB",
              const std::string& fileName = "MCHMisAlignment.root")
{

  // create a regular geometry
  o2::mch::test::createStandaloneGeometry();
  if (!gGeoManager) {
    std::cerr << "gGeoManager == nullptr, must create a geometry first\n";
    return;
  }
  // If not closed, we need to close it
  if (!gGeoManager->IsClosed()) {
    gGeoManager->CloseGeometry();
  }
  // Then add the alignable volumes
  o2::mch::geo::addAlignableVolumes(*gGeoManager);

  std::vector<o2::detectors::AlignParam> params;

  // The misaligner
  o2::mch::geo::MisAligner aGMA(xcartmisaligm, xcartmisaligw, ycartmisaligm, ycartmisaligw, angmisaligm, angmisaligw);
  // aGMA.SetCartMisAlig(xcartmisaligm, xcartmisaligw, ycartmisaligm, ycartmisaligw);
  // aGMA.SetAngMisAlig(angmisaligm, angmisaligw);

  // To generate module mislaignment (not mandatory)
  aGMA.setModuleCartMisAlig(0.1, 0.0, 0.2, 0.0, 0.3, 0.0);
  aGMA.setModuleAngMisAlig(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

  aGMA.misAlign(params);

  // auto transformation = o2::mch::geo::transformationFromTGeoManager(*gGeoManager);
  // auto t = transformation(100);
  // o2::math_utils::Point3D<double> po;
  // t.LocalToMaster(o2::math_utils::Point3D<double>{0, 0, 0}, po);
  // LOG(info) << "0,0,0 for DE100";
  // LOG(info) << fmt::format("X: {:+f} Y: {:+f} Z: {:+f}\n", po.X(), po.Y(), po.Z());
  /*
   // A faire plus tard, pris de la macro dans AliRoot
  // Generate misaligned data in local cdb
  const TClonesArray* array = newTransform->GetMisAlignmentData();

  // 100 mum residual resolution for chamber misalignments?
  misAligner.SetAlignmentResolution(array,-1,0.01,0.01,xcartmisaligw,ycartmisaligw);
   */

  o2::detectors::DetID detMCH("MCH");

  // la suite est prise de la fonction MisAlign
  const std::string& ccdbHost = "http://localhost:8080";
  long tmin = 0;
  long tmax = -1;
  const std::string& objectPath = "";

  if (!ccdbHost.empty()) {
    std::string path = objectPath.empty() ? o2::base::DetectorNameConf::getAlignmentPath(detMCH) : objectPath;
    LOGP(info, "Storing alignment object on {}/{}", ccdbHost, path);
    o2::ccdb::CcdbApi api;
    map<string, string> metadata; // can be empty
    api.init(ccdbHost.c_str());   // or http://localhost:8080 for a local installation
    // store abitrary user object in strongly typed manner
    api.storeAsTFileAny(&params, path, metadata, tmin, tmax);
  }

  if (!fileName.empty()) {
    LOGP(info, "Storing MCH alignment in local file {}", fileName);
    TFile algFile(fileName.c_str(), "recreate");
    algFile.WriteObjectAny(&params, "std::vector<o2::detectors::AlignParam>", "alignment");
    algFile.Close();
  }

  // o2::mch::test::misAlignGeometry();
}
