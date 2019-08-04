// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GeometryManager.h
/// \brief Definition of the GeometryManager class

#ifndef ALICEO2_BASE_GEOMETRYMANAGER_H_
#define ALICEO2_BASE_GEOMETRYMANAGER_H_

#include <TGeoManager.h> // for TGeoManager
#include <TGeoMaterial.h>
#include <TGeoPhysicalNode.h> // for TGeoPNEntry
#include <TGeoShape.h>
#include <TMath.h>
#include <TObject.h> // for TObject
#include <string>
#include "DetectorsCommonDataFormats/DetID.h"
#include "FairLogger.h" // for LOG
#include "MathUtils/Cartesian3D.h"
#include "DetectorsBase/MatCell.h"

class TGeoHMatrix; // lines 11-11
class TGeoManager; // lines 9-9

namespace o2
{
namespace detectors
{
class AlignParam;
}

namespace base
{
/// Class for interfacing to the geometry; it also builds and manages the look-up tables for fast
/// access to geometry and alignment information for sensitive alignable volumes:
/// 1) the look-up table mapping unique volume ids to TGeoPNEntries. This allows to access
/// directly by means of the unique index the associated symbolic name and original global matrix
/// in addition to the functionality of the physical node associated to a given alignable volume
/// 2) the look-up table of the alignment objects associated to the indexed alignable volumes
class GeometryManager : public TObject
{
 public:
  ///< load geometry from file
  static void loadGeometry(std::string geomFileName = "O2geometry.root", std::string geomName = "FAIRGeom");

  ///< Get the global transformation matrix (ideal geometry) for a given alignable volume
  ///< The alignable volume is identified by 'symname' which has to be either a valid symbolic
  ///< name, the query being performed after alignment, or a valid volume path if the query is
  ///< performed before alignment.
  static Bool_t getOriginalMatrix(o2::detectors::DetID detid, int sensid, TGeoHMatrix& m);
  static Bool_t getOriginalMatrix(const char* symname, TGeoHMatrix& m);
  static const char* getSymbolicName(o2::detectors::DetID detid, int sensid);
  static TGeoPNEntry* getPNEntry(o2::detectors::DetID detid, Int_t sensid);
  static TGeoHMatrix* getMatrix(o2::detectors::DetID detid, Int_t sensid);

  static int getSensID(o2::detectors::DetID detid, int sensid)
  {
    /// compose combined detector+sensor ID for sensitive volumes
    return (detid.getMask().to_ulong() << sDetOffset) | (sensid & sSensorMask);
  }

  /// Default destructor
  ~GeometryManager() override = default;

  /// misalign geometry with alignment objects from the array, optionaly check overlaps
  static bool applyAlignment(TObjArray& alObjArray, bool ovlpcheck = false, double ovlToler = 1e-3);

  struct MatBudgetExt {
    double meanRho = 0.;  // mean density: sum(x_i*rho_i)/sum(x_i) [g/cm3]
    double meanX2X0 = 0.; // equivalent rad length fraction: sum(x_i/X0_i) [adimensional]
    double meanA = 0.;    // mean A: sum(x_i*A_i)/sum(x_i) [adimensional]
    double meanZ = 0.;    // mean Z: sum(x_i*Z_i)/sum(x_i) [adimensional]
    double meanZ2A = 0.;  // Z/A mean: sum(x_i*Z_i/A_i)/sum(x_i) [adimensional]
    double length = -1.;  // length: sum(x_i) [cm]
    int nCross = 0;
    ; // number of boundary crosses

    MatBudgetExt() = default;
    ~MatBudgetExt() = default;
    MatBudgetExt(const MatBudgetExt& src) = default;
    MatBudgetExt& operator=(const MatBudgetExt& src) = default;
    void normalize(double nrm);
    ClassDefNV(MatBudgetExt, 1);
  };

  static o2::base::MatBudget meanMaterialBudget(float x0, float y0, float z0, float x1, float y1, float z1);
  static o2::base::MatBudget meanMaterialBudget(const Point3D<float>& start, const Point3D<float>& end)
  {
    return meanMaterialBudget(start.X(), start.Y(), start.Z(), end.X(), end.Y(), end.Z());
  }
  static o2::base::MatBudget meanMaterialBudget(const Point3D<double>& start, const Point3D<double>& end)
  {
    return meanMaterialBudget(start.X(), start.Y(), start.Z(), end.X(), end.Y(), end.Z());
  }

  static MatBudgetExt meanMaterialBudgetExt(float x0, float y0, float z0, float x1, float y1, float z1);
  static MatBudgetExt meanMaterialBudgetExt(const Point3D<float>& start, const Point3D<float>& end)
  {
    return meanMaterialBudgetExt(start.X(), start.Y(), start.Z(), end.X(), end.Y(), end.Z());
  }
  static MatBudgetExt meanMaterialBudgetExt(const Point3D<double>& start, const Point3D<double>& end)
  {
    return meanMaterialBudgetExt(start.X(), start.Y(), start.Z(), end.X(), end.Y(), end.Z());
  }

 private:
  /// Default constructor
  GeometryManager();

  static void accountMaterial(const TGeoMaterial* material, MatBudgetExt& bd);
  static void accountMaterial(const TGeoMaterial* material, o2::base::MatBudget& bd)
  {
    bd.meanRho = material->GetDensity();
    bd.meanX2X0 = material->GetRadLen();
  }

  /// The method returns the global matrix for the volume identified by 'path' in the ideal
  /// detector geometry. The output global matrix is stored in 'm'.
  /// Returns kFALSE in case TGeo has not been initialized or the volume path is not valid.
  static Bool_t getOriginalMatrixFromPath(const char* path, TGeoHMatrix& m);

 private:
  /// sensitive volume identifier composed from (det_mask<<sDetOffset)|(sensid&sSensorMask)
  static constexpr UInt_t sDetOffset = 15; /// detector identifier will start from this bit
  static constexpr UInt_t sSensorMask =
    (0x1 << sDetOffset) - 1; /// mask=max sensitive volumes allowed per detector (0xffff)

  ClassDefOverride(GeometryManager, 0); // Manager of geometry information for alignment
};
} // namespace base
} // namespace o2

#endif
