/// \file MFTMisaligner.C
/// Macros to test the (mis)alignment of the MFT geometry

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "MFTBase/GeometryTGeo.h"
#include "MFTSimulation/GeometryMisAligner.h"
#include "DetectorsBase/GeometryManager.h"
#endif

//_____________________________________________________________________________
void MFTMisaligner(const std::string& ccdbHost = "http://ccdb-test.cern.ch:8080", long tmin = 0, long tmax = -1,
                   double xHalf = 0., double yHalf = 0., double zHalf = 0., double psiHalf = 0., double thetaHalf = 0., double phiHalf = 0.,
                   double xDisk = 0., double yDisk = 0., double zDisk = 0., double psiDisk = 0., double thetaDisk = 0., double phiDisk = 0.,
                   double xLadder = 0., double yLadder = 0., double zLadder = 0., double psiLadder = 0., double thetaLadder = 0., double phiLadder = 0.,
                   double xChip = 0., double yChip = 0., double zChip = 0., double psiChip = 0., double thetaChip = 0., double phiChip = 0.,
                   const std::string& objectPath = "",
                   const std::string& fileName = "MFTAlignment.root",
                   bool verbose = false)
{
  o2::base::GeometryManager::loadGeometry("", false);

  // Initialize the misaligner
  o2::mft::GeometryMisAligner misaligner;

  misaligner.SetHalfCartMisAlig(0., xHalf, 0., yHalf, 0., zHalf);
  misaligner.SetHalfAngMisAlig(0., psiHalf, 0., thetaHalf, 0., phiHalf);
  misaligner.SetDiskCartMisAlig(0., xDisk, 0., yDisk, 0., zDisk);
  misaligner.SetDiskAngMisAlig(0., psiDisk, 0., thetaDisk, 0., phiDisk);
  misaligner.SetLadderCartMisAlig(0., xLadder, 0., yLadder, 0., zLadder);
  misaligner.SetLadderAngMisAlig(0., psiLadder, 0., thetaLadder, 0., phiLadder);
  misaligner.SetSensorCartMisAlig(0., xChip, 0., yChip, 0., zChip);
  misaligner.SetSensorAngMisAlig(0., psiChip, 0., thetaChip, 0., phiChip);

  misaligner.MisAlign(verbose, ccdbHost, tmin, tmax, objectPath, fileName);
}
