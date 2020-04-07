#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TGeoManager.h>
#endif

/// Simple macro to report geometry overlaps
void checkGeomOverlaps(const char* geofilename, double eps = 0.1)
{
  TGeoManager::Import(geofilename);
  if (!gGeoManager) {
    return;
  }
  gGeoManager->CheckOverlaps(eps);
  gGeoManager->PrintOverlaps();
}
