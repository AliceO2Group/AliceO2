#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "Align/Controller.h"
#include "Align/AlignableVolume.h"
#include "Align/AlignableDetectorITS.h"
#endif
#include "Framework/Logger.h"

using namespace o2::align;

void configITS(Controller* c, int par);

int algconf(Controller* c, int par)
{
  LOG(info) << "calling algconf with " << c << " " << par;

  if (c->getDetector(o2::detectors::DetID::ITS)) {
    configITS(c, par);
  }
  c->Print("");
  LOG(info) << "user confid done";
  return 0;
}

void configITS(Controller* c, int par)
{
  const double kCondSig[AlignableVolume::kNDOFGeom] = {0.2, 0.2, 0.3, 1., 1., 1.}; // precondition sigmas
  AlignableDetectorITS* its = (AlignableDetectorITS*)c->getDetector(o2::detectors::DetID::ITS);
  auto volITS = its->getVolume("ITS"); // envelope volume

  volITS->setChildrenConstrainPattern(AlignableVolume::kDOFBitTX | AlignableVolume::kDOFBitTY | AlignableVolume::kDOFBitTZ); // no auto constraint
  volITS->setFreeDOFPattern(0);                                                                                              // fix

  for (int iv = its->getNVolumes(); iv--;) {
    auto vol = its->getVolume(iv);
    for (int idf = AlignableVolume::kNDOFGeom; idf--;) {
      if (std::abs(vol->getParErr(idf)) < 1e-6) { // there is not yer condition
        vol->setParErr(idf, kCondSig[idf]);       // set condition
      }
    }
    if (!vol->isSensor()) {
      // prevent global shift of children in the containers
      vol->setChildrenConstrainPattern(AlignableVolume::kDOFBitTX | AlignableVolume::kDOFBitTY | AlignableVolume::kDOFBitTZ);
    }
  }
  /*
  auto nvol = its->getNVolumes();
  for (int i=0;i<nvol;i++) {
    auto vol = its->getVolume(i);
    vol->Print();
  }
  */
}
