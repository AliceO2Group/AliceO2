#ifndef ALICEO2_MFT_PATCHPANEL_H_
#define ALICEO2_MFT_PATCHPANEL_H_

#include "TGeoVolume.h"
#include "TGeoMatrix.h"
//#include "TNamed.h"

class TGeoVolumeAssembly;

namespace o2
{
namespace mft
{

class PatchPanel 
{

 public:
  PatchPanel();
  ~PatchPanel() = default;

  TGeoVolumeAssembly* createPatchPanel();

 //protected:
 //TGeoVolumeAssembly* mPatchPanel;

 private:
  ClassDef(PatchPanel, 1)
  //ClassDefOverride(PatchPanel, 1)
};
}
}

#endif
