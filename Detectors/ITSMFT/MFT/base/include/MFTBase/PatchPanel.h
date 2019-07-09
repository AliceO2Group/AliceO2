
#ifndef ALICEO2_MFT_PATCHPANEL_H_
#define ALICEO2_MFT_PATCHPANEL_H_

#include "TNamed.h"

class TGeoVolumeAssembly;

namespace o2
{
namespace mft
{

class PatchPanel : public TNamed
{

 public:
  PatchPanel();

  ~PatchPanel() override;

  TGeoVolumeAssembly* createPatchPanel(Int_t half);

 protected:
  TGeoVolumeAssembly* mPatchPanel;

 private:
  ClassDefOverride(PatchPanel, 1)
};
}
}

#endif
