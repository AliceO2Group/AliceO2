/// \file HalfSegmentation.h
/// \brief Segmentation class for each half of the ALICE Muon Forward Tracker
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>
/// \date 09/06/2015

#ifndef ALICEO2_MFT_HALFSEGMENTATION_H_
#define ALICEO2_MFT_HALFSEGMENTATION_H_

#include "TNamed.h"
#include "TXMLEngine.h"

#include "MFTBase/Segmentation.h"
#include "MFTBase/VSegmentation.h"

namespace o2 { namespace MFT { class HalfDiskSegmentation; } }

namespace o2 {
namespace MFT {

class HalfSegmentation : public VSegmentation {

public:
  
  HalfSegmentation();
  HalfSegmentation(const Char_t *initFile, const Short_t id);
  HalfSegmentation(const HalfSegmentation &source);

  ~HalfSegmentation() override;
  void Clear(const Option_t* /*opt*/) override;
  
  Bool_t getID() const {return (GetUniqueID()>>12);};
  
  Int_t getNHalfDisks() const { return mHalfDisks->GetEntries(); }

  HalfDiskSegmentation* getHalfDisk(Int_t iDisk) const { if (iDisk>=0 && iDisk<mHalfDisks->GetEntries()) return (HalfDiskSegmentation*) mHalfDisks->At(iDisk); else return nullptr; }
 
private:
  
  void findHalf(TXMLEngine* xml, XMLNodePointer_t node, XMLNodePointer_t &retnode);
  void createHalfDisks(TXMLEngine* xml, XMLNodePointer_t node);

  TClonesArray *mHalfDisks; ///< \brief Array of pointer to HalfDiskSegmentation

  ClassDefOverride(HalfSegmentation, 1);
  
};

}
}

#endif
