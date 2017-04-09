/// \file HalfDiskSegmentation.h
/// \brief Class for the description of the structure of a half-disk
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>
/// \date 09/06/2015

#ifndef ALICEO2_MFT_HALFDISKSEGMENTATION_H_
#define ALICEO2_MFT_HALFDISKSEGMENTATION_H_

#include "TXMLEngine.h"

#include "MFTBase/LadderSegmentation.h"
#include "MFTBase/VSegmentation.h"

class TClonesArray;

namespace AliceO2 {
namespace MFT {

class HalfDiskSegmentation : public VSegmentation {

public:

  HalfDiskSegmentation();
  HalfDiskSegmentation(UInt_t uniqueID);
  HalfDiskSegmentation(const HalfDiskSegmentation& pt);
  
  virtual ~HalfDiskSegmentation();

  virtual void Clear(const Option_t* /*opt*/);
  
  virtual void Print(Option_t* opt="");
  
  void CreateLadders(TXMLEngine* xml, XMLNodePointer_t node);
  
  /// \brief Get the number of Ladder on the Half-Disk really constructed 
  Int_t    GetNLaddersBuild()  const {return mLadders->GetEntriesFast();};

  /// \brief Get the number of Ladder on the Half-Disk
  Int_t    GetNLadders()  const {return mNLadders;};
  
  /// \brief Set the number of Ladder on the Half-Disk
  void    SetNLadders(Int_t val)   {mNLadders = val;};

  
  /// \brief Returns pointer to the ladder segmentation object
  /// \param iLadder Int_t : ladder number on the Half-Disk
  LadderSegmentation* GetLadder(Int_t iLadder) { return ( (iLadder>=0 && iLadder<GetNLadders())  ? (LadderSegmentation*) mLadders->At(iLadder) : nullptr )  ; }
  
  /// \brief Returns the Z position of the half-disk
  Double_t GetZ() const {const Double_t *pos = GetTransformation()->GetTranslation(); return pos[2];};

  Int_t GetNChips();
  
private:
  
  Int_t mNLadders; ///< \brief Number of ladder holded by the half-disk

  TClonesArray *mLadders; ///< \brief Array of pointer to LadderSegmentation
  
  ClassDef(HalfDiskSegmentation, 1);

};

}
}
	
#endif

