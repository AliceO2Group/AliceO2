/// \file Clusterer.cxx
/// \brief Clusterer for the upgrated ITS
#include "ITSReconstruction/Clusterer.h"
#include "ITSReconstruction/Cluster.h"
#include "ITSBase/UpgradeGeometryTGeo.h"
#include "ITSBase/UpgradeSegmentationPixel.h"
#include "ITSBase/Digit.h"

#include "FairLogger.h"           // for LOG
#include "TClonesArray.h"         // for TClonesArray

ClassImp(AliceO2::ITS::Clusterer)

using namespace AliceO2::ITS;

Clusterer::Clusterer()
{
  fGeometry = new UpgradeGeometryTGeo(kTRUE, kTRUE);
}

Clusterer::~Clusterer()
{
  delete fGeometry;
}

void Clusterer::Process(const TClonesArray *digits, TClonesArray *clusters)
{
  const UpgradeSegmentationPixel *seg =
       (UpgradeSegmentationPixel*)fGeometry->getSegmentationById(0);
  Float_t sigma2=seg->cellSizeX()*seg->cellSizeX()/12.; 
  
  TClonesArray &clref = *clusters;
  for (TIter digP=TIter(digits).Begin(); digP != TIter::End(); ++digP) {
      Digit *dig = static_cast<Digit *>(*digP);
      Int_t ix=dig->GetRow(), iz=dig->GetColumn();
      Double_t charge=dig->GetCharge();
      Int_t lab=dig->GetLabel(0);
     
      Float_t x=0.,y=0.,z=0.;    
      seg->detectorToLocal(ix,iz,x,z);
      Cluster c;
      c.SetVolumeId(dig->GetChipIndex());
      c.SetX(x); c.SetY(y); c.SetZ(z);
      c.SetSigmaY2(sigma2); c.SetSigmaZ2(sigma2); 
      c.SetFrameLoc();
      c.SetLabel(lab,0);

      new(clref[clref.GetEntriesFast()]) Cluster(c);
  }
}

