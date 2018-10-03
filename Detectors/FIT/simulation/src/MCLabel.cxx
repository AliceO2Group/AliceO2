#include "FITSimulation/MCLabel.h"

using namespace o2::fit;

ClassImp(o2::fit::MCLabel);

MCLabel::MCLabel(Int_t trackID, Int_t eventID, Int_t srcID, Int_t qID)
  : o2::MCCompLabel(trackID, eventID, srcID),
    mDetID(qID)
{
 
  //  std::cout<<"@@@ MCLabel constructor "<<std::endl;
}
									 
