/**********************************************
 *
 * Class designed to provide a basic published
 * interface for AliRoot classes to be able 
 * to reference centrality information
 * 
 * For completeness, the actual methods that
 * will typically be invoked at runtime are the 
 * ones in the class: 
 * 
 * OADB/COMMON/MULTIPLICITY/AliMultSelection.cxx
 * 
 * First implementation includes:
 * --- GetMultiplicityPercentile
 * --- GetEvSelCode 
 *
 * Bugs? Problems? Suggestions? Please contact:
 * --- david.dobrigkeit.chinellato@cern.ch 
 *
 **********************************************/

#include <iostream>
#include <TROOT.h>
#include "AliVEvent.h"
#include "AliCentrality.h"
#include "AliMultSelectionBase.h"
using namespace std;

ClassImp(AliMultSelectionBase);
//________________________________________________________________
AliMultSelectionBase::AliMultSelectionBase() :
  TNamed()
{
  // Constructor
}
//________________________________________________________________
AliMultSelectionBase::AliMultSelectionBase(const char * name, const char * title):
TNamed(name,title)
{
  // Constructor
}
//________________________________________________________________
AliMultSelectionBase::~AliMultSelectionBase(){
    // destructor: clean stuff up
    //Nothing to destroy
}

//________________________________________________________________
Float_t AliMultSelectionBase::GetMultiplicityPercentileWithFallback(AliVEvent* lEvent, TString lName ){
    
    Float_t lReturnValue = -1000;
    
    //Step 1: Acquire run number
    Int_t lRunNumber = lEvent->GetRunNumber();
    
    //Use AliCentrality if Run 1
    if( lRunNumber < 200000){
        AliCentrality* centrality;
        centrality = lEvent->GetCentrality();
        if ( centrality ) {
            lReturnValue = centrality->GetCentralityPercentile( lName.Data() );
        }
    }
    
    //Use AliMultSelectionBase virtual function if Run 1
    if( lRunNumber > 200000){
        AliMultSelectionBase *MultSelection = (AliMultSelectionBase*) lEvent -> FindListObject("MultSelection");
        if(MultSelection){
            lReturnValue = MultSelection->GetMultiplicityPercentile( lName.Data() );
        }
    }
    return lReturnValue;
}
