/// \file AliITSUpgradeDigi.cxx
/// \brief Digits structure for ITS digits

#include "Digit.h"

ClassImp(AliceO2::ITS::Digit)

using namespace AliceO2::ITS;


Digit::Digit():
    FairTimeStamp(),
    fIndex(0),
    fCharge(0.),
    fLabels()
    {}
    
Digit::Digit(Int_t index, Double_t charge, Double_t time):
    FairTimeStamp(time),
    fIndex(index),
    fCharge(charge),
    fLabels()
    {}
    
Digit::~Digit(){}
