/// \file Clusterizer.h
/// \brief Implementation of the cluster finder
/// \author bogdan.vulpescu@cern.ch 
/// \date 03/05/2017

#include "ITSMFTBase/Digit.h"

#include "MFTReconstruction/Clusterizer.h"

using o2::ITSMFT::SegmentationPixel;
using o2::ITSMFT::Digit;

using namespace o2::MFT;

//_____________________________________________________________________________
Clusterizer::Clusterizer() = default;

//_____________________________________________________________________________
Clusterizer::~Clusterizer() = default;

//_____________________________________________________________________________
void Clusterizer::process(const SegmentationPixel *seg, const TClonesArray* digits, TClonesArray* clusters)
{

}

