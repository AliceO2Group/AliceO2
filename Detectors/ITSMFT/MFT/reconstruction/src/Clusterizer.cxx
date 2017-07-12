// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

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

