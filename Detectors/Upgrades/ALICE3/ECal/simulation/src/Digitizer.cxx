// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Digitizer.cxx
/// \brief Digitization of ECal MC information
///
/// \author Evgeny Kryshen <evgeny.kryshen@cern.ch>

#include <cstdio>
#include <TRandom.h>
#include <fairlogger/Logger.h>

#include <ECalSimulation/Digitizer.h>
#include <ECalBase/Hit.h>
#include <ECalBase/Geometry.h>
#include <DataFormatsECal/Digit.h>

using namespace o2::ecal;

//==============================================================================
Digitizer::Digitizer()
{
  auto& geo = Geometry::instance();
  mArrayD.resize(geo.getNrows() * geo.getNcols());
}

//==============================================================================
void Digitizer::processHits(const std::vector<Hit>* hits, std::vector<Digit>& digits, o2::dataformats::MCTruthContainer<MCLabel>& labels, int collId)
{
  digits.clear();
  labels.clear();

  LOGP(debug, "nHits = {}", hits->size());
  auto& geo = Geometry::instance();

  for (int i = 0; i < mArrayD.size(); i++) {
    mArrayD[i].setAmplitude(0);
    mArrayD[i].setTimeStamp(1000);
    mArrayD[i].setTower(i);
    mArrayD[i].setLabel(-1);
    // TODO: simulate noise
  }

  for (auto& hit : *hits) {
    int cellID = hit.GetCellID();
    double eloss = hit.GetEnergyLoss();
    double t = hit.GetTime();
    double elossSmeared = eloss;
    bool isCrystal = geo.isCrystal(cellID);
    if (isCrystal) { // crystal
      double elossSmearedNpe = gRandom->Poisson(eloss * mCrystalPePerGeV) / mCrystalPePerGeV;
      if (mSmearCrystal)
        elossSmeared = elossSmearedNpe * gRandom->Gaus(1, 0.007); // light attenuation in crystals
    } else {                                                      // sampling
      elossSmeared *= mSamplingFraction;
    }

    Digit& digit = mArrayD[cellID];
    digit.setAmplitude(digit.getAmplitude() + elossSmeared);
    if (t < digit.getTimeStamp())
      digit.setTimeStamp(t); // setting earliest time, TODO: add time smearing
    LOGF(debug, "  crystal: %d cellID = %5d, eloss = %8.5f elossSmeared = %8.5f time = %8.5f", isCrystal, cellID, eloss, elossSmeared, t);

    // Adding MC info
    MCLabel label(hit.GetTrackID(), collId, 0, false, hit.GetEnergyLoss());
    int labelIndex = digit.getLabel();
    if (labelIndex == -1) {
      labelIndex = labels.getIndexedSize();
      labels.addElement(labelIndex, label);
      digit.setLabel(labelIndex);
    } else {
      labels.addElementRandomAccess(labelIndex, label);
    }
  } // hits

  for (int i = 0; i < mArrayD.size(); i++) {
    if (mArrayD[i].getAmplitude() > mThreshold) {
      digits.push_back(mArrayD[i]);
    }
  }
}
