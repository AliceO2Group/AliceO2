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

///
/// \file    VisualisationTrack.cxx
/// \author  Julian Myrcha
///

#include "EventVisualisationDataConverter/VisualisationCalo.h"

using namespace std;
namespace o2::event_visualisation
{

VisualisationCalo::VisualisationCalo() = default;

VisualisationCalo::VisualisationCalo(const VisualisationCaloVO& vo)
{
  this->mSource = vo.source;
  this->mTime = vo.time;
  this->mEnergy = vo.energy;
  this->mEta = vo.eta;
  this->mPhi = vo.phi;
  this->mGID = vo.gid;
  this->mPID = vo.PID;
}

VisualisationCalo::VisualisationCalo(const VisualisationCalo& src)
{
  this->mSource = src.mSource;
  this->mTime = src.mTime;
  this->mEnergy = src.mEnergy;
  this->mEta = src.mEta;
  this->mPhi = src.mPhi;
  this->mGID = src.mGID;
  this->mPID = src.mPID;
}

} // namespace o2
