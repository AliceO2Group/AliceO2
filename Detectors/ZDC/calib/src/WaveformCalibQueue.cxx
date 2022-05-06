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

#include "Framework/Logger.h"
#include "ZDCCalib/WaveformCalibQueue.h"

namespace o2
{
namespace zdc
{

int WaveformCalibQueue::append(const RecEventAux& ev)
{
  // If queue is empty insert event
  if(mIR.size()==0){
    return appendEv(ev);
  }
  // If queue is not empty and is too long remove first element
  // Check last element
  // If BC are not consecutive, clear queue and then append element
  // IF BC are consecutive append element
}

int WaveformCalibQueue::appendEv(const RecEventAux& ev)
{
  mIR.push(ev.ir);
  return mIR.size();
}

} // namespace zdc
} // namespace o2
