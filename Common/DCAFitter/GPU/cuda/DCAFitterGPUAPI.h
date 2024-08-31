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

#ifndef DCAFITTERN_GPU_API_H_
#define DCAFITTERN_GPU_API_H_

namespace o2::vertexing
{
void doProcessingOnGPU(o2::vertexing::DCAFitterN<2>* ft, o2::track::TrackParCov* t1, o2::track::TrackParCov* t2);
} // namespace o2::vertexing
#endif