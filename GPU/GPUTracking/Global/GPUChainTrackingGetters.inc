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

/// \file GPUChainTrackingGetters.inc
/// \author David Rohr

#ifndef GPUCHAINTRACKINGGETTERS_INC_H
#define GPUCHAINTRACKINGGETTERS_INC_H

#include "GPUChainTracking.h"
#include "GPUConstantMem.h"

namespace o2::gpu
{
inline const CorrectionMapsHelper* GPUChainTracking::GetTPCTransformHelper() const { return processors()->calibObjects.fastTransformHelper; }
inline const TPCPadGainCalib* GPUChainTracking::GetTPCPadGainCalib() const { return processors()->calibObjects.tpcPadGain; }
inline const TPCZSLinkMapping* GPUChainTracking::GetTPCZSLinkMapping() const { return processors()->calibObjects.tpcZSLinkMapping; }
inline const o2::tpc::CalibdEdxContainer* GPUChainTracking::GetdEdxCalibContainer() const { return processors()->calibObjects.dEdxCalibContainer; }
inline const o2::base::MatLayerCylSet* GPUChainTracking::GetMatLUT() const { return processors()->calibObjects.matLUT; }
inline const GPUTRDGeometry* GPUChainTracking::GetTRDGeometry() const { return (GPUTRDGeometry*)processors()->calibObjects.trdGeometry; }
inline const GPUTRDRecoParam* GPUChainTracking::GetTRDRecoParam() const { return processors()->calibObjects.trdRecoParam; }
inline const o2::base::Propagator* GPUChainTracking::GetO2Propagator() const { return processors()->calibObjects.o2Propagator; }
inline void GPUChainTracking::SetMatLUT(const o2::base::MatLayerCylSet* lut) { processors()->calibObjects.matLUT = lut; }
inline void GPUChainTracking::SetTRDGeometry(const o2::trd::GeometryFlat* geo) { processors()->calibObjects.trdGeometry = geo; }
inline void GPUChainTracking::SetTRDRecoParam(const GPUTRDRecoParam* par) { processors()->calibObjects.trdRecoParam = par; }
inline void GPUChainTracking::SetCalibObjects(const GPUCalibObjectsConst& obj) { processors()->calibObjects = obj; }
inline void GPUChainTracking::SetCalibObjects(const GPUCalibObjects& obj) { memcpy((void*)&processors()->calibObjects, (const void*)&obj, sizeof(obj)); }
} // namespace o2::gpu

#endif
