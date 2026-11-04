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
// Compatibility imports for existing ITSMFT consumers. Implementation is owned
// by GenericTracking; new detector-independent code should include it directly.
#ifndef ALICEO2_ITSMFT_TRACKING_BOUNDEDALLOCATOR_H_
#define ALICEO2_ITSMFT_TRACKING_BOUNDEDALLOCATOR_H_

#include "GenericTracking/BoundedAllocator.h"

namespace o2::itsmft::tracking
{
using o2::gtrk::bounded_vector;
using o2::gtrk::BoundedMemoryResource;
using o2::gtrk::BoundedVector;
using o2::gtrk::clearResizeBoundedArray;
using o2::gtrk::clearResizeBoundedVector;
using o2::gtrk::deepVectorClear;
using o2::gtrk::resizeBoundedVector;
using o2::gtrk::toSTDVector;
} // namespace o2::itsmft::tracking

#endif
