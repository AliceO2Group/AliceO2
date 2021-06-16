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

/// \file GPUO2DataTypes.h
/// \author David Rohr

#ifndef O2_GPU_GPUO2DATATYPES_H
#define O2_GPU_GPUO2DATATYPES_H

// Pull in several O2 headers with basic data types, or load a header with empty fake classes if O2 headers not available

#if defined(GPUCA_HAVE_O2HEADERS) && (!defined(__OPENCL__) || defined(__OPENCLCPP__))
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/Digit.h"
#include "DetectorsBase/MatLayerCylSet.h"
#include "DetectorsBase/Propagator.h"
#include "TRDBase/GeometryFlat.h"
#else
#include "GPUO2FakeClasses.h"
#endif

#if !defined(__OPENCL__) || defined(__OPENCLCPP__)
#include "GPUdEdxInfo.h"
#endif

#endif
