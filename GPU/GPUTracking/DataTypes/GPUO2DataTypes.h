// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUO2DataTypes.h
/// \author David Rohr

#ifndef O2_GPU_GPUO2DATATYPES_H
#define O2_GPU_GPUO2DATATYPES_H

// Pull in several O2 headers with basic data types, or load a header with empty fake classes if O2 headers not available

#if defined(HAVE_O2HEADERS) && (!defined(__OPENCL__) || defined(__OPENCLCPP__))
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
