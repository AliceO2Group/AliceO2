// Copyright 2019-2023 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   histogram.h
/// @author michael.lettrich@cern.ch
/// @brief  public interface for building and renorming histograms from source data.

#ifndef RANS_HISTOGRAM_H_
#define RANS_HISTOGRAM_H_

#ifdef __CLING__
#error rANS should not be exposed to root
#endif

#include "rANS/internal/containers/DenseHistogram.h"
#include "rANS/internal/containers/AdaptiveHistogram.h"
#include "rANS/internal/containers/SparseHistogram.h"
#include "rANS/internal/containers/RenormedHistogram.h"
#include "rANS/internal/transform/renorm.h"

#endif /* RANS_HISTOGRAM_H_ */
