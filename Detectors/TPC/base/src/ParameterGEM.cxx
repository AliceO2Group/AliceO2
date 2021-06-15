// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See http://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ParameterGEM.cxx
/// \brief Implementation of the parameter class for the GEM stack
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

// Remark: This file has been modified by Viktor Ratza in order to
// implement the efficiency models for the collection and the
// extraction efficiency.

#include "TPCBase/ParameterGEM.h"

using namespace o2::tpc;
O2ParamImpl(o2::tpc::ParameterGEM);
