// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file qconfigoptions.h
/// \author David Rohr

#include "GPUSettings.h" // Usually we are included from GPUSettings.h, so this doesn't do anything.
                         // But ROOT or qconfig.cpp will include us directly, so load the settings first.
#include "GPUSettingsList.h"
