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

/// \file dumpGPUDefParam.C
/// \author David Rohr

// Run e.g. as (replacing [FILE] and [OUTPUT]:
// echo -e '#define PARAMETER_FILE "[FILE]"\ngInterpreter->AddIncludePath("'`pwd`'/include/GPU");\n.x share/GPU/tools/dumpGPUDefParam.C("[OUTPUT]")\n.q\n' | root -l -b
// To dump the defaults for AMPERE architecture, run
// echo -e '#define GPUCA_GPUTYPE_AMPERE\n#define PARAMETER_FILE "GPUDefParametersDefaults.h"\ngInterpreter->AddIncludePath("'`pwd`'/include/GPU");\n.x share/GPU/tools/dumpGPUDefParam.C("default_AMPERE.par")\n.q\n' | root -l -b

#ifndef PARAMETER_FILE
#error Must provide the PARAMETER_FILE as preprocessor define, e.g. -DPARAMETER_FILE='"GPUDefParametersDefaults.h"'
#endif

#define GPUCA_GPUCODE
#include PARAMETER_FILE

#include "GPUDefParametersLoad.inc"
void dumpGPUDefParam(const char* outputfile = "parameters.out")
{
  auto param = o2::gpu::internal::GPUDefParametersLoad();
  printf("Loaded params:\n%s\nWriting them to %s\n", o2::gpu::internal::GPUDefParametersExport(param, false).c_str(), outputfile);
  FILE* fp = fopen(outputfile, "w+b");
  fwrite(&param, 1, sizeof(param), fp);
  fclose(fp);
}
