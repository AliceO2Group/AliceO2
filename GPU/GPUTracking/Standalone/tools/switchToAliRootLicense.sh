#!/bin/bash
cd ../../../../
if [ $(ls | grep GPU | wc -l) != "1" ]; then
    echo Incorrect directory
    exit 1
fi

git grep -l "^// Copyright CERN and copyright holders of ALICE O2. This software is" | \
    grep "^GPU/Common/\|^GPU/GPUTracking/\|^GPU/TPCFastTransformation|^GPU/TPCSpaceChargeBase\|^cmake" | \
    xargs -r -n 1 \
    sed -i -e '/Copyright CERN and copyright holders of ALICE O2. This software is/,/or submit itself to any jurisdiction/c\
//**************************************************************************\
//* This file is property of and copyright by the ALICE Project            *\
//* ALICE Experiment at CERN, All rights reserved.                         *\
//*                                                                        *\
//* Primary Authors: Matthias Richter <Matthias.Richter@ift.uib.no>        *\
//*                  for The ALICE HLT Project.                            *\
//*                                                                        *\
//* Permission to use, copy, modify and distribute this software and its   *\
//* documentation strictly for non-commercial purposes is hereby granted   *\
//* without fee, provided that the above copyright notice appears in all   *\
//* copies and that both the copyright notice and this permission notice   *\
//* appear in the supporting documentation. The authors make no claims     *\
//* about the suitability of this software for any purpose. It is          *\
//* provided "as is" without express or implied warranty.                  *\
//**************************************************************************\'
