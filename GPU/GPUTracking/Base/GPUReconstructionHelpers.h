// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionHelpers.h
/// \author David Rohr

#ifndef GPURECONSTRUCTIONHELPERS_H
#define GPURECONSTRUCTIONHELPERS_H

#include <mutex>

class GPUReconstructionDeviceBase;
class GPUReconstructionHelpers
{
public:
	class helperDelegateBase
	{
	};
	
	struct helperParam
	{
		pthread_t fThreadId;
		GPUReconstructionDeviceBase* fCls;
		int fNum;
		std::mutex fMutex[2];
		char fTerminate;
		helperDelegateBase* fFunctionCls;
		int (helperDelegateBase::* fFunction)(int, int, helperParam*);
		int fPhase;
		int fCount;
		volatile int fDone;
		volatile char fError;
		volatile char fReset;
	};
};

#endif
