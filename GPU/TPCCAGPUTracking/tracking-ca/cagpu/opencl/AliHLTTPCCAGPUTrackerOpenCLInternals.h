//-*- Mode: C++ -*-
// $Id$

// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

//  @file   AliHLTTPCCAGPUTrackerOpenCL.h
//  @author David Rohr, Sergey Gorbunov
//  @date   
//  @brief  TPC CA Tracker for the NVIDIA GPU
//  @note 


#ifndef ALIHLTTPCCAGPUTRACKEROPENCLINTERNALS_H
#define ALIHLTTPCCAGPUTRACKEROPENCLINTERNALS_H

#include <CL/opencl.h>
#include <CL/cl_ext.h>

struct AliHLTTPCCAGPUTrackerOpenCLInternals
{
	cl_device_id device;
	cl_device_id* devices;
	cl_context context;
	cl_command_queue command_queue[36];
	cl_mem mem_gpu;
	cl_mem mem_constant;
	cl_mem mem_host;
	void* mem_host_ptr;
	cl_event* selector_events;
	cl_program program;

	cl_kernel kernel_neighbours_finder, kernel_neighbours_cleaner, kernel_start_hits_finder, kernel_start_hits_sorter, kernel_tracklet_constructor, kernel_tracklet_selector, kernel_row_blocks;
};

#endif