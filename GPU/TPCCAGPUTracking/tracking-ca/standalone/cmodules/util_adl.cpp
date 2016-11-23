///
///  Copyright (c) 2008 - 2009 Advanced Micro Devices, Inc.
 
///  THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
///  EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
///  WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.

/// \file main.c
/// \brief C/C++ ADL sample application
///
/// Demonstrates some basic ADL functions - create, destroy, obtaining adapter and display information.
/// If the display capabilities allow, increases, decreases and restores the brightness of each display

#ifndef _NO_ADL

#ifdef _WIN32
#define WINDOWS
#else
#define LINUX
#endif

#include "../../ADL/include/adl_sdk.h"
#ifdef LINUX
#include <dlfcn.h>	//dyopen, dlsym, dlclose
#include <unistd.h>	//sleep
#else
#include <windows.h>
#include <winbase.h>
#endif
#include <stdlib.h>	
#include <string.h>	//memeset
#include <stdio.h>

#ifndef MAINPROG
#include "util_adl.h"
#endif

#ifndef STD_OUT
#define STD_OUT stdout
#endif

// Definitions of the used function pointers. Add more if you use other ADL APIs
typedef int ( *ADL_MAIN_CONTROL_CREATE )(ADL_MAIN_MALLOC_CALLBACK, int );
typedef int ( *ADL_MAIN_CONTROL_DESTROY )();
typedef int ( *ADL_ADAPTER_NUMBEROFADAPTERS_GET ) ( int* );
typedef int ( *ADL_ADAPTER_ADAPTERINFO_GET ) ( LPAdapterInfo, int );
typedef int ( *ADL_OVERDRIVE5_TEMPERATURE_GET ) ( int, int , ADLTemperature * );
typedef int ( *ADL_ADAPTER_ACTIVE_GET ) ( int, int* );
typedef int ( *ADL_ADAPTER_VIDEOBIOSINFO_GET ) ( int, ADLBiosInfo* );
typedef int ( *ADL_ADAPTER_ID_GET) ( int, int* );

// Memory allocation function
void* __stdcall ADL_Main_Memory_Alloc ( int iSize )
{
    void* lpBuffer = malloc ( iSize );
    return lpBuffer;
}

ADL_MAIN_CONTROL_CREATE          ADL_Main_Control_Create;
ADL_MAIN_CONTROL_DESTROY         ADL_Main_Control_Destroy;
ADL_ADAPTER_NUMBEROFADAPTERS_GET ADL_Adapter_NumberOfAdapters_Get;
ADL_ADAPTER_ADAPTERINFO_GET      ADL_Adapter_AdapterInfo_Get;
ADL_OVERDRIVE5_TEMPERATURE_GET   ADL_Overdrive5_Temperature_Get;
ADL_ADAPTER_ACTIVE_GET           ADL_Adapter_Active_Get;
ADL_ADAPTER_VIDEOBIOSINFO_GET    ADL_Adapter_VideoBiosInfo_Get;
ADL_ADAPTER_ID_GET               ADL_Adapter_ID_Get;

int nAdapters;
int* nAdapterIndizes;
#ifdef LINUX
void *hDLL;		// Handle to .so library
#else
HINSTANCE hDLL;
#endif

#ifndef LINUX
void* dlsym(HINSTANCE lib, char* name)
{
	return(GetProcAddress(lib, name));
}
#endif

int adl_temperature_check_init()
{
    LPAdapterInfo     lpAdapterInfo = NULL;
    int  iNumberAdapters;
#ifdef LINUX
    setenv("DISPLAY", ":0", 1);
	hDLL = dlopen( "libatiadlxx.so", RTLD_LAZY|RTLD_GLOBAL);
#else
	hDLL = LoadLibrary( "atiadlxx.dll" );
#endif
    

        if (NULL == hDLL)
        {
            printf("ADL library not found!\n");
            return 0;
        }

        ADL_Main_Control_Create = (ADL_MAIN_CONTROL_CREATE) (size_t) dlsym(hDLL,"ADL_Main_Control_Create");
        ADL_Main_Control_Destroy = (ADL_MAIN_CONTROL_DESTROY) (size_t) dlsym(hDLL,"ADL_Main_Control_Destroy");
        ADL_Adapter_NumberOfAdapters_Get = (ADL_ADAPTER_NUMBEROFADAPTERS_GET) (size_t) dlsym(hDLL,"ADL_Adapter_NumberOfAdapters_Get");
        ADL_Adapter_AdapterInfo_Get = (ADL_ADAPTER_ADAPTERINFO_GET) (size_t) dlsym(hDLL,"ADL_Adapter_AdapterInfo_Get");
        ADL_Overdrive5_Temperature_Get = (ADL_OVERDRIVE5_TEMPERATURE_GET) (size_t) dlsym(hDLL,"ADL_Overdrive5_Temperature_Get");
        ADL_Adapter_Active_Get = (ADL_ADAPTER_ACTIVE_GET) (size_t) dlsym(hDLL,"ADL_Adapter_Active_Get");
        ADL_Adapter_VideoBiosInfo_Get = (ADL_ADAPTER_VIDEOBIOSINFO_GET) (size_t) dlsym(hDLL, "ADL_Adapter_VideoBiosInfo_Get");
        ADL_Adapter_ID_Get = (ADL_ADAPTER_ID_GET) (size_t) dlsym(hDLL, "ADL_Adapter_ID_Get");
        
        
		if ( NULL == ADL_Main_Control_Create || NULL == ADL_Main_Control_Destroy || NULL == ADL_Adapter_NumberOfAdapters_Get || NULL == ADL_Adapter_AdapterInfo_Get || NULL == ADL_Overdrive5_Temperature_Get || NULL == ADL_Adapter_Active_Get || NULL == ADL_Adapter_VideoBiosInfo_Get || NULL == ADL_Adapter_ID_Get)
		{
			printf("ADL's API is missing!\n");
			return 0;
		}

        // Initialize ADL. The second parameter is 1, which means:
        // retrieve adapter information only for adapters that are physically present and enabled in the system
        if ( ADL_OK != ADL_Main_Control_Create (ADL_Main_Memory_Alloc, 1) )
	{
		printf("ADL Initialization Error!\n");
		return 0;
	}

        // Obtain the number of adapters for the system
        if ( ADL_OK != ADL_Adapter_NumberOfAdapters_Get ( &iNumberAdapters ) )
	{
		printf("Cannot get the number of adapters!\n");
		return 0;
	}
		
	if (iNumberAdapters == 0)
	{
		printf("No Adapter found\n");
		return(1);
	}
		
	lpAdapterInfo = (AdapterInfo*) malloc( sizeof(AdapterInfo) * iNumberAdapters);
	if (ADL_Adapter_AdapterInfo_Get(lpAdapterInfo, sizeof(AdapterInfo) * iNumberAdapters) != ADL_OK)
	{
		printf("Error getting adapter info\n");
		return(1);
	}

	for (int j = 0;j < 2;j++)
	{
		nAdapters = 0;
		for ( int i = 0; i < iNumberAdapters; i++ )
		{
			int status;
			if (ADL_Adapter_Active_Get(lpAdapterInfo[i].iAdapterIndex, &status) != ADL_OK)
			{
				printf("Error getting adapter status\n");
				return(1);
			}
			if (status == ADL_TRUE)
			{
				if (j)
				{
					nAdapterIndizes[nAdapters] = lpAdapterInfo[i].iAdapterIndex;
#ifdef VERBOSE
					ADLBiosInfo biosInfo;
					ADL_Adapter_VideoBiosInfo_Get(nAdapterIndizes[nAdapters], &biosInfo);
					int UID;
					ADL_Adapter_ID_Get(nAdapterIndizes[nAdapters], &UID);
					printf("Adapter %d Info: Bios %s %s %s, UID %d\n", nAdapters, biosInfo.strPartNumber, biosInfo.strVersion, biosInfo.strDate, UID);
#endif
				}
				nAdapters++;
			}
		}
		if (j == 0) nAdapterIndizes = new int[nAdapters];
	}
	free(lpAdapterInfo);
	return(0);
}

int adl_temperature_check_run(double* max_temperature, int verbose)
{
	*max_temperature = 0.;
	char tmpbuffer[128];
	if (verbose) strcpy(tmpbuffer, "Temperatures:");
	for (int i = 0;i < nAdapters;i++)
	{
		ADLTemperature temp;
		temp.iSize = sizeof(temp);
		if (ADL_Overdrive5_Temperature_Get(nAdapterIndizes[i], 0, &temp) != ADL_OK)
		{
			printf("Error reading temperature from adapter %d\n", i);
			return(1);
		}
		const double temperature = temp.iTemperature / 1000.;
		if (verbose) sprintf(tmpbuffer + strlen(tmpbuffer), " %f", temperature);
		if (temperature > *max_temperature) *max_temperature = temperature;
        }
        if (verbose) fprintf(STD_OUT, "%s\n", tmpbuffer);
        return(0);
}

int adl_temperature_check_exit()
{
    ADL_Main_Control_Destroy ();
#ifdef LINUX
    dlclose(hDLL);
#else
	FreeLibrary(hDLL);
#endif

    return(0);
}

#ifdef MAINPROG
int main (int argc, char** argv)
{
	double temperature;
	if (adl_temperature_check_init())
	{
		printf("Error initializing ADL\n");
		return(1);
	}
	if (adl_temperature_check_run(&temperature, 1))
	{
		printf("Error running ADL temperature check\n");
		return(1);
	}
	printf("Maximum Temperature: %f\n", temperature);
	if (adl_temperature_check_exit())
	{
		printf("Error exiting ADL\n");
		return(1);
	}
}
#endif

#endif
