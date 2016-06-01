//-*- Mode: C++ -*-
// @(#) $Id$
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCATRACKERCOMPONENT_H
#define ALIHLTTPCCATRACKERCOMPONENT_H

#include "AliHLTProcessor.h"
#include "AliHLTComponentBenchmark.h"

class AliHLTTPCCATrackerFramework;
class AliHLTTPCCASliceOutput;
class AliHLTTPCCAClusterData;

/**
 * @class AliHLTTPCCATrackerComponent
 * The Cellular Automaton tracker component.
 */
class AliHLTTPCCATrackerComponent : public AliHLTProcessor
{
  public:
    /** standard constructor */
    AliHLTTPCCATrackerComponent();

    /** dummy copy constructor, defined according to effective C++ style */
    AliHLTTPCCATrackerComponent( const AliHLTTPCCATrackerComponent& );

    /** dummy assignment op, but defined according to effective C++ style */
    AliHLTTPCCATrackerComponent& operator=( const AliHLTTPCCATrackerComponent& );

    /** standard destructor */
    virtual ~AliHLTTPCCATrackerComponent();

    // Public functions to implement AliHLTComponent's interface.
    // These functions are required for the registration process

    /** @see component interface @ref AliHLTComponent::GetComponentID */
    const char* GetComponentID() ;

    /** @see component interface @ref AliHLTComponent::GetInputDataTypes */
    void GetInputDataTypes( vector<AliHLTComponentDataType>& list )  ;

    /** @see component interface @ref AliHLTComponent::GetOutputDataType */
    AliHLTComponentDataType GetOutputDataType() ;

    /** @see component interface @ref AliHLTComponent::GetOutputDataSize */
    virtual void GetOutputDataSize( unsigned long& constBase, double& inputMultiplier ) ;

    /** @see component interface @ref AliHLTComponent::Spawn */
    AliHLTComponent* Spawn() ;

  protected:

    // Protected functions to implement AliHLTComponent's interface.
    // These functions provide initialization as well as the actual processing
    // capabilities of the component.

    /** @see component interface @ref AliHLTComponent::DoInit */
    int DoInit( int argc, const char** argv );

    /** @see component interface @ref AliHLTComponent::DoDeinit */
    int DoDeinit();

    /** reconfigure **/
    int Reconfigure( const char* cdbEntry, const char* chainId );

    /** @see component interface @ref AliHLTProcessor::DoEvent */
    int DoEvent( const AliHLTComponentEventData& evtData, const AliHLTComponentBlockData* blocks,
                 AliHLTComponentTriggerData& trigData, AliHLTUInt8_t* outputPtr,
                 AliHLTUInt32_t& size, vector<AliHLTComponentBlockData>& outputBlocks );

  private:

    static const int fgkNSlices = 36;       //* N slices
    static const int fgkNPatches = 6;       //* N slices

    /** the tracker object */
    AliHLTTPCCATrackerFramework* fTracker;                      //! transient
    AliHLTTPCCAClusterData* fClusterData;                       //Storage classes for cluser data in slice
    AliHLTTPCCASliceOutput* fSliceOutput[fgkNSlices];           //Pointers to slice tracker output structures

    //The following parameters are maintained for compatibility to be able to change the component
    //such to process less than all 36 slices. Currently, fMinSlice is always 0 and fSliceCount is 36
    int fMinSlice;                                              //minimum slice number to be processed
    int fSliceCount;                                            //Number of slices to be processed

    /** magnetic field */
    double fSolenoidBz;               // see above
    int fMinNTrackClusters;           //* required min number of clusters on the track
    double fMinTrackPt;               //* required min Pt of tracks
    double fClusterZCut;              //* cut on cluster Z position (for noise rejection at the age of TPC)
    double fNeighboursSearchArea;     //* area in cm for the neighbour search algorithm
    double fClusterErrorCorrectionY;  // correction for the cluster errors
    double fClusterErrorCorrectionZ;  // correction for the cluster errors

    AliHLTComponentBenchmark fBenchmark; // benchmarks
    bool fAllowGPU;                   //* Allow this tracker to run on GPU
    int fGPUHelperThreads;            // Number of helper threads for GPU tracker, set to -1 to use default number
    int fCPUTrackers;                 //Number of CPU trackers to run in addition to GPU tracker
    bool fGlobalTracking;             //Activate global tracking feature
	int fGPUDeviceNum;				  //GPU Device to use, default -1 for auto detection
	TString fGPULibrary;			  //Name of the library file that provides the GPU tracker object

    /** set configuration parameters **/
    void SetDefaultConfiguration();
    int ReadConfigurationString(  const char* arguments );
    int ReadCDBEntry( const char* cdbEntry, const char* chainId );
    int Configure( const char* cdbEntry, const char* chainId, const char *commandLine );
    void ConfigureSlices();

    ClassDef( AliHLTTPCCATrackerComponent, 0 );

};
#endif //ALIHLTTPCCATRACKERCOMPONENT_H
