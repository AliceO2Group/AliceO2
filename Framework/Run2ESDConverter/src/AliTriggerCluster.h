#ifndef ALITRIGGERCLUSTER_H
#define ALITRIGGERCLUSTER_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

///////////////////////////////////////////////////////////////////////////////
//
//  Class to define a Trigger Cluster  
//
//  A Trigger Cluster is a group of detector to be trigger together
//
//////////////////////////////////////////////////////////////////////////////
class TObject;
class TString;

class AliTriggerCluster : public TNamed {

public:
                          AliTriggerCluster();
			  AliTriggerCluster( TString & name, UChar_t index, TString & detectors );
			  AliTriggerCluster( const AliTriggerCluster &clus );
               virtual   ~AliTriggerCluster() {}

  //  Getters
	    const char*   GetDetectorsInCluster() const { return GetTitle(); }
                Bool_t    IsDetectorInCluster( TString & det );
		UChar_t   GetClusterMask() const { return fClusterMask; }

          virtual void    Print( const Option_t* opt ="" ) const;

private:
	       UChar_t    fClusterMask; // The trigger cluster mask pattern
                Bool_t    IsSelected( TString detName, TString & detectors ) const;
		AliTriggerCluster&   operator=(const AliTriggerCluster& clus);

   ClassDef( AliTriggerCluster, 1 )  // Define a Trigger Cluster
};

#endif
