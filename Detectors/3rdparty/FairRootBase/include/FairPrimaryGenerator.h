/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH *
 *                                                                              *
 *              This software is distributed under the terms of the *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE" *
 ********************************************************************************/

/** FairPrimaryGenerator.h
 *@author V.Friese  <v.friese@gsi.de>
 *
The FairPrimaryGenerator is responsible for the handling of the MC input.
Several input generators can be registered to it; these have to be
derived from the FairGenerator class. The FairPrimaryGenerator defines
position and (optionally) smearing of the primary vertex.
This class should be instantised only once.

Modified 05.06.07:  add a method DoTracking(Bool_t) to be able to switch on/off
the tracking from the macro (M. Al-Turany)

**/

#ifndef FAIRPRIMARYGENERATOR_H
#define FAIRPRIMARYGENERATOR_H

#include "FairGenerator.h"   // for FairGenerator

#include <Rtypes.h>   // for Double_t, Bool_t, Int_t, etc
#include <TMCProcess.h>
#include <TNamed.h>      // for TNamed
#include <TObjArray.h>   // for TObjArray
#include <TVector3.h>    // for TVector3
#include <iostream>      // for cout

class FairGenericStack;
class FairMCEventHeader;
class TIterator;

class FairPrimaryGenerator : public TNamed
{

  public:
    /** Default constructor. **/
    FairPrimaryGenerator();

    /** Constructor with name and title **/
    FairPrimaryGenerator(const char *name, const char *title = "FAIR Generator");

    /** Destructor. **/
    virtual ~FairPrimaryGenerator();

    /** Initialize the generater (if needed!)*/
    virtual Bool_t Init();

    /** Register a generator derived from FairGenerator.  **/
    void AddGenerator(FairGenerator *generator)
    {
        if (!fGenList) {
            std::cout << "Empty fGenList pointer ! " << std::endl;
            return;
        }
        fGenList->Add(generator);
    }

    void SetEventNr(Int_t evtNr) { fEventNr = evtNr; }

    /** Public method GenerateEvent
      To be called at the beginning of each event from FairMCApplication.
      Generates an event vertex and calls the ReadEvent methods from the
      registered generators.
      *@param pStack The particle stack
      *@return kTRUE if successful, kFALSE if not
      **/
    virtual Bool_t GenerateEvent(FairGenericStack *pStack);

    /** Public method AddTrack
      Adding a track to the MC stack. To be called within the ReadEvent
      methods of the registered generators.
      *@param pdgid Particle ID (PDG code)
      *@param px,py,pz Momentum coordinates [GeV]
      *@param vx,vy,vz Track origin relative to event vertex
      **/
    virtual void AddTrack(Int_t pdgid,
                          Double_t px,
                          Double_t py,
                          Double_t pz,
                          Double_t vx,
                          Double_t vy,
                          Double_t vz,
                          Int_t parent = -1,
                          Bool_t wanttracking = true,
                          Double_t e = -9e9,
                          Double_t tof = 0.,
                          Double_t weight = 0.,
                          TMCProcess proc = kPPrimary);

    /** Clone this object (used in MT mode only) */
    virtual FairPrimaryGenerator *ClonePrimaryGenerator() const;

    /** Set beam position and widths.
     *@param beamX0      mean x position of beam at target
     *@param beamY0      mean y position of beam at target
     *@param beamSigmaX  Gaussian beam width in x
     *@param beamSigmaY  Gaussian beam width in y
     **/
    void SetBeam(Double_t beamX0, Double_t beamY0, Double_t beamSigmaX, Double_t beamSigmaY);

    /** Set nominal beam angle and angle widths.
     *@param beamAngleX0      mean x angle of beam at target
     *@param beamAngleY0      mean y angle of beam at target
     *@param beamAngleSigmaX  Gaussian beam angle width in x
     *@param beamAngleSigmaY  Gaussian beam angle width in y
     **/
    void SetBeamAngle(Double_t beamAngleX0, Double_t beamAngleY0, Double_t beamAngleSigmaX, Double_t beamAngleSigmaY);

    /** Public method SetEventPlane
     **@param phiMin   Lower limit for event plane angle [rad]
     **@param phiMax   Upper limit for event plane angle [rad]
     **If set, an event plane angle will be generated with flat
     **distrtibution between phiMin and phiMax.
     **/
    void SetEventPlane(Double_t phiMin, Double_t phiMax);

    /** Set target position and thickness.
     *@param targetZ   z position of target center
     *@param targetDz  full target thickness
     **/
    void SetTarget(Double_t targetZ, Double_t targetDz);

    /** Set target position for multiple tagets. The thickness
     * is the same for all targets.
     *@param nroftargets   number of targets
     *@param *targetZ      z positions of target center
     *@param targetDz      full target thickness
     **/
    void SetMultTarget(Int_t nroftargets, Double_t *targetZ, Double_t targetDz);

    /** Enable vertex smearing in z and/or xy direction **/
    void SmearVertexZ(Bool_t flag);
    void SmearGausVertexZ(Bool_t flag);
    void SmearVertexXY(Bool_t flag);
    void SmearGausVertexXY(Bool_t flag);

    TObjArray *GetListOfGenerators() { return fGenList; }

    /** Set the pointer to the MCEvent **/
    void SetEvent(FairMCEventHeader *event) { fEvent = event; };

    /** Accessor to the MCEvent **/
    FairMCEventHeader *GetEvent() { return fEvent; };

    /** Swich on/off the tracking of a particle*/

    void DoTracking(Bool_t doTracking = kTRUE) { fdoTracking = doTracking; }

    Int_t GetTotPrimary() { return fTotPrim; }

  protected:
    /**  Copy constructor */
    FairPrimaryGenerator(const FairPrimaryGenerator &);
    /**  Assignment operator */
    FairPrimaryGenerator &operator=(const FairPrimaryGenerator &);

    /**  Nominal beam position at target in x [cm] */
    Double_t fBeamX0;
    /** Nominal beam position at target in y [cm]*/
    Double_t fBeamY0;
    /** Beam width (Gaussian) in x [cm]*/
    Double_t fBeamSigmaX;
    /**  Beam width (Gaussian) in y [cm]*/
    Double_t fBeamSigmaY;

    /**  Nominal beam angle at target in x [rad] */
    Double_t fBeamAngleX0;
    /**  Nominal beam angle at target in y [rad] */
    Double_t fBeamAngleY0;
    /**  Actual beam angle at target in x [rad] */
    Double_t fBeamAngleX;
    /**  Actual beam angle at target in y [rad] */
    Double_t fBeamAngleY;
    /** Beam angle width (Gaussian) in x [rad]*/
    Double_t fBeamAngleSigmaX;
    /** Beam angle width (Gaussian) in y [rad]*/
    Double_t fBeamAngleSigmaY;
    /** Actual beam direction at the vertex */
    TVector3 fBeamDirection;

    /** Lower limit for the event plane rotation angle [rad] */
    Double_t fPhiMin;
    /** Upper limit for the event plane rotation angle [rad] */
    Double_t fPhiMax;
    /** Actual event plane rotation angle [rad] */
    Double_t fPhi;

    /**  Nominal z position of center of targets [cm]*/
    Double_t *fTargetZ;   //!
    /**  Number of targets;*/
    Int_t fNrTargets;
    /**  Full target thickness [cm]*/
    Double_t fTargetDz;

    /** Vertex position of current event [cm]*/
    TVector3 fVertex;

    /** Number of primary tracks in current event*/
    Int_t fNTracks;

    /**  Flag for uniform vertex smearing in z*/
    Bool_t fSmearVertexZ;
    /**  Flag for gaus vertex smearing in z*/
    Bool_t fSmearGausVertexZ;
    /**  Flag for vertex smearing in xy*/
    Bool_t fSmearVertexXY;
    /**  Flag for gaus vertex smearing in xy*/
    Bool_t fSmearGausVertexXY;
    /**  Flag for beam gradient calculation*/
    Bool_t fBeamAngle;
    /**  Flag for event plane rotation*/
    Bool_t fEventPlane;

    /**  Pointer to MC stack*/
    FairGenericStack *fStack;   //!
    /**  List of registered generators */
    TObjArray *fGenList;
    /**  Iterator over generator list */
    TIterator *fListIter;   //!
    /**  Pointer to MCEventHeader */
    FairMCEventHeader *fEvent;   //!
    /** go to tracking  */
    Bool_t fdoTracking;   //!
    /** Number of MC tracks before a Generator is called, needed for MC index
     * update */
    Int_t fMCIndexOffset;   //!
    /** Number of all primaries of this run*/
    static Int_t fTotPrim;   //!
    /** Event number (Set by the primary generator if not set already by one of
      the specific generators
   **/
    Int_t fEventNr;

    /** Private method MakeVertex. If vertex smearing in xy is switched on,
      the event vertex is smeared Gaussianlike in x and y direction
      according to the mean beam positions and widths set by the
      SetBeam method. If vertex smearing in z is switched on, the z
      coordinate of the event vertex is flatly distributed over the
      extension of the target.
      To be called at the beginning of the event from the
      GenerateEvent method.
  **/
    virtual void MakeVertex();

    /** Private method MakeBeamAngle. If beam angle smearing in xy
  is switched on, all tracks in an event are rotated by a Gaussianlike
  angle distribution around the x and y axis according to the mean
  beam angle and angle widths set by the SetBeamAngle method.
  To be called at the beginning of the event from the GenerateEvent
  method.
  **/
    virtual void MakeBeamAngle();

    /** Private method MakeEventPlane. If the rotation of the event around the
      z-axis by a random angle is switched on, the complete event is rotated
      by the chosen angle. This function is called at the beginning of the
      event from the GenerateEvent method. The function pick a random
      rotation angle between fPhiMin and fPhiMax which are set using the
      function SetEventPlane.
  **/
    void MakeEventPlane();

    ClassDef(FairPrimaryGenerator, 5);
};

#endif
