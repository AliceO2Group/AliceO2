/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
// -------------------------------------------------------------------------
// -----                    FairGenerator header file                   -----
// -----          Created 09/06/04  by D. Bertini / V. Friese          -----
// -------------------------------------------------------------------------

/** FairGenerator.h
 *@author D.Bertini <d.bertini@gsi.de>
 *@author V.Friese  <v.friese@gsi.de>
 *
The FairGenerator is the abtract base class for the generators used to
generate input for the transport simulation.Each concrete generator class
derived from this one must implement the abtract method ReadEvent,
which has to use the method FairPrimaryGenerator::AddTrack.
**/

#ifndef FAIRGENERATOR_H
#define FAIRGENERATOR_H

#include <Rtypes.h>   // for Bool_t, etc
#include <TNamed.h>   // for TNamed

class FairPrimaryGenerator;

class FairGenerator : public TNamed
{
  public:
    /** Default constructor. **/
    FairGenerator();

    /** Constructor with name and title **/
    FairGenerator(const char* name, const char* title = "FAIR Generator");

    /** Destructor. **/
    virtual ~FairGenerator();

    /** Abstract method ReadEvent must be implemented by any derived class.
        It has to handle the generation of input tracks (reading from input
        file) and the handing of the tracks to the FairPrimaryGenerator. I
        t is called from FairMCApplication.
        *@param pStack The stack
        *@return kTRUE if successful, kFALSE if not
    **/
    virtual Bool_t ReadEvent(FairPrimaryGenerator* primGen) = 0;

    /**Initialize the generator if needed */
    virtual Bool_t Init() { return kTRUE; }

    /** Clone this object (used in MT mode only) */
    virtual FairGenerator* CloneGenerator() const;

  protected:
    /** Copy constructor */
    FairGenerator(const FairGenerator&);
    /** Assignment operator */
    FairGenerator& operator=(const FairGenerator&);

    ClassDef(FairGenerator, 1);
};

#endif
