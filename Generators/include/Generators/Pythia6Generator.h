// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
// -------------------------------------------------------------------------
// -----                  M. Al-Turany   June 2014                     -----
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
// -----                Pythia6Generator header file                 -----
// -----          Created 08/08/08  by S. Spataro                      -----
// -------------------------------------------------------------------------

/**  Pythia6Generator.h 
 *@author S.Spataro  <spataro@to.infn.it>
 *
 The Pythia6Generator reads a Pythia6 input file. The file must contain
 for each event a header line of the format:

 [start]
1 20
         3     -2212         0         0         0         0  0.00000000E+00  0.00000000E+00  0.14000000E+02  0.14031406E+02  0.93827000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
         3      2212         0         0         0         0  0.00000000E+00  0.00000000E+00 -0.13444107E-16  0.93827000E+00  0.93827000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
         3        -2         1         0         0         0 -0.70661074E+00 -0.81156104E+00  0.49379331E+01  0.50538217E+01  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
         3         2         2         0         0         0 -0.79043780E+00  0.32642680E+00  0.10299757E+01  0.13387293E+01  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
         3        -2         3         0         0         0 -0.70661074E+00 -0.81156104E+00  0.49379331E+01  0.50538217E+01  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
         3         2         4         0         0         0 -0.79043780E+00  0.32642680E+00  0.10299757E+01  0.13387293E+01  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
         3        23         5         6         0         0 -0.14970485E+01 -0.48513424E+00  0.59679088E+01  0.63925510E+01  0.16650116E+01  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
         3        11         7         0         0         0 -0.16024130E+01  0.20883507E-01  0.32233123E+01  0.35997091E+01  0.51000000E-03  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
         3       -11         7         0         0         0  0.10536443E+00 -0.50601774E+00  0.27445965E+01  0.27928419E+01  0.51000000E-03  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
         2        23         7         0        11        13 -0.14970485E+01 -0.48513424E+00  0.59679088E+01  0.63925510E+01  0.16650116E+01  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
         1       -11         9         0         0         0  0.10536442E+00 -0.50601769E+00  0.27445962E+01  0.27928416E+01  0.51000000E-03  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
         1        11         8         0         0         0 -0.16017105E+01  0.20863360E-01  0.32218964E+01  0.35981284E+01  0.51000000E-03  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
         1        22         8         0         0         0 -0.70246172E-03  0.20096726E-04  0.14161816E-02  0.15809575E-02  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
         2     -2103         1         0        16        16  0.70661074E+00  0.81156104E+00  0.69735272E+01  0.70980957E+01  0.77133000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
         2      2101         2         0        16        16  0.79043780E+00 -0.32642680E+00  0.10585640E+01  0.14790292E+01  0.57933000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
         2        92        14        15        17        18  0.14970485E+01  0.48513424E+00  0.80320912E+01  0.85771248E+01  0.25643853E+01  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
         2     -2214        16         0        19        20  0.81382620E+00  0.70515618E+00  0.61830559E+01  0.63829315E+01  0.11627878E+01  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
         1      2212        16         0         0         0  0.68322234E+00 -0.22002195E+00  0.18490353E+01  0.21941934E+01  0.93827000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
         1     -2112        17         0         0         0  0.78494059E+00  0.52384336E+00  0.55944336E+01  0.57507411E+01  0.93957000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
         1      -211        17         0         0         0  0.28885610E-01  0.18131282E+00  0.58862227E+00  0.63219038E+00  0.13957000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
 2 25
         3     -2212         0         0         0         0  0.00000000E+00  0.00000000E+00  0.14000000E+02  0.14031406E+02  0.93827000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
         3      2212         0         0         0         0  0.00000000E+00  0.00000000E+00 -0.13444107E-16  0.93827000E+00  0.93827000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00

  
...
 [stop]
 
 where the first row has the number of event and the number of particles, and below "N" is the line
 number of the event, 

 Derived from FairGenerator.
**/

#ifndef PND_PYTHIAGENERATOR_H
#define PND_PYTHIAGENERATOR_H

#ifdef __CLING__
#define _DLFCN_H_
#define _DLFCN_H
#endif

#include <cstdio>           // for FILE
#include "FairGenerator.h"  // for FairGenerator
#include "Rtypes.h"         // for Int_t, Pythia6Generator::Class, Bool_t, etc
class FairPrimaryGenerator; // lines 68-68

namespace o2
{
namespace eventgen
{

class Pythia6Generator : public FairGenerator
{

 public:
  /** Default constructor without arguments should not be used. **/
  Pythia6Generator();

  /** Standard constructor. 
   ** @param fileName The input file name
   **/
  Pythia6Generator(const char* fileName);

  /** Destructor. **/
  ~Pythia6Generator() override;

  /** Reads on event from the input file and pushes the tracks onto
   ** the stack. Abstract method in base class.
   ** @param primGen  pointer to the CbmrimaryGenerator
   **/
  Bool_t ReadEvent(FairPrimaryGenerator* primGen) override;

  void SetVerbose(Int_t verb) { mVerbose = verb; };

 private:
  const Char_t* mFileName; //! Input file Name
  FILE* mInputFile;        //! File
  Int_t mVerbose;          //! Verbose Level

  /** Private method CloseInput. Just for convenience. Closes the 
   ** input file properly. Called from destructor and from ReadEvent. **/
  void CloseInput();

  /** PDG data base */

  //  TDatabasePDG *mPDG; //!

  ClassDefOverride(Pythia6Generator, 1);
};

} // namespace eventgen
} // namespace o2
#endif
