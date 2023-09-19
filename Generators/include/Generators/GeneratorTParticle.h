// -*- C++ -*-
// Copyright 2023 CERN and copyright holders of ALICE O2.
//
// See https://alice-o2.web.cern.ch/copyright for details of the
// copyright holders.  All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General
// Public License v3 (GPL Version 3), copied verbatim in the file
// "COPYING".
//
// In applying this license CERN does not waive the privileges and
// immunities granted to it by virtue of its status as an
// Intergovernmental Organization or submit itself to any
// jurisdiction.
//
/// @author: Christian Holm Christensen <cholm@nbi.dk>
#ifndef ALICEO2_GENERATORTPARTICLE_H_
#define ALICEO2_GENERATORTPARTICLE_H_
#include <FairGenerator.h>
#include <Generators/Generator.h>
#include <list>

// Forward decls
class TChain;
class TParticle;
class TClonesArray;

namespace o2
{
  namespace eventgen
  {
    /// A class that reads in particles of class @c TParticle from a
    /// branch in a @c TChain.
    ///
    /// Optionally, a program that generates such a @c TTree can be
    /// spawn, and the @c TParticles written to a file from which this
    /// object reads them in.  This is done with
    ///
    ///  --configKeyValues "TParticle.progCmd=<eg program and options>"
    ///
    /// which will execute the specified EG program with the given
    /// options.  The EG program _must_ support the command line options
    ///
    ///   -n NUMBER   Number of events to generate
    ///   -o FILENAME Name of file to write to 
    ///
    /// The tree name and particle branch names can be configured.
    ///
    ///   --configKeyValues "TParticle.treeName=T,TParticle.branchName=P"
    ///
    /// File(s) to read are also configurable
    ///
    ///   --configKeyValues "TParticle.fileNames=foo.root,bar.root"
    ///
    class GeneratorTParticle : public Generator 
    {
    public:
      /** CTOR */
      GeneratorTParticle() = default;
      /** CTOR */
      GeneratorTParticle(const std::string& name)
	: Generator(name.c_str(),"ALICEo2 TParticle Generator")
      {}
      /** DTOR */
      virtual ~GeneratorTParticle();

      /** Initialize this generator.  This will set up the chain.
	  Optionally, if a command line was specified by @c
	  TParticle.progCmd then that command line is executed in the
	  background and events are read from the output file of that
	  program */ 
      Bool_t Init() override;

      /** Read in the next entry from the chain.  Returns false in
	  case of errors or no more entries to read. */
      Bool_t generateEvent() override;

      /** Import the read-in particles into the steer particle
	  stack */
      Bool_t importParticles() override;

      /** Set the names of files to read, separated by commas */
      void setFileNames(const std::string& val);
      /** Set the name of the tree in the files.  The tree _must_
	  reside in the top-level directory of the files. */
      void setTreeName(const std::string& val) { mTreeName = val; }
      /** Set the branch name of the branch that holds a @c
	  TClonesArray of @c TParticle objects */ 
      void setBranchName(const std::string& val) { mBranchName = val; }
      /** Set child program command line to (optionally) execute */
      void setProgCmd(const std::string& val) { mProgCmd = val; }
      /** Set the number of events to generate. */
      void setNEvents(unsigned int nev) { mNEvents = nev; }
    protected:
      std::string mTreeName = "T";
      std::string mBranchName = "Particles";
      std::string mProgCmd = "";
      std::list<std::string> mFileNames;
      unsigned int mNEvents = 0;
      unsigned int mEntry = 0;
      TChain* mChain;
      TClonesArray* mTParticles;

      void waitForData();

      ClassDefOverride(GeneratorTParticle,1);
    };
  }
}
#endif
//
// EOF
//

      
	

