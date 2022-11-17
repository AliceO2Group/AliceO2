// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <boost/program_options.hpp>
#include <string>
#include <iostream>
#include <Algorithm/RangeTokenizer.h>
#include <regex>
#include "Steer/InteractionSampler.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "SimulationDataFormat/DigitizationContext.h"
#include <cmath>
#include <TRandom.h>
#include <numeric>
#include <fairlogger/Logger.h>
#include "Steer/MCKinematicsReader.h"

//
// Created by Sandro Wenzel on 13.07.21.
//

// A utility to create/engineer (later modify/display) collision contexts

// options struct filled from command line
struct Options {
  std::vector<std::string> interactionRates;
  std::string outfilename;    //
  double timeframelengthinMS; // timeframe length in milliseconds
  int orbits;                 // number of orbits to generate (can be a multiple of orbitsPerTF --> determine fraction or multiple of timeframes)
  long seed;                  //
  bool printContext = false;
  std::string bcpatternfile;
  int tfid = 0;          // tfid -> used to calculate start orbit for collisions
  int orbitsPerTF = 256; // number of orbits per timeframe --> used to calculate start orbit for collisions
  bool useexistingkinematics = false;
  bool noEmptyTF = false; // prevent empty timeframes; the first interaction will be shifted backwards to fall within the range given by Options.orbits
};

enum class InteractionLockMode {
  NOLOCK,
  EVERYN,
  MINTIMEDISTANCE
};

struct InteractionSpec {
  std::string name; // name (prefix for transport simulation); may also serve as unique identifier
  float interactionRate;
  std::pair<int, float> synconto; // if this interaction locks on another interaction; takes precedence over interactionRate
  InteractionLockMode syncmode = InteractionLockMode::NOLOCK;
  char syncmodeop = 0;         // syncmode operation ("@" --> embedd; "r" --> replace)
  int mcnumberasked = -1;      // number of MC events asked (but can be left -1) in which case it will be determined from timeframelength
  int mcnumberavail = -1;      // number of MC events avail (but can be left -1); if avail < asked there will be reuse of events
  bool randomizeorder = false; // whether order of events will be randomized
};

InteractionSpec parseInteractionSpec(std::string const& specifier, std::vector<InteractionSpec> const& existingPatterns, bool adjustEventCount)
{
  // An interaction specification is a command-separated string
  // of the following form:
  // SPEC=NAMESTRING,INTERACTIONSTRING[,MCNUMBERSTRING]
  //
  // where
  //
  // NAMESTRING : a simple named specifier for the interaction; matching to a simulation prefix used by o2-sim
  //
  // INTERACTIONSTRING: irate | @ID:[ed]FLOATVALUE
  //      - either: a simple number irate specifying the interaction rate in kHz
  //      -     or: a string such as @0:e5, saying that this interaction should match/sync
  //                with collisions of the 0-th interaction, but inject only every 5 collisions.
  //                Alternatively @0:d10000 means to inject but leaving a timedistance of at least 10000ns between signals
  //      -     or: a string r0:e5, saying that this interaction should sync with collisions of the 0-th interaction but
  //                **overwrite** every 5-th interaction with a collision from this interaction name
  // MCNUMBERSTRING: NUMBER1:r?NUMBER2 can specify how many collisions NUMBER1 to produce, taking from a sample of NUMBER2 available collisions
  //      - this option is only supported on the first interaction which is supposed to be the background interaction
  //      - if the 'r' character is present we randomize the order of the MC events

  // tokens are separated by comma
  std::vector<std::string> tokens = o2::RangeTokenizer::tokenize<std::string>(specifier);

  float rate = -1.;
  std::pair<int, float> synconto(-1, 1);

  // extract (kinematics prefix) name
  std::string name = tokens[0];

  // extract the MC number spec if given
  int collisionsasked = -1;
  int collisionsavail = -1;
  bool randomizeorder = false;
  if (tokens.size() > 2) {
    auto mctoken = tokens[2];
    std::regex re("([0-9]*):(r?)([0-9]*)$", std::regex_constants::extended);

    std::cmatch m;
    if (std::regex_match(mctoken.c_str(), m, re)) {
      collisionsasked = std::atoi(m[1].str().c_str());
      if (m[2].str().compare("r") == 0) {
        randomizeorder = true;
      }
      collisionsavail = std::atoi(m[3].str().c_str());
    } else {
      LOG(error) << "Could not parse " << mctoken << " as MCNUMBERSTRING";
      exit(1);
    }
  }

  if (adjustEventCount) {
    // if the number of collisionsavail has not been specified, we should
    // try to extract it from the kinematics directly
    o2::steer::MCKinematicsReader mcreader(name, o2::steer::MCKinematicsReader::Mode::kMCKine);
    if (collisionsavail > 0) {
      collisionsavail = std::min((size_t)collisionsavail, (size_t)mcreader.getNEvents(0));
    } else {
      collisionsavail = mcreader.getNEvents(0);
    }
  }

  // extract interaction rate ... or locking
  auto& interactionToken = tokens[1];
  if (interactionToken[0] == '@' || interactionToken[0] == 'r') {
    try {
      // locking onto some other interaction
      std::regex re("[@r]([0-9]*):([ed])([0-9]*[.]?[0-9]?)$", std::regex_constants::extended);

      std::cmatch m;
      if (std::regex_match(interactionToken.c_str(), m, re)) {
        auto crossindex = std::atoi(m[1].str().c_str());
        auto mode = m[2].str();
        auto modevalue = std::atof(m[3].str().c_str());

        if (crossindex > existingPatterns.size()) {
          LOG(error) << "Reference to non-existent interaction spec";
          exit(1);
        }
        synconto = std::pair<int, float>(crossindex, modevalue);

        InteractionLockMode lockMode;
        if (mode.compare("e") == 0) {
          lockMode = InteractionLockMode::EVERYN;
        }
        if (mode.compare("d") == 0) {
          lockMode = InteractionLockMode::MINTIMEDISTANCE;
        }
        return InteractionSpec{name, rate, synconto, lockMode, interactionToken[0], collisionsasked, collisionsavail, randomizeorder};
      } else {
        LOG(error) << "Could not parse " << interactionToken << " as INTERACTIONSTRING";
        exit(1);
      }
    } catch (std::regex_error e) {
      LOG(error) << "Exception during regular expression match " << e.what();
      exit(1);
    }
  } else {
    rate = std::atof(interactionToken.c_str());
    return InteractionSpec{name, rate, synconto, InteractionLockMode::NOLOCK, 0, collisionsasked, collisionsavail, randomizeorder};
  }
}

bool parseOptions(int argc, char* argv[], Options& optvalues)
{
  namespace bpo = boost::program_options;
  bpo::options_description options(
    "A utility to create and manipulate digitization contexts (MC collision structure within a timeframe).\n\n"
    "Allowed options");

  options.add_options()(
    "interactions,i", bpo::value<std::vector<std::string>>(&optvalues.interactionRates)->multitoken(), "name,IRate|LockSpecifier")(
    "outfile,o", bpo::value<std::string>(&optvalues.outfilename)->default_value("collisioncontext.root"), "Outfile of collision context")(
    "orbits", bpo::value<int>(&optvalues.orbits)->default_value(-1),
    "Number of orbits to generate maximally (if given, can be used to determine the number of timeframes). "
    "Otherwise, the context will be generated by using collision numbers from the interaction specification.")(
    "seed", bpo::value<long>(&optvalues.seed)->default_value(0L), "Seed for random number generator (for time sampling etc). Default 0: Random")(
    "show-context", "Print generated collision context to terminal.")(
    "bcPatternFile", bpo::value<std::string>(&optvalues.bcpatternfile)->default_value(""), "Interacting BC pattern file (e.g. from CreateBCPattern.C)")(
    "orbitsPerTF", bpo::value<int>(&optvalues.orbitsPerTF)->default_value(256), "Orbits per timeframes")(
    "use-existing-kine", "Read existing kinematics to adjust event counts")(
    "timeframeID", bpo::value<int>(&optvalues.tfid)->default_value(0), "Timeframe id of the first timeframe int this context. Allows to generate contexts for different start orbits")(
    "maxColsPerTF", bpo::value<int>(&optvalues.orbitsPerTF)->default_value(-1), "Maximal number of MC collisions to put into one timeframe.")(
    "noEmptyTF", bpo::bool_switch(&optvalues.noEmptyTF), "Enforce to have at least one collision");

  options.add_options()("help,h", "Produce help message.");

  bpo::variables_map vm;
  try {
    bpo::store(bpo::command_line_parser(argc, argv).options(options).run(), vm);
    bpo::notify(vm);

    // help
    if (vm.count("help")) {
      std::cout << options << std::endl;
      return false;
    }
    if (vm.count("show-context")) {
      optvalues.printContext = true;
    }
    if (vm.count("use-existing-kine")) {
      optvalues.useexistingkinematics = true;
    }

  } catch (const bpo::error& e) {
    std::cerr << e.what() << "\n\n";
    std::cerr << "Error parsing options; Available options:\n";
    std::cerr << options << std::endl;
    return false;
  }
  return true;
}

int main(int argc, char* argv[])
{
  Options options;
  if (!parseOptions(argc, argv, options)) {
    exit(1);
  }

  // init random generator
  gRandom->SetSeed(options.seed);

  std::vector<InteractionSpec> ispecs;
  // building the interaction spec
  for (auto& i : options.interactionRates) {
    // this is created as output from
    ispecs.push_back(parseInteractionSpec(i, ispecs, options.useexistingkinematics));
  }

  std::vector<std::pair<o2::InteractionTimeRecord, std::vector<o2::steer::EventPart>>> collisions;
  std::vector<o2::BunchFilling> bunchFillings; // vector of bunch filling objects; generated by interaction samplers

  // now we generate the collision structure (interaction type by interaction type)
  bool usetimeframelength = options.orbits > 0;
  o2::InteractionTimeRecord limitInteraction(0, options.orbits);

  for (int id = 0; id < ispecs.size(); ++id) {
    auto mode = ispecs[id].syncmode;
    if (mode == InteractionLockMode::NOLOCK) {
      o2::steer::InteractionSampler sampler;
      sampler.setInteractionRate(ispecs[id].interactionRate);
      if (!options.bcpatternfile.empty()) {
        sampler.setBunchFilling(options.bcpatternfile);
      }
      auto orbitstart = options.tfid * options.orbitsPerTF;
      o2::InteractionTimeRecord record;
      // this loop makes sure that the first collision is within the range of orbits asked (if noEmptyTF is enabled)
      do {
        sampler.setFirstIR(o2::InteractionRecord(0, orbitstart));
        sampler.init();
        record = sampler.generateCollisionTime();
      } while (options.noEmptyTF && usetimeframelength && record.orbit >= orbitstart + options.orbits);
      int count = 0;
      do {
        if (usetimeframelength && record.orbit >= orbitstart + options.orbits) {
          break;
        }
        std::vector<o2::steer::EventPart> parts;
        parts.emplace_back(id, count);

        std::pair<o2::InteractionTimeRecord, std::vector<o2::steer::EventPart>> insertvalue(record, parts);
        auto iter = std::lower_bound(collisions.begin(), collisions.end(), insertvalue, [](std::pair<o2::InteractionTimeRecord, std::vector<o2::steer::EventPart>> const& a, std::pair<o2::InteractionTimeRecord, std::vector<o2::steer::EventPart>> const& b) { return a.first < b.first; });
        collisions.insert(iter, insertvalue);
        record = sampler.generateCollisionTime();
        count++;
      } while ((ispecs[id].mcnumberasked > 0 && count < ispecs[id].mcnumberasked));

      // we support randomization etc on non-injected/embedded interactions
      // and we can apply them here
      auto random_shuffle = [](auto first, auto last) {
        auto n = last - first;
        for (auto i = n - 1; i > 0; --i) {
          using std::swap;
          swap(first[i], first[(int)(gRandom->Rndm() * n)]);
        }
      };
      std::vector<int> eventindices(count);
      std::iota(eventindices.begin(), eventindices.end(), 0);
      // apply randomization of order if any
      if (ispecs[id].randomizeorder) {
        random_shuffle(eventindices.begin(), eventindices.end());
      }
      if (ispecs[id].mcnumberavail > 0) {
        // apply cutting to number of available entries
        for (auto& e : eventindices) {
          e = e % ispecs[id].mcnumberavail;
        }
      }
      // make these transformations final:
      for (auto& col : collisions) {
        for (auto& part : col.second) {
          if (part.sourceID == id) {
            part.entryID = eventindices[part.entryID];
          }
        }
      }

      // keep bunch filling information produced by these samplers
      bunchFillings.push_back(sampler.getBunchFilling());

    } else {
      // we are in some lock/sync mode and modify existing collisions
      int lastcol = -1;
      double lastcoltime = -1.;
      auto distanceval = ispecs[id].synconto.second;
      auto lockonto = ispecs[id].synconto.first;
      int eventcount = 0;

      for (int colid = 0; colid < collisions.size(); ++colid) {
        auto& col = collisions[colid];
        auto coltime = col.first.getTimeNS();

        bool rightinteraction = false;
        // we are locking only on collisions which have the referenced interaction present
        // --> there must be an EventPart with the right sourceID
        for (auto& eventPart : col.second) {
          if (eventPart.sourceID == lockonto) {
            rightinteraction = true;
            break;
          }
        }
        if (!rightinteraction) {
          continue;
        }

        bool inject = false;
        // we always start with first one
        if (lastcol == -1) {
          inject = true;
        }
        if (mode == InteractionLockMode::EVERYN && (colid - lastcol) >= distanceval) {
          inject = true;
        }
        if (mode == InteractionLockMode::MINTIMEDISTANCE && (coltime - lastcoltime) >= distanceval) {
          inject = true;
        }

        if (inject) {
          if (ispecs[id].syncmodeop == 'r') {
            LOG(debug) << "Replacing/overwriting another event ";
            // Syncing is replacing; which means we need to take out the original
            // event that we locked onto.
            // We take out this event part immediately (and complain if there is a problem).
            int index = 0;
            auto iter = std::find_if(col.second.begin(), col.second.end(), [lockonto](auto val) { return lockonto == val.sourceID; });
            if (iter != col.second.end()) {
              col.second.erase(iter);
            } else {
              LOG(error) << "Expected to replace another event part but did not find one for source " << lockonto << " and collision " << colid;
            }
          }

          if (ispecs[id].mcnumberavail >= 0) {
            col.second.emplace_back(id, eventcount % ispecs[id].mcnumberavail);
          } else {
            col.second.emplace_back(id, eventcount);
          }
          eventcount++;
          lastcol = colid;
          lastcoltime = coltime;
        }
      }
    }
  }

  // create DigitizationContext
  o2::steer::DigitizationContext digicontext;
  // we can fill this container
  auto& parts = digicontext.getEventParts();
  // we can fill this container
  auto& records = digicontext.getEventRecords();
  // copy over information
  size_t maxParts = 0;
  for (auto& p : collisions) {
    records.push_back(p.first);
    parts.push_back(p.second);
    maxParts = std::max(p.second.size(), maxParts);
  }
  digicontext.setNCollisions(collisions.size());
  digicontext.setMaxNumberParts(maxParts);
  // merge bunch filling info
  for (int i = 1; i < bunchFillings.size(); ++i) {
    bunchFillings[0].mergeWith(bunchFillings[i]);
  }
  digicontext.setBunchFilling(bunchFillings[0]);
  std::vector<std::string> prefixes;
  for (auto& p : ispecs) {
    prefixes.push_back(p.name);
  }
  digicontext.setSimPrefixes(prefixes);

  digicontext.finalizeTimeframeStructure(options.tfid * options.orbitsPerTF, options.orbitsPerTF);
  if (options.printContext) {
    digicontext.printCollisionSummary();
  }
  digicontext.saveToFile(options.outfilename);

  // convert large collision context to individual timeframe contexts
  // (apply MC limits, split into individual timeframes; adjust time within; adjust signal id)

  return 0;
}
