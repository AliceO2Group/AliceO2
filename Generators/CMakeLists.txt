# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

if(pythia6_FOUND)
  set(pythia6Target ROOT::EGPythia6 MC::Pythia6)
endif()

if(pythia_FOUND)
  set(pythiaTarget MC::Pythia)
endif()

if(HepMC3_FOUND)
  set(hepmcTarget MC::HepMC3)
endif()

o2_add_library(Generators
               SOURCES src/Generator.cxx
                       src/Trigger.cxx
                       src/TriggerParticle.cxx
                       src/GeneratorTGenerator.cxx
                       src/GeneratorExternalParam.cxx
                       src/GeneratorFromFile.cxx
                       src/GeneratorFromO2KineParam.cxx
                       src/GeneratorFileOrCmd.cxx
                       src/GeneratorFileOrCmdParam.cxx
                       src/PrimaryGenerator.cxx
                       src/PrimaryGeneratorParam.cxx
                       src/TriggerExternalParam.cxx
                       src/TriggerParticleParam.cxx
                       src/BoxGunParam.cxx
                       src/QEDGenParam.cxx
                       src/GenCosmicsParam.cxx
                       src/GeneratorFactory.cxx
                       src/GeneratorGeantinos.cxx
                       src/GeneratorTParticle.cxx
                       src/GeneratorTParticleParam.cxx
                       $<$<BOOL:${pythia6_FOUND}>:src/GeneratorPythia6.cxx>
                       $<$<BOOL:${pythia6_FOUND}>:src/GeneratorPythia6Param.cxx>
                       $<$<BOOL:${pythia_FOUND}>:src/GeneratorPythia8.cxx>
                       $<$<BOOL:${pythia_FOUND}>:src/DecayerPythia8.cxx>
                       $<$<BOOL:${pythia_FOUND}>:src/GeneratorPythia8Param.cxx>
                       $<$<BOOL:${pythia_FOUND}>:src/DecayerPythia8Param.cxx>
                       $<$<BOOL:${HepMC3_FOUND}>:src/GeneratorHepMC.cxx>
                       $<$<BOOL:${HepMC3_FOUND}>:src/GeneratorHepMCParam.cxx>
               PUBLIC_LINK_LIBRARIES FairRoot::Base O2::SimConfig O2::CommonUtils O2::DetectorsBase O2::ZDCBase
                                     O2::SimulationDataFormat ${pythia6Target} ${pythiaTarget} ${hepmcTarget}
                                     FairRoot::Gen
               TARGETVARNAME targetName)

if(pythia6_FOUND)
  target_compile_definitions(${targetName} PUBLIC GENERATORS_WITH_PYTHIA6)
endif()

if(pythia_FOUND)
  target_compile_definitions(${targetName} PUBLIC GENERATORS_WITH_PYTHIA8)
endif()

if(HepMC3_FOUND)
  target_compile_definitions(${targetName} PUBLIC GENERATORS_WITH_HEPMC3)
endif()

set(headers
    include/Generators/Generator.h
    include/Generators/Trigger.h
    include/Generators/TriggerParticle.h
    include/Generators/GeneratorTGenerator.h
    include/Generators/GeneratorExternalParam.h
    include/Generators/GeneratorFromFile.h
    include/Generators/GeneratorFromO2KineParam.h
    include/Generators/GeneratorFileOrCmd.h
    include/Generators/GeneratorFileOrCmdParam.h
    include/Generators/PrimaryGenerator.h
    include/Generators/PrimaryGeneratorParam.h
    include/Generators/TriggerExternalParam.h
    include/Generators/TriggerParticleParam.h
    include/Generators/BoxGunParam.h
    include/Generators/QEDGenParam.h
    include/Generators/GenCosmicsParam.h
    include/Generators/GeneratorGeantinos.h
    include/Generators/GeneratorTParticle.h
    include/Generators/GeneratorTParticleParam.h
    )

if (pythia6_FOUND)
   list(APPEND headers include/Generators/GeneratorPythia6.h
                       include/Generators/GeneratorPythia6Param.h)
endif()

if(pythia_FOUND)
  list(APPEND headers
              include/Generators/GeneratorPythia8.h
              include/Generators/DecayerPythia8.h
              include/Generators/GeneratorPythia8Param.h
              include/Generators/DecayerPythia8Param.h
              include/Generators/GeneratorFactory.h)
endif()

if(HepMC3_FOUND)
  list(APPEND headers include/Generators/GeneratorHepMC.h)
  list(APPEND headers include/Generators/GeneratorHepMCParam.h)
endif()

o2_target_root_dictionary(Generators HEADERS ${headers})

o2_add_test_root_macro(share/external/extgen.C
                       PUBLIC_LINK_LIBRARIES O2::Generators FairRoot::Base
                       LABELS generators COMPILE_ONLY)
if(pythia6_FOUND)
  o2_add_test_root_macro(share/external/pythia6.C
                         PUBLIC_LINK_LIBRARIES O2::Generators
                         LABELS generators COMPILE_ONLY)
endif()

if(doBuildSimulation)
  o2_add_test_root_macro(share/external/QEDLoader.C
                         PUBLIC_LINK_LIBRARIES O2::Generators
                                               O2::SimConfig
                                               LABELS generators)

  o2_add_test_root_macro(share/external/GenCosmicsLoader.C
                         PUBLIC_LINK_LIBRARIES O2::Generators
                                               O2::SimConfig
                                               LABELS generators)
endif()

o2_add_test_root_macro(share/external/tgenerator.C
                       PUBLIC_LINK_LIBRARIES O2::Generators
                       LABELS generators)

o2_add_test_root_macro(share/external/trigger_multiplicity.C
                       PUBLIC_LINK_LIBRARIES O2::Generators
                       LABELS generators)

if(pythia_FOUND)
o2_add_test_root_macro(share/external/trigger_mpi.C
                       PUBLIC_LINK_LIBRARIES O2::Generators
                       LABELS generators)

o2_add_test_root_macro(share/egconfig/pythia8_userhooks_charm.C
        PUBLIC_LINK_LIBRARIES O2::Generators
                       LABELS generators)
endif()

o2_data_file(COPY share/external DESTINATION Generators)
o2_data_file(COPY share/egconfig DESTINATION Generators)
o2_data_file(COPY share/pythia8 DESTINATION Generators)
