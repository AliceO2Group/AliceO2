# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

if(pythia_FOUND)
  set(pythiaTarget pythia FairRoot::Gen)
endif()

o2_add_library(Generators
               SOURCES src/Generator.cxx
                       src/GeneratorTGenerator.cxx
                       src/GeneratorFromFile.cxx
                       src/Pythia6Generator.cxx
                       src/PDG.cxx
                       src/PrimaryGenerator.cxx
                       src/InteractionDiamondParam.cxx
                       src/BoxGunParam.cxx
                       $<$<BOOL:${pythia_FOUND}>:src/Pythia8Generator.cxx>
                       $<$<BOOL:${pythia_FOUND}>:src/GeneratorFactory.cxx>
               PUBLIC_LINK_LIBRARIES FairRoot::Base O2::SimConfig
                                     O2::SimulationDataFormat ${pythiaTarget}
               TARGETVARNAME targetName)

if(pythia_FOUND)
  target_compile_definitions(${targetName} PUBLIC GENERATORS_WITH_PYTHIA8)
endif()

set(headers
    include/Generators/Generator.h
    include/Generators/GeneratorTGenerator.h
    include/Generators/GeneratorFromFile.h
    include/Generators/Pythia6Generator.h
    include/Generators/PDG.h
    include/Generators/PrimaryGenerator.h
    include/Generators/InteractionDiamondParam.h
    include/Generators/BoxGunParam.h)

if(pythia_FOUND)
  list(APPEND headers include/Generators/Pythia8Generator.h
              include/Generators/GeneratorFactory.h)
endif()

o2_target_root_dictionary(Generators HEADERS ${headers})

o2_add_test_root_macro(share/external/extgen.C
                       PUBLIC_LINK_LIBRARIES O2::Generators FairRoot::Base
                       LABELS generators)
if(pythia6_FOUND)
  o2_add_test_root_macro(share/external/pythia6.C
                         PUBLIC_LINK_LIBRARIES O2::Generators pythia6
                         LABELS generators)
endif()
o2_add_test_root_macro(share/external/tgenerator.C
                       PUBLIC_LINK_LIBRARIES O2::Generators
                       LABELS generators)

install(FILES share/external/extgen.C share/external/pythia6.C
              share/external/tgenerator.C
        DESTINATION share/Generators/external/)
