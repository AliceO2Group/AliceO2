o2_add_library(MemoryResources SOURCES src/MemoryResources.cxx
               PUBLIC_LINK_LIBRARIES FairMQ::FairMQ)

o2_add_test(MemoryResources
            SOURCES
            test/testMemoryResources.cxx
            PUBLIC_LINK_LIBRARIES
            O2::MemoryResources
            COMPONENT_NAME
            MemoryResources)

o2_add_test(observer_ptr
            SOURCES
            test/test_observer_ptr.cxx
            PUBLIC_LINK_LIBRARIES
            O2::MemoryResources
            COMPONENT_NAME
            MemoryResources)
