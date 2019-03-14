\page refUtilities Package 'Utilities'

Package 'Utilities'
===================

<!--  \cond EXCLUDE_FOR_DOXYGEN -->
Below is how the subpages can be linked to be accessed on github.
These links are however not compatible with Doxygen.

A suggestion that compatible links could be achieved by using topic names which would not need the path to a file (see below) did not work for me. The links appear as links in documentation generated
with Doxygen, but are broken links.
Eg.

    DataCompression/README.md -> DataCompression/DataCompressionREADME.md
    could be linked as [DataCompressionREADME].

- [DataCompression](DataCompression/README.md)
- [MCStepLogger](MCStepLogger/README.md)
- [O2Device](O2Device/README.md)
- [O2MessageMonitor](O2MessageMonitor/README.md)
- [PCG](PCG/README.md)
- [Publishers](Publishers/README.md)
- [Tools](Tools/README.md)
- [aliceHLTwrapper](aliceHLTwrapper/README.md)
- [hough](hough/README.md)
<!--  \endcond  -->

Below is how the subpages are linked to be accessed in Doxygen documentation. We would need to hide this text from Markdown but keep it processing by Doxygen.

However as doxygen pre-processes all comments according to the Markdown format first and then the output of markdown processing is further processed by doxygen, once we filter these out from Markdown, they do not get to Doxygen.

- \subpage refUtilitiesDataCompression
- \subpage refUtilitiesDataFlow
- \subpage refUtilitiesMCStepLogger
- \subpage refUtilitiesO2Device
- \subpage refUtilitiesO2MessageMonitor
- \subpage refUtilitiesPCG
- \subpage refUtilitiesPublishers
- \subpage refUtilitiesTools
- \subpage refUtilitiesaliceHLTwrapper
- \subpage refUtilitieshough
