<!-- doxy
\page refrunSimExamplesAliRoot_AMPT Example AliRoot_AMPT
/doxy -->

This is a complex simulation example showing how to run event simulation using the AMPT event generator interface from AliRoot.
A wrapper class AliRoot_AMPT is defined to keep the AliGenAmpt instance and configure it.
It also provides methods to set a random event plane before event generation and to update the event header.
The overall setup is steered by the function `ampt(double energy = 5020., double bMin = 0., double bMax = 20.)` defined in the macro `aliroot_ampt.macro`.

The macro file is specified via `--configKeyValues` setting `GeneratorExternal.fileName` whereas the specific function call to retrieven the configuration is specified via `GeneratorExternal.funcName`.
 
# IMPORTANT
To run this example you need to load an AliRoot package compatible with the O2.
for more details, https://alice.its.cern.ch/jira/browse/AOGM-246

AliRoot needs to be loaded **after** O2 in the following sense:
`alienv enter O2/latest,AliRoot/latest`
The other order may show unresolved symbol problems.

# WARNING
The physics output of this simulation is not fully tested and validated.
