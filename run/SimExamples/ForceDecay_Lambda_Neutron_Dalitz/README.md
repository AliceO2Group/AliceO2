<!-- doxy
\page refrunSimExamplesForceDecay_Lambda_Neutron_Dalitz Example ForceDecay_Lambda_Neutron_Dalitz
/doxy -->

This is a simple simulation example showing how to force a specific particle decay chain.
To simplify the example, we will use a Box generator to inject Lambda0 particles in the simulation.

What we want to achieve is a forced decay chain of the Lambda0
```
Lambda0 --> n pi0 --> n e+ e- gamma
```
Given that both the Lambda0 and the pi0 are by default decayed by Geant, we have to turn on the external decayer for these particles.
This can be done with
```
--configKeyValues 'SimUserDecay.pdg=3122 111'
```

On top of that we have to setup the external decayer to perform the decay forcing the channels we desire.
The default external decayer configuration is loaded from
```
${O2_ROOT}/share/Generators/pythia8/decays/base.cfg
which is assigned to the slot #0 of the configuration parameter 'DecayerPythia8.config'.
```
What we want to do is to add on top of the default configuration some extra settings that are provided in the `decay_lambda_neutron_dalitz.cfg` file.
This can be done with
```
--configKeyValues 'DecayerPythia8.config[1]=decay_lambda_neutron_dalitz.cfg'
```

