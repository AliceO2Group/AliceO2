# CTPRateFetcher

## The ZDC cross-section ratio and the pile-up correction

`CTPRateFetcher::fetch(..., "ZNC hadronic")` gives the PbPb hadronic interaction rate. Two things
happen to the raw ZNC counting rate `R` on the way there: the Poisson inversion in
`pileUpCorrection`, and the division by `sigma_ZNC / sigma_had = 28`. The order matters.

Write `mu` for a mean number of interactions per colliding crossing, `N` for the number of
colliding bunch pairs and `f` for the revolution frequency. Then

    mu_ZNC = 28 * mu_had                 exact: the ZDC sees hadronic and EMD, and means add
    R      = N f (1 - exp(-mu_ZNC))      a crossing is counted once, however many interactions
    mu_ZNC = -ln(1 - R/(N f))            which is what pileUpCorrection computes

and therefore

    hadronic rate = N f mu_had = pileUpCorrection(R) / 28

Invert first, divide second. `pileUpCorrection(R/28)` inverts the saturation of a rate that never
saturated - the saturation is in `R` - and comes out low by

    [-ln(1 - x/28)] / [(-ln(1 - x)) / 28] ,     x = R/(N f)

The two agree to first order in `x` and drift apart with pile-up. Measured on three 2024 PbPb runs
at 10 %, 50 % and 90 % of the run:

    run    N_bc   R_ZNC [Hz]  x=R/(Nf)   mu_ZNC   divide first   divide last   ratio
    544490 1088    1238358.4   0.10121  0.10671        44307.2       46628.8  0.9502
    544490 1088    1158151.9   0.09466  0.09944        41432.6       43453.3  0.9535
    544490 1088    1086013.1   0.08876  0.09295        38847.8       40616.7  0.9564
    559856 1032    1268830.4   0.10933  0.11578        45404.1       47989.4  0.9461
    559856 1032     959082.2   0.08264  0.08626        34303.6       35751.4  0.9595
    559856 1032     756641.1   0.06520  0.06742        27054.4       27944.1  0.9682
    568721 1032     633548.4   0.05459  0.05614        22648.8       23267.8  0.9734
    568721 1032     623888.1   0.05376  0.05526        22303.1       22903.0  0.9738
    568721 1032     635486.5   0.05476  0.05631        22718.2       23341.0  0.9733

Both columns come from an unpatched build: `fetch(..., "ZNC")`, without `"hadronic"`, returns
`pileUpCorrection(R)` already, so the second column is `fetch(..., "ZNC") / 28`.

`getLumi` divides after `pileUpCorrection` and needs no change.
