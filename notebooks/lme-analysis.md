Use the command
`R -e "rmarkdown::render('lme-analysis.Rmd',output_file='lme-analysis.md')"`
to auto-generate the `.md` file from the `.Rmd` file. To easily find the
AIC results that go into the paper, search for
`#****this line for section 4.` in the `.md` version of the file.

    library(tidyverse)

    ## ── Attaching packages ─────────────────────────────────────── tidyverse 1.3.0 ──

    ## ✓ ggplot2 3.3.3     ✓ purrr   0.3.4
    ## ✓ tibble  3.0.6     ✓ dplyr   1.0.4
    ## ✓ tidyr   1.1.2     ✓ stringr 1.4.0
    ## ✓ readr   1.4.0     ✓ forcats 0.5.1

    ## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ## x dplyr::filter() masks stats::filter()
    ## x dplyr::lag()    masks stats::lag()

    library(tidyr)
    library(lme4)

    ## Loading required package: Matrix

    ## 
    ## Attaching package: 'Matrix'

    ## The following objects are masked from 'package:tidyr':
    ## 
    ##     expand, pack, unpack

    library(brms)

    ## Loading required package: Rcpp

    ## Loading 'brms' package (version 2.14.4). Useful instructions
    ## can be found by typing help('brms'). A more detailed introduction
    ## to the package is available through vignette('brms_overview').

    ## 
    ## Attaching package: 'brms'

    ## The following object is masked from 'package:lme4':
    ## 
    ##     ngrps

    ## The following object is masked from 'package:stats':
    ## 
    ##     ar

    corpus_sizes <- c("xs"=1,"sm"=4.8,"md"=14,"lg"=42)

    dat <- read_csv("../data/suites_df.csv") %>%
      mutate(architecture=model_name,size=gsub("^.*(xs|sm|md|lg).*$","\\1",corpus),performance=correct) 

    ## Warning: Missing column names filled in: 'X1' [1]

    ## 
    ## ── Column specification ────────────────────────────────────────────────────────
    ## cols(
    ##   X1 = col_double(),
    ##   model_name = col_character(),
    ##   corpus = col_character(),
    ##   seed = col_double(),
    ##   suite = col_character(),
    ##   pretty_model_name = col_character(),
    ##   pretty_corpus = col_character(),
    ##   correct = col_double(),
    ##   tag = col_character(),
    ##   circuit = col_character(),
    ##   correct_delta = col_double()
    ## )

    dat.by_suite <- dat %>%
      group_by(size,architecture,seed,suite,circuit) %>%
      summarise(performance=mean(correct)) %>%
      mutate(instance=paste(architecture,size,seed,sep="-")) %>%
      mutate(numwords=corpus_sizes[size])

    ## `summarise()` has grouped output by 'size', 'architecture', 'seed', 'suite'. You can override using the `.groups` argument.

    ## Ideally we would impose inequality constraints on the "size" effects, but I don't think we can do this with lme4.
    ## So, the analysis below uses number of words in the corpus (either raw or log) as a numeric predictor.
    ## note that the bobyqa optimizer helps us get model convergence.

    #m.lin <- lmer(performance ~ numwords + architecture + (numwords + architecture | suite) + (1|instance),data=dat.by_suite,REML=F,control=lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=200000) ))
    m.log <- lmer(performance ~ log(numwords) + architecture + (log(numwords) + architecture | suite) + (1|instance),data=dat.by_suite,REML=F,control=lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=200000) ))
    #m0.lin <- lmer(performance ~ architecture + (numwords + architecture | suite) + (1|instance),data=dat.by_suite,REML=F,control=lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=200000) ))
    m0.log <- lmer(performance ~ architecture + (log(numwords) + architecture | suite) + (1|instance),data=dat.by_suite,REML=F,control=lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=200000) ))
    #m0.arch.lin <- lmer(performance ~ numwords + (numwords + architecture | suite) + (1|instance),data=dat.by_suite,REML=F,control=lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=200000) ))
    m0.arch.log <- lmer(performance ~ log(numwords) + (log(numwords) + architecture | suite) + (1|instance),data=dat.by_suite,REML=F,control=lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=200000) ))

    #summary(m.lin)
    summary(m.log)

    ## Linear mixed model fit by maximum likelihood  ['lmerMod']
    ## Formula: performance ~ log(numwords) + architecture + (log(numwords) +  
    ##     architecture | suite) + (1 | instance)
    ##    Data: dat.by_suite
    ## Control: lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e+05))
    ## 
    ##      AIC      BIC   logLik deviance df.resid 
    ##   -752.6   -602.3    405.3   -810.6     1291 
    ## 
    ## Scaled residuals: 
    ##     Min      1Q  Median      3Q     Max 
    ## -4.5129 -0.4203 -0.0628  0.4160  4.8301 
    ## 
    ## Random effects:
    ##  Groups   Name                        Variance  Std.Dev. Corr             
    ##  instance (Intercept)                 0.0004507 0.02123                   
    ##  suite    (Intercept)                 0.0497953 0.22315                   
    ##           log(numwords)               0.0004846 0.02201  -0.17            
    ##           architecturengram           0.0598971 0.24474  -0.37 -0.36      
    ##           architectureordered-neurons 0.0481272 0.21938   0.01  0.72 -0.38
    ##           architecturernng            0.0523651 0.22883  -0.17  0.72 -0.10
    ##           architecturevanilla         0.0090947 0.09537   0.27 -0.08  0.02
    ##  Residual                             0.0233421 0.15278                   
    ##             
    ##             
    ##             
    ##             
    ##             
    ##             
    ##   0.84      
    ##   0.38  0.45
    ##             
    ## Number of obs: 1320, groups:  instance, 40; suite, 33
    ## 
    ## Fixed effects:
    ##                              Estimate Std. Error t value
    ## (Intercept)                  0.238438   0.043092   5.533
    ## log(numwords)                0.023046   0.005431   4.243
    ## architecturengram           -0.065352   0.048929  -1.336
    ## architectureordered-neurons  0.218265   0.043356   5.034
    ## architecturernng             0.261404   0.044522   5.871
    ## architecturevanilla          0.074366   0.025722   2.891
    ## 
    ## Correlation of Fixed Effects:
    ##             (Intr) lg(nm) archtctrn archt- archtctrr
    ## log(nmwrds) -0.233                                  
    ## archtctrngr -0.429 -0.224                           
    ## archtctrrd- -0.153  0.476 -0.153                    
    ## archtctrrnn -0.295  0.469  0.058     0.811          
    ## archtctrvnl -0.103 -0.036  0.240     0.473  0.513

    #anova(m0.lin,m.lin)
    anova(m0.log,m.log) # ****this line for section 4.2

    ## Data: dat.by_suite
    ## Models:
    ## m0.log: performance ~ architecture + (log(numwords) + architecture | 
    ## m0.log:     suite) + (1 | instance)
    ## m.log: performance ~ log(numwords) + architecture + (log(numwords) + 
    ## m.log:     architecture | suite) + (1 | instance)
    ##        npar     AIC     BIC logLik deviance  Chisq Df Pr(>Chisq)    
    ## m0.log   28 -739.77 -594.58 397.88  -795.77                         
    ## m.log    29 -752.64 -602.26 405.32  -810.64 14.872  1  0.0001151 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    #anova(m0.arch.lin,m.lin)
    anova(m0.arch.log,m.log) #****this line for section 4.2

    ## Data: dat.by_suite
    ## Models:
    ## m0.arch.log: performance ~ log(numwords) + (log(numwords) + architecture | 
    ## m0.arch.log:     suite) + (1 | instance)
    ## m.log: performance ~ log(numwords) + architecture + (log(numwords) + 
    ## m.log:     architecture | suite) + (1 | instance)
    ##             npar     AIC     BIC logLik deviance  Chisq Df Pr(>Chisq)    
    ## m0.arch.log   25 -735.09 -605.45 392.54  -785.09                         
    ## m.log         29 -752.64 -602.26 405.32  -810.64 25.553  4  3.893e-05 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    m.brm.log <- brm(performance ~ log(numwords) + architecture + (log(numwords) + architecture | suite) + (1|instance),data=dat.by_suite,iter=5000,warmup=1000)
    summary(m.brm.log)
    dat.by_suite <- mutate(dat.by_suite,w=numwords/100) # to help make the effect size easily visible in brms' summary()
    m.brm.lin <- brm(performance ~ w + architecture + (w + architecture | suite) + (1|instance),data=dat.by_suite,iter=5000,warmup=1000)
    summary(m.brm.lin)

    m <- lmer(performance ~ log(numwords)*circuit + architecture*circuit + (log(numwords) + architecture | suite) + (1|instance),data=dat.by_suite,REML=F,control=lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=200000) ))

    ## boundary (singular) fit: see ?isSingular

    m0.size_circuit <- lmer(performance ~ log(numwords) + architecture*circuit + (log(numwords) + architecture | suite) + (1|instance),data=dat.by_suite,REML=F,control=lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=200000) ))

    ## boundary (singular) fit: see ?isSingular

    m0.architecture_circuit <- lmer(performance ~ log(numwords)*circuit + architecture + (log(numwords) + architecture | suite) + (1|instance),data=dat.by_suite,REML=F,control=lmerControl(optimizer="bobyqa",optCtrl=list(maxfun=200000) ))

    ## boundary (singular) fit: see ?isSingular

    summary(m)

    ## Linear mixed model fit by maximum likelihood  ['lmerMod']
    ## Formula: performance ~ log(numwords) * circuit + architecture * circuit +  
    ##     (log(numwords) + architecture | suite) + (1 | instance)
    ##    Data: dat.by_suite
    ## Control: lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e+05))
    ## 
    ##      AIC      BIC   logLik deviance df.resid 
    ##   -858.9   -552.9    488.4   -976.9     1261 
    ## 
    ## Scaled residuals: 
    ##     Min      1Q  Median      3Q     Max 
    ## -4.6380 -0.4234 -0.0604  0.4277  4.8559 
    ## 
    ## Random effects:
    ##  Groups   Name                        Variance  Std.Dev. Corr             
    ##  instance (Intercept)                 0.0004304 0.020747                  
    ##  suite    (Intercept)                 0.0172046 0.131166                  
    ##           log(numwords)               0.0000890 0.009434 -0.69            
    ##           architecturengram           0.0174797 0.132211  0.04 -0.37      
    ##           architectureordered-neurons 0.0141149 0.118806 -0.31  0.87 -0.63
    ##           architecturernng            0.0123181 0.110987 -0.41  0.52 -0.38
    ##           architecturevanilla         0.0044081 0.066394  0.11  0.14 -0.75
    ##  Residual                             0.0229785 0.151587                  
    ##             
    ##             
    ##             
    ##             
    ##             
    ##             
    ##   0.64      
    ##   0.51  0.74
    ##             
    ## Number of obs: 1320, groups:  instance, 40; suite, 33
    ## 
    ## Fixed effects:
    ##                                                               Estimate
    ## (Intercept)                                                   -0.11080
    ## log(numwords)                                                  0.07565
    ## circuitCenter Embedding                                        0.70352
    ## circuitGarden-Path Effects                                     0.35359
    ## circuitGross Syntactic State                                   0.41751
    ## circuitLicensing                                               0.20515
    ## circuitLong-Distance Dependencies                              0.53433
    ## architecturengram                                             -0.03070
    ## architectureordered-neurons                                    0.25082
    ## architecturernng                                               0.45108
    ## architecturevanilla                                           -0.02193
    ## log(numwords):circuitCenter Embedding                         -0.05940
    ## log(numwords):circuitGarden-Path Effects                      -0.04718
    ## log(numwords):circuitGross Syntactic State                    -0.03854
    ## log(numwords):circuitLicensing                                -0.07976
    ## log(numwords):circuitLong-Distance Dependencies               -0.04779
    ## circuitCenter Embedding:architecturengram                     -0.21037
    ## circuitGarden-Path Effects:architecturengram                   0.24871
    ## circuitGross Syntactic State:architecturengram                -0.34973
    ## circuitLicensing:architecturengram                             0.10110
    ## circuitLong-Distance Dependencies:architecturengram           -0.22838
    ## circuitCenter Embedding:architectureordered-neurons           -0.26911
    ## circuitGarden-Path Effects:architectureordered-neurons         0.13152
    ## circuitGross Syntactic State:architectureordered-neurons       0.31946
    ## circuitLicensing:architectureordered-neurons                  -0.22428
    ## circuitLong-Distance Dependencies:architectureordered-neurons -0.04501
    ## circuitCenter Embedding:architecturernng                      -0.33741
    ## circuitGarden-Path Effects:architecturernng                    0.04698
    ## circuitGross Syntactic State:architecturernng                  0.10977
    ## circuitLicensing:architecturernng                             -0.38283
    ## circuitLong-Distance Dependencies:architecturernng            -0.30965
    ## circuitCenter Embedding:architecturevanilla                    0.05467
    ## circuitGarden-Path Effects:architecturevanilla                 0.24275
    ## circuitGross Syntactic State:architecturevanilla               0.04276
    ## circuitLicensing:architecturevanilla                           0.05987
    ## circuitLong-Distance Dependencies:architecturevanilla          0.10528
    ##                                                               Std. Error
    ## (Intercept)                                                      0.09036
    ## log(numwords)                                                    0.01154
    ## circuitCenter Embedding                                          0.14174
    ## circuitGarden-Path Effects                                       0.10979
    ## circuitGross Syntactic State                                     0.11859
    ## circuitLicensing                                                 0.10221
    ## circuitLong-Distance Dependencies                                0.10512
    ## architecturengram                                                0.09936
    ## architectureordered-neurons                                      0.08746
    ## architecturernng                                                 0.08287
    ## architecturevanilla                                              0.06454
    ## log(numwords):circuitCenter Embedding                            0.01786
    ## log(numwords):circuitGarden-Path Effects                         0.01384
    ## log(numwords):circuitGross Syntactic State                       0.01495
    ## log(numwords):circuitLicensing                                   0.01288
    ## log(numwords):circuitLong-Distance Dependencies                  0.01325
    ## circuitCenter Embedding:architecturengram                        0.15537
    ## circuitGarden-Path Effects:architecturengram                     0.12035
    ## circuitGross Syntactic State:architecturengram                   0.12999
    ## circuitLicensing:architecturengram                               0.11204
    ## circuitLong-Distance Dependencies:architecturengram              0.11523
    ## circuitCenter Embedding:architectureordered-neurons              0.13686
    ## circuitGarden-Path Effects:architectureordered-neurons           0.10601
    ## circuitGross Syntactic State:architectureordered-neurons         0.11450
    ## circuitLicensing:architectureordered-neurons                     0.09869
    ## circuitLong-Distance Dependencies:architectureordered-neurons    0.10149
    ## circuitCenter Embedding:architecturernng                         0.12963
    ## circuitGarden-Path Effects:architecturernng                      0.10041
    ## circuitGross Syntactic State:architecturernng                    0.10845
    ## circuitLicensing:architecturernng                                0.09348
    ## circuitLong-Distance Dependencies:architecturernng               0.09613
    ## circuitCenter Embedding:architecturevanilla                      0.10028
    ## circuitGarden-Path Effects:architecturevanilla                   0.07768
    ## circuitGross Syntactic State:architecturevanilla                 0.08390
    ## circuitLicensing:architecturevanilla                             0.07231
    ## circuitLong-Distance Dependencies:architecturevanilla            0.07437
    ##                                                               t value
    ## (Intercept)                                                    -1.226
    ## log(numwords)                                                   6.556
    ## circuitCenter Embedding                                         4.963
    ## circuitGarden-Path Effects                                      3.221
    ## circuitGross Syntactic State                                    3.521
    ## circuitLicensing                                                2.007
    ## circuitLong-Distance Dependencies                               5.083
    ## architecturengram                                              -0.309
    ## architectureordered-neurons                                     2.868
    ## architecturernng                                                5.443
    ## architecturevanilla                                            -0.340
    ## log(numwords):circuitCenter Embedding                          -3.325
    ## log(numwords):circuitGarden-Path Effects                       -3.409
    ## log(numwords):circuitGross Syntactic State                     -2.578
    ## log(numwords):circuitLicensing                                 -6.192
    ## log(numwords):circuitLong-Distance Dependencies                -3.607
    ## circuitCenter Embedding:architecturengram                      -1.354
    ## circuitGarden-Path Effects:architecturengram                    2.067
    ## circuitGross Syntactic State:architecturengram                 -2.690
    ## circuitLicensing:architecturengram                              0.902
    ## circuitLong-Distance Dependencies:architecturengram            -1.982
    ## circuitCenter Embedding:architectureordered-neurons            -1.966
    ## circuitGarden-Path Effects:architectureordered-neurons          1.241
    ## circuitGross Syntactic State:architectureordered-neurons        2.790
    ## circuitLicensing:architectureordered-neurons                   -2.273
    ## circuitLong-Distance Dependencies:architectureordered-neurons  -0.444
    ## circuitCenter Embedding:architecturernng                       -2.603
    ## circuitGarden-Path Effects:architecturernng                     0.468
    ## circuitGross Syntactic State:architecturernng                   1.012
    ## circuitLicensing:architecturernng                              -4.096
    ## circuitLong-Distance Dependencies:architecturernng             -3.221
    ## circuitCenter Embedding:architecturevanilla                     0.545
    ## circuitGarden-Path Effects:architecturevanilla                  3.125
    ## circuitGross Syntactic State:architecturevanilla                0.510
    ## circuitLicensing:architecturevanilla                            0.828
    ## circuitLong-Distance Dependencies:architecturevanilla           1.416

    ## 
    ## Correlation matrix not shown by default, as p = 36 > 12.
    ## Use print(x, correlation=TRUE)  or
    ##     vcov(x)        if you need it

    ## optimizer (bobyqa) convergence code: 0 (OK)
    ## boundary (singular) fit: see ?isSingular

    #summary(m0.size_circuit)
    anova(m0.size_circuit,m) #****this line for section 4.3

    ## Data: dat.by_suite
    ## Models:
    ## m0.size_circuit: performance ~ log(numwords) + architecture * circuit + (log(numwords) + 
    ## m0.size_circuit:     architecture | suite) + (1 | instance)
    ## m: performance ~ log(numwords) * circuit + architecture * circuit + 
    ## m:     (log(numwords) + architecture | suite) + (1 | instance)
    ##                 npar     AIC     BIC logLik deviance  Chisq Df Pr(>Chisq)    
    ## m0.size_circuit   54 -837.86 -557.85 472.93  -945.86                         
    ## m                 59 -858.87 -552.93 488.43  -976.87 31.009  5  9.329e-06 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    anova(m0.architecture_circuit,m) #****this line for section 4.3

    ## Data: dat.by_suite
    ## Models:
    ## m0.architecture_circuit: performance ~ log(numwords) * circuit + architecture + (log(numwords) + 
    ## m0.architecture_circuit:     architecture | suite) + (1 | instance)
    ## m: performance ~ log(numwords) * circuit + architecture * circuit + 
    ## m:     (log(numwords) + architecture | suite) + (1 | instance)
    ##                         npar     AIC     BIC logLik deviance  Chisq Df
    ## m0.architecture_circuit   39 -795.56 -593.33 436.78  -873.56          
    ## m                         59 -858.87 -552.93 488.43  -976.87 103.31 20
    ##                         Pr(>Chisq)    
    ## m0.architecture_circuit               
    ## m                        3.212e-13 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
