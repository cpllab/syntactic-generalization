Use the command
`R -e "rmarkdown::render('lme-analysis.Rmd',output_file='lme-analysis.md')"`
to auto-generate the `.md` file from the `.Rmd` file. To easily find the
AIC results that go into the paper, search for
`#****this line for section 4.` in the `.md` version of the file.

    library(tidyverse)

    ## Registered S3 methods overwritten by 'ggplot2':
    ##   method         from 
    ##   [.quosures     rlang
    ##   c.quosures     rlang
    ##   print.quosures rlang

    ## Registered S3 method overwritten by 'rvest':
    ##   method            from
    ##   read_xml.response xml2

    ## ── Attaching packages ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── tidyverse 1.2.1 ──

    ## ✔ ggplot2 3.1.1       ✔ purrr   0.3.2  
    ## ✔ tibble  2.1.1       ✔ dplyr   0.8.0.1
    ## ✔ tidyr   0.8.3       ✔ stringr 1.4.0  
    ## ✔ readr   1.3.1       ✔ forcats 0.4.0

    ## ── Conflicts ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── tidyverse_conflicts() ──
    ## ✖ dplyr::filter() masks stats::filter()
    ## ✖ dplyr::lag()    masks stats::lag()

    library(tidyr)
    library(lme4)

    ## Loading required package: Matrix

    ## 
    ## Attaching package: 'Matrix'

    ## The following object is masked from 'package:tidyr':
    ## 
    ##     expand

    library(brms)

    ## Loading required package: Rcpp

    ## Registered S3 method overwritten by 'xts':
    ##   method     from
    ##   as.zoo.xts zoo

    ## Loading 'brms' package (version 2.12.0). Useful instructions
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

    ## Parsed with column specification:
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
    ##   -589.9   -440.6    324.0   -647.9     1242 
    ## 
    ## Scaled residuals: 
    ##     Min      1Q  Median      3Q     Max 
    ## -4.4689 -0.4527 -0.0280  0.4422  4.7727 
    ## 
    ## Random effects:
    ##  Groups   Name                        Variance  Std.Dev. Corr             
    ##  instance (Intercept)                 0.0065703 0.08106                   
    ##  suite    (Intercept)                 0.0344967 0.18573                   
    ##           log(numwords)               0.0005966 0.02443  -0.12            
    ##           architecturengram           0.0556973 0.23600  -0.21 -0.48      
    ##           architectureordered-neurons 0.0646542 0.25427   0.08  0.35 -0.23
    ##           architecturernng            0.0545865 0.23364   0.03  0.51 -0.16
    ##           architecturevanilla         0.0301742 0.17371   0.23 -0.42  0.22
    ##  Residual                             0.0244510 0.15637                   
    ##             
    ##             
    ##             
    ##             
    ##             
    ##             
    ##   0.87      
    ##   0.58  0.49
    ##             
    ## Number of obs: 1271, groups:  instance, 41; suite, 31
    ## 
    ## Fixed effects:
    ##                              Estimate Std. Error t value
    ## (Intercept)                  0.474348   0.054815   8.654
    ## log(numwords)                0.037631   0.010630   3.540
    ## architecturengram           -0.316112   0.071483  -4.422
    ## architectureordered-neurons -0.051501   0.066393  -0.776
    ## architecturernng            -0.001798   0.062546  -0.029
    ## architecturevanilla         -0.195286   0.055317  -3.530
    ## 
    ## Correlation of Fixed Effects:
    ##             (Intr) lg(nm) archtctrn archt- archtctrr
    ## log(nmwrds) -0.370                                  
    ## archtctrngr -0.457 -0.103                           
    ## archtctrrd- -0.402  0.179  0.219                    
    ## archtctrrnn -0.436  0.189  0.266     0.761          
    ## archtctrvnl -0.415 -0.077  0.447     0.626  0.612

    #anova(m0.lin,m.lin)
    anova(m0.log,m.log) # ****this line for section 4.2

    ## Data: dat.by_suite
    ## Models:
    ## m0.log: performance ~ architecture + (log(numwords) + architecture | 
    ## m0.log:     suite) + (1 | instance)
    ## m.log: performance ~ log(numwords) + architecture + (log(numwords) + 
    ## m.log:     architecture | suite) + (1 | instance)
    ##        Df     AIC     BIC logLik deviance  Chisq Chi Df Pr(>Chisq)    
    ## m0.log 28 -580.71 -436.57 318.35  -636.71                             
    ## m.log  29 -589.90 -440.62 323.95  -647.90 11.198      1  0.0008187 ***
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
    ##             Df     AIC     BIC logLik deviance  Chisq Chi Df Pr(>Chisq)
    ## m0.arch.log 25 -573.59 -444.90 311.79  -623.59                         
    ## m.log       29 -589.90 -440.62 323.95  -647.90 24.317      4  6.901e-05
    ##                
    ## m0.arch.log    
    ## m.log       ***
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
    ##   -673.7   -370.0    395.9   -791.7     1212 
    ## 
    ## Scaled residuals: 
    ##     Min      1Q  Median      3Q     Max 
    ## -4.6035 -0.4502 -0.0200  0.4589  4.4993 
    ## 
    ## Random effects:
    ##  Groups   Name                        Variance Std.Dev. Corr             
    ##  instance (Intercept)                 0.006432 0.08020                   
    ##  suite    (Intercept)                 0.019469 0.13953                   
    ##           log(numwords)               0.000118 0.01086  -0.26            
    ##           architecturengram           0.023279 0.15257  -0.20 -0.53      
    ##           architectureordered-neurons 0.026871 0.16392  -0.48  0.13 -0.02
    ##           architecturernng            0.017655 0.13287  -0.50  0.01 -0.02
    ##           architecturevanilla         0.015470 0.12438  -0.31 -0.39  0.04
    ##  Residual                             0.024078 0.15517                   
    ##             
    ##             
    ##             
    ##             
    ##             
    ##             
    ##   0.79      
    ##   0.80  0.82
    ##             
    ## Number of obs: 1271, groups:  instance, 41; suite, 31
    ## 
    ## Fixed effects:
    ##                                                                Estimate
    ## (Intercept)                                                    0.254130
    ## log(numwords)                                                  0.098744
    ## circuitCenter Embedding                                        0.459898
    ## circuitGarden-Path Effects                                     0.242841
    ## circuitGross Syntactic State                                   0.309802
    ## circuitLicensing                                               0.113404
    ## circuitLong-Distance Dependencies                              0.346113
    ## architecturengram                                             -0.441496
    ## architectureordered-neurons                                   -0.149316
    ## architecturernng                                               0.044952
    ## architecturevanilla                                           -0.432724
    ## log(numwords):circuitCenter Embedding                         -0.073315
    ## log(numwords):circuitGarden-Path Effects                      -0.054291
    ## log(numwords):circuitGross Syntactic State                    -0.044735
    ## log(numwords):circuitLicensing                                -0.085396
    ## log(numwords):circuitLong-Distance Dependencies               -0.064872
    ## circuitCenter Embedding:architecturengram                      0.060887
    ## circuitGarden-Path Effects:architecturengram                   0.373588
    ## circuitGross Syntactic State:architecturengram                -0.229715
    ## circuitLicensing:architecturengram                             0.204028
    ## circuitLong-Distance Dependencies:architecturengram            0.067028
    ## circuitCenter Embedding:architectureordered-neurons           -0.004274
    ## circuitGarden-Path Effects:architectureordered-neurons         0.253112
    ## circuitGross Syntactic State:architectureordered-neurons       0.436617
    ## circuitLicensing:architectureordered-neurons                  -0.123948
    ## circuitLong-Distance Dependencies:architectureordered-neurons  0.169191
    ## circuitCenter Embedding:architecturernng                      -0.068965
    ## circuitGarden-Path Effects:architecturernng                    0.170423
    ## circuitGross Syntactic State:architecturernng                  0.228539
    ## circuitLicensing:architecturernng                             -0.281040
    ## circuitLong-Distance Dependencies:architecturernng            -0.072933
    ## circuitCenter Embedding:architecturevanilla                    0.325924
    ## circuitGarden-Path Effects:architecturevanilla                 0.367627
    ## circuitGross Syntactic State:architecturevanilla               0.162781
    ## circuitLicensing:architecturevanilla                           0.162799
    ## circuitLong-Distance Dependencies:architecturevanilla          0.370641
    ##                                                               Std. Error
    ## (Intercept)                                                     0.101028
    ## log(numwords)                                                   0.014950
    ## circuitCenter Embedding                                         0.146228
    ## circuitGarden-Path Effects                                      0.113267
    ## circuitGross Syntactic State                                    0.122343
    ## circuitLicensing                                                0.105446
    ## circuitLong-Distance Dependencies                               0.113267
    ## architecturengram                                               0.119453
    ## architectureordered-neurons                                     0.116274
    ## architecturernng                                                0.100555
    ## architecturevanilla                                             0.096210
    ## log(numwords):circuitCenter Embedding                           0.018812
    ## log(numwords):circuitGarden-Path Effects                        0.014572
    ## log(numwords):circuitGross Syntactic State                      0.015739
    ## log(numwords):circuitLicensing                                  0.013566
    ## log(numwords):circuitLong-Distance Dependencies                 0.014572
    ## circuitCenter Embedding:architecturengram                       0.168621
    ## circuitGarden-Path Effects:architecturengram                    0.130613
    ## circuitGross Syntactic State:architecturengram                  0.141078
    ## circuitLicensing:architecturengram                              0.121594
    ## circuitLong-Distance Dependencies:architecturengram             0.130613
    ## circuitCenter Embedding:architectureordered-neurons             0.169483
    ## circuitGarden-Path Effects:architectureordered-neurons          0.131281
    ## circuitGross Syntactic State:architectureordered-neurons        0.141800
    ## circuitLicensing:architectureordered-neurons                    0.122216
    ## circuitLong-Distance Dependencies:architectureordered-neurons   0.131281
    ## circuitCenter Embedding:architecturernng                        0.143449
    ## circuitGarden-Path Effects:architecturernng                     0.111115
    ## circuitGross Syntactic State:architecturernng                   0.120018
    ## circuitLicensing:architecturernng                               0.103442
    ## circuitLong-Distance Dependencies:architecturernng              0.111115
    ## circuitCenter Embedding:architecturevanilla                     0.136312
    ## circuitGarden-Path Effects:architecturevanilla                  0.105587
    ## circuitGross Syntactic State:architecturevanilla                0.114047
    ## circuitLicensing:architecturevanilla                            0.098296
    ## circuitLong-Distance Dependencies:architecturevanilla           0.105587
    ##                                                               t value
    ## (Intercept)                                                     2.515
    ## log(numwords)                                                   6.605
    ## circuitCenter Embedding                                         3.145
    ## circuitGarden-Path Effects                                      2.144
    ## circuitGross Syntactic State                                    2.532
    ## circuitLicensing                                                1.075
    ## circuitLong-Distance Dependencies                               3.056
    ## architecturengram                                              -3.696
    ## architectureordered-neurons                                    -1.284
    ## architecturernng                                                0.447
    ## architecturevanilla                                            -4.498
    ## log(numwords):circuitCenter Embedding                          -3.897
    ## log(numwords):circuitGarden-Path Effects                       -3.726
    ## log(numwords):circuitGross Syntactic State                     -2.842
    ## log(numwords):circuitLicensing                                 -6.295
    ## log(numwords):circuitLong-Distance Dependencies                -4.452
    ## circuitCenter Embedding:architecturengram                       0.361
    ## circuitGarden-Path Effects:architecturengram                    2.860
    ## circuitGross Syntactic State:architecturengram                 -1.628
    ## circuitLicensing:architecturengram                              1.678
    ## circuitLong-Distance Dependencies:architecturengram             0.513
    ## circuitCenter Embedding:architectureordered-neurons            -0.025
    ## circuitGarden-Path Effects:architectureordered-neurons          1.928
    ## circuitGross Syntactic State:architectureordered-neurons        3.079
    ## circuitLicensing:architectureordered-neurons                   -1.014
    ## circuitLong-Distance Dependencies:architectureordered-neurons   1.289
    ## circuitCenter Embedding:architecturernng                       -0.481
    ## circuitGarden-Path Effects:architecturernng                     1.534
    ## circuitGross Syntactic State:architecturernng                   1.904
    ## circuitLicensing:architecturernng                              -2.717
    ## circuitLong-Distance Dependencies:architecturernng             -0.656
    ## circuitCenter Embedding:architecturevanilla                     2.391
    ## circuitGarden-Path Effects:architecturevanilla                  3.482
    ## circuitGross Syntactic State:architecturevanilla                1.427
    ## circuitLicensing:architecturevanilla                            1.656
    ## circuitLong-Distance Dependencies:architecturevanilla           3.510

    ## 
    ## Correlation matrix not shown by default, as p = 36 > 12.
    ## Use print(x, correlation=TRUE)  or
    ##     vcov(x)        if you need it

    ## convergence code: 0
    ## boundary (singular) fit: see ?isSingular

    #summary(m0.size_circuit)
    anova(m0.size_circuit,m) #****this line for section 4.3

    ## Data: dat.by_suite
    ## Models:
    ## m0.size_circuit: performance ~ log(numwords) + architecture * circuit + (log(numwords) + 
    ## m0.size_circuit:     architecture | suite) + (1 | instance)
    ## m: performance ~ log(numwords) * circuit + architecture * circuit + 
    ## m:     (log(numwords) + architecture | suite) + (1 | instance)
    ##                 Df     AIC     BIC logLik deviance  Chisq Chi Df
    ## m0.size_circuit 54 -654.08 -376.11 381.04  -762.08              
    ## m               59 -673.72 -370.01 395.86  -791.72 29.636      5
    ##                 Pr(>Chisq)    
    ## m0.size_circuit               
    ## m                1.739e-05 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    anova(m0.architecture_circuit,m) #****this line for section 4.3

    ## Data: dat.by_suite
    ## Models:
    ## m0.architecture_circuit: performance ~ log(numwords) * circuit + architecture + (log(numwords) + 
    ## m0.architecture_circuit:     architecture | suite) + (1 | instance)
    ## m: performance ~ log(numwords) * circuit + architecture * circuit + 
    ## m:     (log(numwords) + architecture | suite) + (1 | instance)
    ##                         Df     AIC     BIC logLik deviance  Chisq Chi Df
    ## m0.architecture_circuit 39 -622.70 -421.95 350.35  -700.70              
    ## m                       59 -673.72 -370.01 395.86  -791.72 91.017     20
    ##                         Pr(>Chisq)    
    ## m0.architecture_circuit               
    ## m                         4.92e-11 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
