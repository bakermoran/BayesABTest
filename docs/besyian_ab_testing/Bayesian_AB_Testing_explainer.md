# Bayesian Statistics

There are plenty of articles out there motivating why we should use Bayesian statistics in product analytics. I'm not going to try to reinvent the wheel here, so I will provide only a brief explaination and link to some articles that I found helpful in my learning. I will also explain why I created this package.

## Motivation

### Why Bayesian Statistics

Bayesian statistics is a completely different way to think about statistics from what you likely learned in high school and college (typically referred to as frequentist statistics). Bayesian statistics allows us to much more easily answer a wider variety of questions that are more relevant for the business world. Although most people are _used_ to hearing a p-value for a statistical test, many don't actually know what it means or how to interpret it beyond the "satistical significance is when p <= 0.05" we were programmed to regurgitate. Bayesian statistics allows a much more intuitive interpretation of the results of a test. Examples of questions Bayesian statistics is purpose built to answer:

* What is the probability that Variant we are testing is better than the Control?
* How much better is the Cariant than the Control?
* If we make the wrong choice, how much worse could it actually be?

### The Issue with Traditional AB Testing

The goal of using frequentist statistics is to minimize the probability of being wrong when we pick the variant over the control. P-values are designed to be biased towards the control. In business we often run an experiment because _we believe we are making an improvement to the product_. There clearly needs to be statistical rigor, but a question I often get when the variant is _slightly better_ than the control, is "why can't we just pick the variant?". Bayesian statistics allows for statistical support, even when picking a variant that is only slightly better.

Frequentist statistics protects us against choosing something new that isn't actually better. This is important in things like medicine; it's not that important in the business world. In business, we want to run lots of tests as quickly as we can, in order to make the best decisions we can about the business. Changing the color of a button on the website likely will not result in lives lost, while a new medication could. Bayesian statistics allows us to control the risk we are taking on every decision we make; we can choose to make a decision with less data than we would need with Frequentist methods to approve a new medication.

### The Power of Bayes Theorem

Bayesian statistics is designed to use our belief about the world in order to help us make a decision. To me at first, this was confusing because it sounded like an arbitrary choice. It is; but the key to understanding why Bayesian statistics is so powerful is that we are making even strong assumptions when using frequentist statistics. In business and tech, we often have access to a lot of data and have a pretty good idea about conversion rates. I would argue that _not_ using any of that information is a more egregious mistake than using the wrong prior with Bayesian statistics.

## Goal of this package

I created this package originally out of necessity and a desire to learn.

* I previously had little exposure to Bayesian statistics, and I found myself wanting to learn about it. The best way for me to learn things is to teach other people and build something.
* The code we used at Root for testing only included functionality for one variant conversion data AB tests. I often needed the ability to use different priors for continuous data, as well as additional variants.
* Additionally, I would much rather use Python than R. Initially, I built this to only add that functionality to my daily work. Then I realized, there are several very robust and really good R packages for Bayesian AB testing (I'll link a few). But I couldn't find anything that fit my needs in Python. For this reason, I decided to expand and productionalize this package into something that analysts at other data driven and testing oriented companies can use.
* This package is meant to be an easy to use method for analysts to report AB test results in a Bayesian framework using Python.

### Further Reading

* <https://www.countbayesie.com/blog/2015/4/25/bayesian-ab-testing> (this entire blog is great for the budding Bayesian statistitian)
* <https://medium.com/convoy-tech/the-power-of-bayesian-a-b-testing-f859d2219d5> (how bayesian statistics allows us to innovate quickly)
* <https://cdn2.hubspot.net/hubfs/310840/VWO_SmartStats_technical_whitepaper.pdf> (a technical paper on defining risk with bayesian statistics)
* <https://sl8r000.github.io/ab_testing_statistics/> (how to use bayesian statistics correctly when AB testing)
* <https://medium.com/airbnb-engineering/experiments-at-airbnb-e2db3abf39e7#.miqyczkzb> (AB testing at Airbnb)

### R packages for Bayesian AB testing

* <https://github.com/FrankPortman/bayesAB> - bayesAB by Frank Portman
* <https://github.com/convoyinc/abayes> - abayes by Convoy Inc.
