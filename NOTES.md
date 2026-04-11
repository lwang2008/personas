-We care about the VALUES, WANTS, and OPINIONS of the personas. 
-Demographics/other data determine the probability of a person having certain values, wants and opinions
-Values/wants/opinions are not mutually exclusive, pro-abortion may increase probability of being pro-LGBTQ rights -> to be implemented
-Upbringing also informs values/wants/opinions, but this is random
-Region/state of resident in the United States may inform values/wants/opinions too -> to be implemented.

-We are able to generate personas based on 1 or 2 fixed parameters, ie. "18-29", "male", "straight + male", "30-49 + Asian
-Currently, the nonfixed attributes are only based on the fixed attributes. Attributes are not generated one at a time, with a new attribute being built off the previous ones.
-For example, if we are studying the group of 18-29 year olds, the probability of the person being male/female, straight/gay/bi, etc. is only dependent on the fact that they are 18-29.  

TASKS
1. Choose parameters
    -age
    -state/region
    -socieconomic status
    -political alignment
    -race
    -gender

2. Determine how to generate personas
    -RNG function, each digit represents parameter (age, state/region, socioeconomic status, political alignment, race, gender, sexual orientation, big five personality traits, marital status, education level, etc.)
        -Need to weigh each parameter based on previous parameters
        -Generate random 100 people based on popoulation proportions and conditionals
        -Simulates a survey is 100% unbiased, completely random 

    -Using LLM (may result in similar profiles/lack of diversity)

3. Create single select survey questions
    -LLM takes in data for each "persona" then generates response based on parameters

4. Compare to real survey responses
    -Provide metric for accuracy of personas
    -Refine/use different methods
    -Pew Research, ANES, General Social Survey



Create "Value" and "Wants" Function
    -Based on demographics/parameters, how much/probability that a certain persona has certain values or wants
    -Wants: lower student debt, public safety, lower gas prices
    -Values: religious values, abortion rights, racial equality, small government, etc.

First Iteration
//Array containing 5 parameters (matching Pew Research)

1. Age Ranges
    1 -> 18-29
    2 -> 30-49
    3 -> 50-64
    4 -> 65+

2. Gender
    1 -> Men
    2 -> Women

3. Sexual Orientation
    1 -> Straight
    2 -> Gay/Lesbian/Bisexual

4. Race
    1 -> White
    2 -> Hispanic
    3 -> Black
    4 -> Asian

5. Political Alignment
    1 -> Democrat
    2 -> Republican



Problems and Fixes:
1. Republican/Democrat is binary, leading to all personas shifting towards the extreme. Added party strength to add nuance to political leanings of personas.

2. LLMs avoid outputs that violate individual liberties/are harmful, ie. questions regarding racial profiling. Rephrased question inputted to LLM in order to provide accurate reponses based on inputted personas.

3. Many people vote republican but disagree on specific issues. LLM is collapsing the variation in value score. Add translation layer to convert float to natural language (ie. 0 means neutral, 0.93 means strongly support, etc.)

4. RLHF when prompting LLM -> persona fights but does not always win. LLM does not understand context behind certain survey questions as deeply as a human is able to. Provide more context to LLM, asking it to reason from each persona's perspective first. Framing of the question affects LLM's response. Downside -> more credits, much slower. 


