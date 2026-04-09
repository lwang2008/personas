-We care about the VALUES, WANTS, and OPINIONS of the personas. 
-Demographics/other data determine the probability of a person having certain values, wants and opinions
-Values/wants/opinions are not mutually exclusive, pro-abortion may increase probability of being pro-LGBTQ rights -> to be implemented
-Upbringing also informs values/wants/opinions, but this is random
-Region/state of resident in the United States may inform values/wants/opinions too -> to be implemented.





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

