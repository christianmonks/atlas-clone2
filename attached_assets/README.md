# matched_markets_ui
Repo for the Matched Market Testing product UI, named Atlas.

### Data Folder Structure

The data folder contains census information from the countries US, MX, and BR. It is divided into two subfolders: audience, which contains audience information, and census, which holds demographic information. The naming convention for the files should follow the format: countrycode_levelofspatialdisaggregation, for example: *br_municipality_audience.csv* and *br_municipality_data.csv*.

Internally, the audience datasets must have the following column names:

- *Population*: for the total number of inhabitants
- *Females*: for the female population
- *Males*: for the male population

For disaggregation, the first letter should be in upper case followed by numbers or intervals, for example: *P18+* or *M20-24*.
