TIER = "KPI Tier"
PERCENT_RANK = "Percent Rank"
FEATURE = "Feature"
WEIGHT = "Weight"
CPM = "CPM"
SCORE = "Score"
RANK = "Rank"
DMA_CODE = "DMA Code"
DMA_NAME = "DMA Name"
COUNTRY_CODE = "Country Code"
COUNTRY_NAME = "Country Name"
STATE_CODE = "State Code"
STATE_NAME = "State Name"
VARIABLE_CORRELATION_THRESHOLD = 0.5
MAX_DEFAULT = 9

#MARKET_LEVELS = [c.replace('Code','') for c in [COUNTRY_CODE, STATE_CODE, DMA_CODE]]
MARKETS = [DMA_CODE, COUNTRY_CODE, STATE_CODE]
MARKET_LEVELS = ["US DMA", "US State", "BR Municipality", 'MX Municipality', 'Other'] #, "MX City"] # Format: country marketType
MARKET_COLUMN = 'Market'

AUDIENCE_BUILDER_DATASETS = [
    'age', 'education_short', 'employment_status', 'foreign_born', 'household_income',
    'household_language', 'housing_value', 'internet', 'race', 'voter'
]

DEFAULT_DMA_COLS = [
  "Gdp",
  "Median Household Income",
  "In Labor Force", 
  "Broadband Any Source", 
  "Total Bachelors Or Higher",
  "Total Spanish Household",
  "Owner Occupied",
  "Default CPM",
]

DEFAULT_STATE_COLS = [
 'Gdp',
 'Median Household Income',
 'Per Capita Income',
 'Median Housing Value',
 'In Labor Force',
 'Broadband Any Source',
 'Total Bachelors Or Higher',
 'Total Spanish',
 'Owner Occupied',
]

DEFAULT_WORLD_COLS = [
    "Population Density (Per Km2, 2017)",
    "Gdp Per Capita (Current Us$)",
    "Economy: Industry (% Of Gva)",
    "Economy: Services And Other Activity (% Of Gva)",
    "Employment: Agriculture (% Of Employed)",
    "Employment: Industry (% Of Employed)",
    "Employment: Services (% Of Employed)",
    "Labour Force Participation (Female/Male Pop. %)",
]

DEFAULT_BR_COLS = [
    "Average Monthly Income (Heads, All)",
]

DEFAULT_MX_COLS = [
    "Average Household Size",
    "Average Years Of Schooling",
    "Economically Active Population",
    "Housing Units With Electricity",
    "Housing Units With Internet",
    "Total Households",
    "Total Housing Units",
    "Unemployed Population",
]

DEFAULT_COLUMNS = {'US_Dma': DEFAULT_DMA_COLS, 
                   'US_State': DEFAULT_STATE_COLS, 
                   'BR_Municipality' : DEFAULT_BR_COLS, 
                   'MX_Municipality' : DEFAULT_MX_COLS,
                   'Country': DEFAULT_WORLD_COLS}
