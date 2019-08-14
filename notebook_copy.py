import pandas as pd
import datetime
from dateutil import parser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytz
import requests
import json
import pprint
import os
import dateutil
import time

#####

# historical data
import calendar

def next_weekday(d, weekday):
    days_ahead = weekday - d.weekday()
    if days_ahead < 0: # Target day already happened this week
        days_ahead += 7
    return d + datetime.timedelta(days_ahead)

def bucketed(df, start_on="Sunday"):
    df['day_of_week'] = pd.to_datetime(df['created_at']).dt.day_name()
    
    df['created_at'] = pd.to_datetime(df['created_at']).dt.date
    min_date = df["created_at"].min()
    min_date = next_weekday(min_date, list(calendar.day_name).index(start_on))
    max_date = df["created_at"].max()
    tweet_counts = df.groupby('created_at').agg('count')["text"]
    dates = pd.date_range(min_date, max_date, freq='D')
    counts = pd.DataFrame({ "count": tweet_counts},index=dates).fillna(0)
    counts = counts.resample('7D').sum()
    return counts.drop(counts.tail(1).index) # drop last row in case its a count over less than the full time bucket

########################################

ACCOUNT_BALANCE = 58
N_TWEETS = 0

BEARER_TOKEN="AAAAAAAAAAAAAAAAAAAAAAXT9gAAAAAAoITLBCf%2B2K7BMSqakqcbsHUSLrk%3DLz95o8CkkhjOTthpcyEEg6BdNav0zphRcrEYdeG4GXXV3Qkftk"

## value functions
def expected_value(potential_win, chance_win, potential_loss, chance_loss):
    return (potential_win * chance_win) - (potential_loss * chance_loss)

def allocation(account_balance, expected_value):
    pct_alloc = min(( expected_value * 5 ) / 10, .03)
    alloc = account_balance * pct_alloc
    #risk_coef = 1 - (1 / (proba * 100) )
    #risk_adjusted = alloc * risk_coef
    #return risk_adjusted
    return alloc
def allocation(price_per_share, proba):
    payoff_odds = (1 / price_per_share) - 1

def recommended_shares(account_balance, expected_value, price_per_share):
    return allocation(account_balance, expected_value) / price_per_share;

def to_proba(buckets, categories=None):
    vals = buckets.value_counts()
    # [ (range(0,2), "0-2"), range(3-5), "3-5" ]
    #for c in categories:
    #    rnge = c[0]
    #    id_str = c[1]
    #    for r in range:
            
    s = vals.sum()
    return vals/s

## portfolio management
def shares_bought(c, yes_or_no, positions):
    bought = 0
    if c in positions and yes_or_no in positions[c]:
        for pos in positions[c][yes_or_no]:
            bought += pos[1]
    return bought

def recommendation_buy(contract, yes_or_no, account_balance, expected_value, price_per_share, positions):
    shares = recommended_shares(account_balance, expected_value, price_per_share) - shares_bought(contract, yes_or_no, positions)
    shares = int(round(shares))
    if shares > 0:
        print("BUY {yn} shares for contract {n}: {shares} shares @{price} (EV: {ev}, TOTAL: {t})".format(n=contract,shares=shares, price=price_per_share, ev=expected_value, yn=yes_or_no.upper(), t=shares*price_per_share))

def recommendation_sell(contract, yes_or_no, expected_value, price_per_share, n_shares, bought_at):
    print("SELL {yn} shares for contract {n}_{bought_at}_{n_shares}: ALL shares @{price} (EV: {ev}, TOTAL: {t})".format(n=contract, price=price_per_share, ev=expected_value, yn=yes_or_no.upper(), t=n_shares*price_per_share, bought_at=bought_at, n_shares=n_shares))
    
## market evaluation
def fetch_market_data(market_id):
    url = "https://www.predictit.org/api/marketdata/markets/{id}".format(id=market_id)
    r = requests.get(url=url)
    return r.json()


#######################################

def get_twitter_user_timeline(screen_name, max_id=None, since_id=None):
    url = "https://api.twitter.com/1.1/statuses/user_timeline.json"
    headers = { "Authorization": "Bearer {t}".format(t=BEARER_TOKEN)}
    params = {
        "count": "200",
        "trim_user": "true",
        "screen_name": screen_name
    }
    if max_id: 
        params["max_id"] = max_id
    if since_id:
        params["since_id"] = since_id
        
    r = requests.get(url=url,headers=headers, params=params)
    raw = r.json()
    transformed = json.dumps([ { "id": tweet["id"], "created_at": tweet["created_at"], "text": tweet["text"] } for tweet in raw])
    return pd.read_json(transformed, orient="records")

def get_recent_tweets(screen_name, from_date=None):
    df = get_twitter_user_timeline(screen_name)
    df["created_at"] = df["created_at"].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
    if from_date:
        df = df[df["created_at"] > from_date]
    return df
    
# the twitter api returns different results for the same request...
def _get_twitter_history(screen_name, max_id=None):
    get_next = True
    df = pd.DataFrame(columns=["id","created_at", "text"])
    while get_next:
        tweets = get_twitter_user_timeline(screen_name, max_id)
        print(len(tweets.index))
        if len(tweets.index) > 0:
            df = tweets if df.empty else pd.concat([df, tweets], axis=0)
            last_row = tweets.tail(1).iloc[0]
            max_id = last_row["id"] - 1
        else:
            get_next = False
    return df

def get_twitter_history(screen_name, cache=True):
    fname = "data/tweets/{sn}.csv".format(sn=screen_name)
    max_id = None
    if cache and os.path.isfile(fname):
        df = pd.read_csv(fname)
        max_id = int(df.tail(1).iloc[0]["id"]) -1
    df = _get_twitter_history(screen_name, max_id);
    if not os.path.isdir("data/tweets"):
        os.mkdir("data/tweets")
    if len(df) > 0:
        df.to_csv(fname, mode='a')

def fetch_full_trump_tweet_history(rnge, cache=True):
    fname = "data/tweets/@realDonaldTrump.csv"
    df = None
    for year in rnge:
        url = None
        if year == 2019:
            url = "http://www.trumptwitterarchive.com/data/realdonaldtrump/2019.json"
        else:
            url = "http://d5nxcu7vtzvay.cloudfront.net/data/realdonaldtrump/{y}.json".format(y=str(year))
        _df  = pd.read_json(url)
        if df is None:
            df = _df
        else:
            df = pd.concat([df,_df])
        time.sleep(1)

    if not os.path.isdir("data/tweets"):
        os.mkdir("data/tweets")
    if len(df) > 0:
        df.to_csv(fname, mode='w')

#########################
def plot_tweet_distributions_per_day(source_df):
    df = pd.DataFrame(columns=["proba","day"])
    df.index.name = "n_tweets"
    for x in range(0,7,1):
        weekday = calendar.day_name[x]
        b = bucketed(source_df, start_on=weekday)
        proba = b['count']/b['count'].sum()
        _df = pd.DataFrame({ "proba": proba.values, "day": x }, index=proba.index)
        df = pd.concat([df, _df])
        df["n_tweets"] = df.index

    fig, ax = plt.subplots()
    for key, _grp in df.groupby(['n_tweets']):
        grp = _grp.sort_values(by="day", ascending=False)
        ax = grp.plot(ax=ax, kind='line', x="day", y='proba', label=str(grp["n_tweets"].iloc[0]))

    plt.legend(loc='best')
    plt.show()
    
#_df = pd.read_csv('./data/fake_news_tweets.csv')
#plot_tweet_distributions_per_day(_df)

########################
def show_twitter_market_research(csv_path):
    df = pd.read_csv(csv_path)
    
    # number of tweets per week
    b=bucketed(df)
    b.plot(title="Tweets per Week")
    plt.show()
    
    # distribution of tweets per week
    vals = b["count"].value_counts()
    bins = vals.size
    b["count"].plot(kind="hist",bins=bins, title="Tweets per Week Distribution")
    plt.show()
    
    # freq of tweets per day
    df['day_of_week'] = pd.to_datetime(df['created_at']).dt.day_name()
    
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['day_of_week'].value_counts().reindex(weekdays).plot(kind='bar', title="Tweets per Calendar Day")
    plt.show()

#####################
def to_range_str(range):
    return str(range.start) + "-" + str(range.stop-1)

def append_count(series, count, category_range):
    return series.append(pd.Series([ count ], index=[ to_range_str(category_range) ]))

# takes dataframe with tweet counts bucketed per n days
# returns a data frame that returns counts for a category based, excluding coun
# this answers: what is the probability that we end in a category, given that we have already seen curr_n values
def count_adjusted(df, categories, curr_n):
    grouped = pd.Series()
    for rnge in categories:
        adjusted_range = range(max(rnge.start-curr_n, 0), max(rnge.stop-curr_n, 0 ))
        count = df[df["count"].between(adjusted_range.start, adjusted_range.stop-1)].shape[0]
        grouped = append_count(grouped, count, rnge)
    return grouped

#######################
def eval_twitter_market(market, path, data=None, ts=None, show_market_research=False):
    if show_market_research:
        show_twitter_market_research(path)
        
    if data is None:
        data = fetch_market_data(market["id"])
        
    contracts = data["contracts"]
    for c in contracts:
        c_id = str(c["id"])
        annotations = market["contract_map"][c_id]
        c["range"] = annotations["range"]
        c["category"] = to_range_str(c["range"])
    print(data["shortName"])
        
    end_date_str = contracts[0]["dateEnd"]
    end_date = parser.parse(end_date_str)
    start_date = end_date - datetime.timedelta(days=7)
    n_days = days_left(end_date, ts)
    print("Days left:", n_days)
    
    timezone = pytz.timezone("US/Eastern")
    from_date = timezone.localize(start_date)
    recent = get_recent_tweets(market["twitter_handle"], from_date=from_date)
    if ( "filter" in market.keys() ):
        recent = recent[recent["text"].str.contains(market["filter"],case=False)]
        #n_matching_tweets = len(recent[recent["text"].str.contains("fake news|fakenews",case=False)])
    n_matching_tweets = len(recent)
    print("Matching tweets:", n_matching_tweets)
    
    df = pd.read_csv(path)
    
    df['day_of_week'] = pd.to_datetime(df['created_at']).dt.day_name()
    weekdays = calendar.day_name
    timezone = pytz.timezone("US/Eastern")
    from_date = timezone.localize(datetime.datetime.now())
    circular_weekdays = np.tile(weekdays, 2)
    idx = np.where(circular_weekdays == from_date.strftime("%A"))[0][0]
    weekdays_left = circular_weekdays[idx:idx+n_days]
    
    df = df[df["day_of_week"].isin(weekdays_left)]   
    b=bucketed(df, start_on=weekdays[idx])
    c=count_adjusted(b, [c["range"] for c in contracts], n_matching_tweets )
    proba = c/c.sum()
    print("Category probabilities:")
    pprint.pprint(proba)

        
#    for c in contracts: 
#        #print("Contract", c["name"])
#        category = c["category"]
#        expected_values = eval_trade_variations(c, proba, category, positions)
#        #print(json.dumps(expected_values, indent=4))
#        for k, v in expected_values.items():
#            ev = v[0]
#            price = v[1]
#            yes_or_no = "yes" if "yes" in k else "no"
#            action = "buy" if "buy" in k else "sell"
#            if ev > 0:
#                # transaction - action (b/s), type (y/n), price, quantity, ev
#                
#                #place_order({
#                #    "action": action, 
#                #    "type": yes_or_no, 
#                #    "price": price,
#                #    "quantity": quantity,
#                #    "ev": ev,
#                #    "market_id": market["id"],
#                #    "contract_id": c["id"]
#                #})
#                if "buy" in k:
#                    recommendation_buy(category, yes_or_no, ACCOUNT_BALANCE, ev, price, positions)
#                else:
#                    # is a sell
#                    p = k.split('_')
#                    bought_at = float(p[2])
#                    quantity = float(p[3])
#                    recommendation_sell(category, yes_or_no, ev, price, quantity, bought_at)
    
    category_stats = pd.DataFrame({ "price_per_share": [], "proba": [] })
    for c in contracts:
        s = pd.Series({ "price_per_share": c["bestBuyYesCost"], "proba": proba[c["category"]] })
        s.name = c["category"]
        category_stats = category_stats.append(s)
    alloc = kelly_criterion(category_stats)
    print(alloc)
    place_orders(market["id"], contracts, alloc, ACCOUNT_BALANCE * .1)
    
    #outcomes(positions, [c["category"] for c in contracts])

def eval_trade_variations(contract, proba, category, positions):
    proba_yes = proba[category]
    
    buy_yes = contract["bestBuyYesCost"]
    buy_no = contract["bestBuyNoCost"]
    sell_yes = contract["bestSellYesCost"]
    sell_no = contract["bestSellNoCost"]
    
    if buy_yes and buy_no and 1 - buy_yes - buy_no > 0:
            print("Arbitrage opportunity BUY:", category, "contract, ", buy_yes, buy_no)
    
    d = {}
    
    if buy_yes:
        d["buy_yes"] = (expected_value(1-buy_yes, proba_yes, buy_yes, 1-proba_yes), buy_yes)
        
    if buy_no:
        d["buy_no"] = (expected_value(1-buy_no, 1-proba_yes, buy_no, proba_yes), buy_no)

    if category in positions and "yes" in positions[category]:
        yes_positions = positions[category]["yes"]
        if not sell_yes:
            # if there are no buyers on market, calculate EV of a sell at 99 cents so we may determine if we should list at all
            sell_yes = .99
        for pos in yes_positions:
            strike_price = pos[0]
            quantity = pos[1]
            ev = (sell_yes - strike_price) - expected_value(1-strike_price, proba_yes, strike_price, 1-proba_yes)
            key = "sell_yes_"+str(strike_price)+"_"+str(quantity)
            d[key] = (ev, sell_yes)
                    
    if category in positions and "no" in positions[category]:
        no_positions = positions[category]["no"]
        if not sell_no:
            # if there are no buyers on market, calculate EV of a sell at 99 cents so we may determine if we should list at all
            sell_no = .99
        for pos in no_positions:
            strike_price = pos[0]
            quantity = pos[1]
            ev = (sell_no - strike_price) - expected_value(1-strike_price, 1-proba_yes, strike_price, proba_yes)
            key = "sell_no_"+str(strike_price)+"_"+str(quantity)
            d[key] = (ev, sell_no)
    return d

def days_left(end_date, ts=None):
    if not ts:
        ts = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=7)
    delta = ts - start_date
    days_left = ((7*24) - (delta.total_seconds()/3600))/24
    return max(round(days_left),1)
                    
def outcomes(positions, categories):
    for c in categories:
        total = 0
        for pp in positions:
            if pp == c:
                if "yes" in positions[pp]:
                    for x in positions[pp]["yes"]:
                        total += (1 - x[0])*x[1]
                if "no" in positions[pp]:
                    for x in positions[pp]["no"]:
                        total -= x[0]*x[1]
            else:
                if "yes" in positions[pp]:
                    for x in positions[pp]["yes"]:
                        total -= x[0]*x[1]
                if "no" in positions[pp]:
                    for x in positions[pp]["no"]:
                        total += (1-x[0])*x[1]
        print(c, total)

##############################
markets = [
    { 
        "id": 5457, 
        "twitter_handle": "@vp", 
        "contract_map": {
            "15263": { "range": range(0, 20) }, 
            "15267": { "range": range(20, 25) }, 
            "15266": { "range": range(25, 30) }, 
            "15268": { "range": range(30, 35) },
            "15264": { "range": range(35, 40) },
            "15269": { "range": range(40, 45) },
            "15265": { "range": range(45, 100) }
        },
        "positions":{ 
        }
    },
    { 
        "id": 5407, 
        "twitter_handle": "@whitehouse", 
        "contract_map": {
            "14983": { "range": range(0, 80) }, 
            "14985": { "range": range(80, 85) }, 
            "14984": { "range": range(85, 90) },
            "14986": { "range": range(90, 95) },
            "14987": { "range": range(95, 100) },
            "14988": { "range": range(100, 105) }, 
            "14989": { "range": range(105, 300) }
        },
        #"contract_map": [ ("14983", range(0, 80)), ("14985",range(80, 85)), ("14984", range(85, 90)), ("14986", range(90, 95)), ("14987",range(95, 100)), ("14988", range(100, 105)), ("14989",range(105, 300))],
        "positions": { 
        }
    },
    {
        "id": 5404, 
        "twitter_handle": "@realDonaldTrump",
        "contract_map": {
            "14968": { "range": range(0, 60) }, 
            "14963": { "range": range(60, 65) }, 
            "14967": { "range": range(65, 70) },
            "14965": { "range": range(70, 75) },
            "14964": { "range": range(75, 80) },
            "14966": { "range": range(80, 85) }, 
            "14962": { "range": range(85, 200) }
        },
    },
    { 
        "id": 5458, 
        "twitter_handle": "@potus", 
        "contract_map": {
            "15270": { "range": range(0, 45) }, 
            "15274": { "range": range(45, 50) },
            "15275": { "range": range(50, 55) },
            "15271": { "range": range(55, 60) }, 
            "15272": { "range": range(60, 65) }, 
            "15273": { "range": range(65, 69) },
            "15276": { "range": range(70, 200) }
        },
        "positions": {
        }
    }
]

def eval_markets(show_market_research=False):
    for market in markets:
        eval_twitter_market(market, "data/tweets/{handle}.csv".format(handle=market["twitter_handle"]), show_market_research=show_market_research)
        print("----------------------------------------\n\n")

#################
eval_markets(show_market_research=False)
#################
# 1 pct = .005
# 2 pct = .01
# 4 pct = .02
# 8 pct = .03
def alloc(expected_value, proba): 
    pct_alloc = min( expected_value / 2, .03)
    risk_adjusted = pct_alloc# * ??
    return risk_adjusted
###############

TAX_RATE = .1
def kelly_criterion(outcomes):
    # category, price_per_share, proba
    er = []
    betas = []
    for index, o in outcomes.iterrows():
        payoff_odds = (1 / o["price_per_share"]) - 1
        beta = 1 / (1 + payoff_odds)
        dividend_rate = 1 - TAX_RATE
        expected_revenue_rate = (dividend_rate / beta) * o["proba"]
        er.append(expected_revenue_rate)
        betas.append(beta)
        
    outcomes["expected_revenue_rate"] = er
    outcomes["beta"] = betas
    outcomes = outcomes.sort_values("expected_revenue_rate", ascending=False)
        
    reserve_rate = 1
    optimal_set = pd.DataFrame()
    for index, o in outcomes.iterrows():
        if o["expected_revenue_rate"] > reserve_rate:
            optimal_set = optimal_set.append(o)
            reserve_rate = (1 - optimal_set["proba"].sum()) / (1 - (optimal_set["beta"] / dividend_rate).sum())
        else:
            break
    
    pct_alloc = [] 
    for index, o in optimal_set.iterrows():
        pct = (o["expected_revenue_rate"] - reserve_rate) / ( dividend_rate / o["beta"] )
        pct_alloc.append(pct)
    optimal_set["pct_alloc"] = pct_alloc
    return optimal_set

def place_orders(market_id, contracts, optimal_set, account_balance):
    for contract in contracts:
        category = contract["category"]
        price_per_share = contract["bestBuyYesCost"]
        current_quantity = current_alloc(market_id, contract["id"])
        
        if category not in optimal_set.index:
            # TODO: this doesnt take into consideration sell price, which in this market is usually less than buy price
            # could sell at best buy price...
            if current_quantity > 0:
                place_order({
                    "action": "sell",
                    "category": category,
                    "type": "yes", 
                    "price_per_share": price_per_share,
                    "quantity": current_quantity,
                    #"ev": "unknown",
                    "market_id": market_id,
                    "contract_id": contract["id"]
                })
        else:
            row = optimal_set.loc[category,:]
            optimal_alloc = (row["pct_alloc"] * account_balance)
            optimal_quantity = round( abs(optimal_alloc / price_per_share) )
            quantity = optimal_quantity - current_quantity

            if quantity > 0:
                place_order({
                    "action": "buy", 
                    "category": category,
                    "type": "yes", 
                    "price_per_share": price_per_share,
                    "quantity": quantity,
                    #"ev": row["proba"] - row["price_per_share"],
                    "market_id": market_id,
                    "contract_id": contract["id"]
                })
                
#############################
import psycopg2
import pandas.io.sql as psql

#conn = psycopg2.connect(database="predictit", host="localhost", port="5432")
#conn = create_engine("postgresql+psycopg2://@localhost:5432/predictit"
db_string = "postgresql+psycopg2://@localhost:5432/predictit"
def current_alloc(market_id, contract_id):
    contract_orders = psql.read_sql("SELECT * from orders WHERE market_id = \'{m_id}\' AND contract_id = \'{c_id}\'".format(m_id=market_id, c_id=contract_id), db_string)

    quantity = 0
    for i,o in contract_orders.iterrows():
        multiplier = -1 if o["action"] == "sell" else 1
        quantity += o["quantity"] * multiplier
    return quantity

def place_order(order, verbose=True):
    df = pd.Series(order).to_frame().transpose()
    print(df)
    df.to_sql('orders', con=db_string, if_exists='append', index=False)
    if verbose:
        print(order)
        
def record_timepoint(market_id=5458):
    data = fetch_market_data(market_id)
    twitter_handle = re.match(r".*@(\w{1,15})",data["shortName"]).group(0).split(' ')[-1]
    df = pd.Series({ "timestamp": data["timeStamp"], "market_id": data["id"], "handle": twitter_handle, "data": json.dumps(data) }).to_frame().transpose()
    df.to_sql('market_data', con=db_string, if_exists='append', index=False)   
record_timepoint()

#################################
# bought_at: .10
# EV: .70
# price: .75
# SELL (not in optimal set)

# bought_at: .10
# EV: .70
# price: .50
# BUY (but would have more shares than recommended)

# bought_at: .10 
# EV: .70
# price: .05
# BUY (difference over current alloc)

# bought_at: .90
# EV: .70
# price: .75
# SELL (not in optimal set)

# bought_at: .90
# EV: .70
# price: .50
# BUY (likely allocation is less than what you have, in which case you sell)

# bought_at: .90
# EV: -.10
# price: .50
#

market = { 
    "id": 5411, 
    "twitter_handle": "@potus", 
    "contract_map": [ ("15008", range(0, 35)), ("15010",range(35, 40)), ("15011", range(40, 45)), ("15012", range(45, 50)), ("15013",range(50, 55)), ("15009", range(55, 60)), ("15014",range(60, 200))]
}
df = pd.DataFrame({ "price_per_share": [.2, .51, .40, .01, .10], "proba": [.30, .10, .60, .02, .7] }, index=["0-59", "60-64", "65-69", "70-71", "test"])
df = pd.DataFrame({ "price_per_share": [.10, .2, .40], "proba": [.3, .30, .60] }, index=["1", "2", "3"])
alloc = kelly_criterion(df)
alloc

#################
# bet A - nothing
# bet B - .04
# if A is in optimal group

# if you dont have shares for a category, see if its in optimal set at the current price. if so, buy.
# if you do have shares for a category, see if its still in optimal set at that average price. 
  # if number of optimal shares is less than current (including 0), sell shares to get to optimal alloc
  # eval optimal set with current price to determine if we should buy more
    
# OR

# eval current portfolio first
# at current average price, compute optimal set. if less, sell
#fetch_market_data(5458)
####################
def simulate_market(market):
    historical_data = psql.read_sql("SELECT * from market_data WHERE handle = \'{handle}\'".format(handle=market["twitter_handle"]), db_string)
    path = "data/tweets/{handle}.csv".format(handle=market["twitter_handle"])
    for i, data_point in historical_data.iterrows():
        ts = parser.parse(data_point["timestamp"])
        eval_twitter_market(market, path, data=json.loads(data_point["data"]), ts=ts, show_market_research=False)

def simulate_markets():
    for m in markets:
        simulate_market(m)
###################
market = { 
    "id": 5458, 
    "twitter_handle": "@potus", 
    "contract_map": {
        "15270": { "range": range(0, 45) }, 
        "15274": { "range": range(45, 50) },
        "15275": { "range": range(50, 55) },
        "15271": { "range": range(55, 60) }, 
        "15272": { "range": range(60, 65) }, 
        "15273": { "range": range(65, 69) },
        "15276": { "range": range(70, 200) }
    }
}

simulate_markets()
