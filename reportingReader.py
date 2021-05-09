import csv
import pandas as pd
import numpy as np
import math
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import datetime
import logging
import seaborn as sns
import matplotlib.dates as md
from scipy import stats as st

#Return errands within a session with time difference and number of clicks between subsequent 'Enter Profile'-events and 'Save Changes'-events.
def getErrands(session):
    errands = []
    startTime, endTime, clicks = 0, 0, 0
    date = None
    version = 'original'
    lastTab = 'Overview Click'
    changeSaved = False
    tabClicks = {'OverviewClick': 0, 'AddressClick': 0, 'ContactClick': 0, 'IndividualClick': 0, 'TransactionClick': 0, 'PurchasesClicks': 0, 'RewardsClicks': 0, 'OrdersClicks': 0, 'InteractionsClicks': 0, 'ProfileClicks': 0}
    clicksInTab = {'ClicksInOverview': 0, 'ClicksInAddress': 0, 'ClicksInContact': 0, 'ClicksInIndividual': 0, 'ClicksInTransaction': 0, 'ClicksInPurchases': 0, 'ClicksInRewards': 0, 'ClicksInOrders': 0, 'ClicksInInteractions': 0, 'ClicksInProfile': 0}
    lastTabs = {'lastOverview': 0, 'lastAddress': 0, 'lastContact': 0, 'lastIndividual': 0, 'lastTransaction': 0, 'lastPurchases': 0, 'lastRewards': 0, 'lastOrders': 0, 'lastInteractions': 0, 'lastProfile': 0}   
    #increase value to the key in tabClicks that match with action
    def increaseTabClicks(action):
        if action in ['Overview Click', 'Address Click', 'Contact Click', 'Individual Click', 'Transaction Click', 'Purchases Clicks', 'Rewards Clicks', 'Orders Clicks', 'Interactions Clicks', 'Profile Clicks']:
            tabClicks[action.replace(' ', '')] += 1
    #incrase click for action's respective tab 
    def increaseClicksInTab(action):
        clicksInTab['ClicksIn' + action.split(' ')[0]] += 1

    session = session.sort_values(by=['Timestamp'])#Sort events after ascending timestamp
    for index, row in session.iterrows(): #Iterate through every event within session
        date = row['Date Hour and Minute']
        increaseTabClicks(row['Event Action'])
        if clicks and row['Event Action'] != 'Enter Profile' and not changeSaved:#increase number of clicks only between entering profile and saving change or within the same profile
            clicks +=1
            if row['Event Action'] == 'Non Key Clicks':
                increaseClicksInTab(lastTab)
        if 'Click' in row['Event Action'] and not changeSaved and row['Event Action'] != 'Non Key Clicks':#Update latest tab
            lastTab = row['Event Action']
        if 'Full URL' in row and '/start' in row['Full URL'] or row['Event Action'] in ['Purchases Clicks', 'Rewards Clicks', 'Orders Clicks', 'Interactions Clicks', 'Profile Clicks']:
            version = 'treatment'
        if row['Event Action'] == 'Enter Profile' or row['Event Action'] == 'Search Page' and row['Total Events'] == 1.0:
            if not clicks or clicks == 1:#if profile triggers for the first time or second in a row, an errand is initiated
                startTime = float(row['Timestamp'])
                clicks = 1
            elif clicks>1:#if profile triggers after other activities, the errand has ended
                if not changeSaved:#if no changes have been made, endTime is updated
                    endTime = float(row['Timestamp'])
                lastTabs['last' + lastTab.split(' ')[0]] = 1#TODO look after different lastTab - Landing Page
                errands.append({'changeSaved': changeSaved, 'timeDiff': endTime-startTime, 'clicks': float(clicks), 'date': date, 'version': version, 'lastTab': lastTab,
                'OverviewClicks': tabClicks['OverviewClick'], 'AddressClicks': tabClicks['AddressClick'], 'ContactClicks': tabClicks['ContactClick'], 'IndividualClicks': tabClicks['IndividualClick'], 
                'TransactionClicks': tabClicks['TransactionClick'], 'PurchasesClicks': tabClicks['PurchasesClicks'], 'RewardsClicks': tabClicks['RewardsClicks'], 'OrdersClicks': tabClicks['OrdersClicks'], 
                'InteractionsClicks': tabClicks['InteractionsClicks'], 'ProfileClicks': tabClicks['ProfileClicks'], 'ClicksInOverview': clicksInTab['ClicksInOverview'], 'ClicksInAddress': clicksInTab['ClicksInAddress'], 
                'ClicksInContact': clicksInTab['ClicksInContact'], 'ClicksInIndividual': clicksInTab['ClicksInIndividual'], 'ClicksInTransaction': clicksInTab['ClicksInTransaction'], 'ClicksInPurchases': clicksInTab['ClicksInPurchases'], 
                'ClicksInRewards': clicksInTab['ClicksInRewards'], 'ClicksInOrders': clicksInTab['ClicksInOrders'], 'ClicksInInteractions': clicksInTab['ClicksInInteractions'], 'ClicksInProfile': clicksInTab['ClicksInProfile'],
                'lastOverview': lastTabs['lastOverview'], 'lastAddress': lastTabs['lastAddress'], 'lastContact': lastTabs['lastContact'], 'lastIndividual': lastTabs['lastIndividual'], 'lastTransaction': lastTabs['lastTransaction'], 
                'lastPurchases': lastTabs['lastPurchases'], 'lastRewards': lastTabs['lastRewards'], 'lastOrders': lastTabs['lastOrders'], 'lastInteractions': lastTabs['lastInteractions'], 'lastProfile': lastTabs['lastProfile']})
                startTime, endTime, clicks = 0, 0, 0
                changeSaved = False
                lastTabs['last' + lastTab.split(' ')[0]] = 0
                tabClicks = dict.fromkeys(tabClicks, 0)
                clicksInTab = dict.fromkeys(clicksInTab, 0)
        elif row['Event Action'] == 'Save Changes' and startTime and not endTime and row['Total Events'] == 1.0:
            endTime = float(row['Timestamp'])
            changeSaved = True
    return errands

#Print test-statistics, n lags and p-value from a dickey-fuller test of the timeDiffs in dataframe
def dickeyfuller(dataframe):
    result = adfuller(dataframe.loc[:, ['timeDiff']], autolag='AIC')
    logging.info(f'ADF Statistic: {result[0]}')
    logging.info(f'n_lags: {result[1]}')
    logging.info(f'p-value: {result[1]}')
    for key, value in result[4].items():
        logging.info('Critial Values:')
        logging.info(f'   {key}, {value}')  

#print the number of users in each variant, assuming the desired confidence level is 95% and the desired power is 80%
#delta is the amount of change you want to detect
def userNeeded(dataframe, delta):
    n = 16*pow(dataframe.std(),2)['timeDiff']/pow(delta, 2)
    print('number of sessions needed in each variant is', n)

#Creates and prints the accuracy of a sequential model with three layers for the timeDiff series in dataFrame d.
def forecastULeffect(d):
    X = d.loc[:, ['timeDiff']]
    y = d.loc[:, ['date']]
    model = Sequential([Dense(units=64, activation='relu'), Dense(units=64, activation='relu'), Dense(units=1)])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=150)
    _, accuracy = model.evaluate(X, y)
    print('Accuracy: %.2f' % (accuracy*100))

#Return a dataFrame populated with sessionID, timeDiff, clicks, date and version.
def populateDataFrames():
    df = pd.read_csv('reporting.csv')
    columns = ['changeSaved', 'timeDiff', 'clicks',
                'date', 'version', 'lastTab', 'OverviewClicks', 'AddressClicks', 
                'ContactClicks', 'IndividualClicks', 'TransactionClicks', 'PurchasesClicks', 
                'RewardsClicks', 'OrdersClicks', 'InteractionsClicks', 'ProfileClicks', 
                'ClicksInOverview', 'ClicksInAddress', 'ClicksInContact', 'ClicksInIndividual', 
                'ClicksInTransaction', 'ClicksInPurchases', 'ClicksInRewards', 'ClicksInOrders', 
                'ClicksInInteractions', 'ClicksInProfile', 'lastOverview', 'lastAddress', 
                'lastContact', 'lastIndividual', 'lastTransaction', 'lastPurchases', 'lastRewards', 
                'lastOrders', 'lastInteractions', 'lastProfile']
    d1 = pd.DataFrame(columns = columns)
    d2 = pd.DataFrame(columns = columns)
    for sessionID in df['SessionID'].unique():
        errands = getErrands(df[df['SessionID'].str.contains(sessionID)])
        if errands == []:
            continue
        for errand in errands:
            if errand['version'] == 'original':
                d1 = d1.append(errand, ignore_index=True)
            else:
                d2 = d2.append(errand, ignore_index=True)
    return d1, d2   

#Return dataFrame with mean, count, std and 90% confidence intervall for pairs of units and dataframes in dict.
def getStats(dict):
    metrics = []
    for key in dict.keys():
        for metric in key.split(', '):
            metrics.append(metric)
    stats = pd.DataFrame(columns = metrics)
    for units in dict:
        df = dict[units]
        for index, row in df.iterrows():
            stats = stats.append(row[units.split(', ')], ignore_index=True)
    stats = stats.agg(['mean', 'count', 'std']).transpose()
    ci90 = []
    for i in stats.index:
        m, c, s = stats.loc[i]
        ci90.append(1.64*s/math.sqrt(c))#1.96 for 95% confidence
    stats['+-ci90'] = ci90
    return stats

#Round dateTime to closest day
def getDateAvg(units, df):
    data = df
    data['Datetime'] = pd.to_datetime(data.loc[:, 'date'], format='%Y%m%d%H%M', errors='ignore')
    data['Datetime'] = data['Datetime'].apply(lambda dt: datetime.datetime(dt.year, dt.month, dt.day))
    data = data.drop(['date'], axis=1)
    #data = data.set_index('Datetime')
    #data = data.groupby(data.index.date).mean()
    return data

print('Populating DataFrames')
d1, d2 = populateDataFrames()

#userNeeded(d1, 16)

#STATISTICS
print('Calculating Statistics')
statsOriginal = getStats({'timeDiff': d1[d1['changeSaved']], 'changeSaved, clicks, errandTimeDiff, OverviewClicks, AddressClicks, ContactClicks, IndividualClicks, TransactionClicks, PurchasesClicks, RewardsClicks, OrdersClicks, InteractionsClicks, ProfileClicks, ClicksInOverview, ClicksInAddress, ClicksInContact, ClicksInIndividual, ClicksInTransaction, ClicksInPurchases, ClicksInRewards, ClicksInOrders, ClicksInInteractions, ClicksInProfile, lastOverview, lastAddress, lastContact, lastIndividual, lastTransaction, lastPurchases, lastRewards, lastOrders, lastInteractions, lastProfile': d1.rename(columns = {'timeDiff': 'errandTimeDiff'}, inplace = False)})
statsTreatment = getStats({'timeDiff': d2[d2['changeSaved']], 'changeSaved, clicks, errandTimeDiff, OverviewClicks, AddressClicks, ContactClicks, IndividualClicks, TransactionClicks, PurchasesClicks, RewardsClicks, OrdersClicks, InteractionsClicks, ProfileClicks, ClicksInOverview, ClicksInAddress, ClicksInContact, ClicksInIndividual, ClicksInTransaction, ClicksInPurchases, ClicksInRewards, ClicksInOrders, ClicksInInteractions, ClicksInProfile, lastOverview, lastAddress, lastContact, lastIndividual, lastTransaction, lastPurchases, lastRewards, lastOrders, lastInteractions, lastProfile': d2.rename(columns = {'timeDiff': 'errandTimeDiff'}, inplace = False)})

statsMerge = statsOriginal.join(statsTreatment, lsuffix='_original', rsuffix='_treatment')
pValues = []
for index, row in statsMerge.iterrows():
    df = row['count_original'] + row['count_treatment'] -2
    tValue = (row['mean_treatment'] - row['mean_original'])/np.sqrt(row['std_original']**2/row['count_original']+row['std_treatment']**2/row['count_treatment'])
    pValue = 1 - st.t.cdf(tValue,df=df)
    pValues.append(pValue)
statsMerge['p-value'] = pValues
statsMerge.pop('std_original')
statsMerge.pop('std_treatment')
statsMerge.pop('count_original')
statsMerge.pop('count_treatment')

logging.basicConfig(filename='logfile' + str(datetime.date.today()) + '.log', level=logging.INFO, format='%(message)s')
logging.info('Metric Comparison')
logging.info(statsMerge)

logging.info('Dickey-Fuller Test - Original')
dickeyfuller(d1)

logging.info('Dickey-Fuller Test - Treatment')
dickeyfuller(d2)

#GRAPH PLOT
print('Plotting Graphs')
dateAvg1 = getDateAvg(['timeDiff', 'clicks'], d1.loc[:, ['timeDiff', 'clicks', 'date']])
dateAvg2 = getDateAvg(['timeDiff', 'clicks'], d2.loc[:, ['timeDiff', 'clicks', 'date']])

fig, ax = plt.subplots(figsize = (15, 7))
sns.lineplot(ax = ax, x='Datetime', y='clicks', data=dateAvg1, ci = 80).set_title('Average nbr of clicks per errand')
sns.lineplot(ax = ax, x='Datetime', y='clicks', data=dateAvg2, ci = 80).set_title('Average nbr of clicks per errand')

plt.setp(ax.xaxis.get_majorticklabels(), rotation = 90)#rotate by 90° the labels

plt.show()

#forecastULeffect(d1)
#d1 = d1.set_index('date')
#print(d1)
#print('-'*50)
