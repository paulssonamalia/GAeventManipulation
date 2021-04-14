import csv
import pandas as pd
import numpy as np
import math
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from datetime import date
import logging

#Return errands within a session with time difference and number of clicks between subsequent 'Enter Profile'-events and 'Save Changes'-events.
def getErrands(session):
    errands = []
    startTime, endTime, clicks = 0, 0, 0
    date = None
    version = 'original'
    lastTab = 'Landing page'
    changeSaved = False
    tabClicks = {'OverviewClick': 0, 'AddressClick': 0, 'ContactClick': 0, 'IndividualClick': 0, 'TransactionClick': 0, 'PurchasesClicks': 0, 'RewardsClicks': 0, 'OrdersClicks': 0, 'InteractionsClicks': 0, 'ProfileClicks': 0}
    clicksInTab = {'ClicksInOverview': 0, 'ClicksInAddress': 0, 'ClicksInContact': 0, 'ClicksInIndividual': 0, 'ClicksInTransaction': 0, 'ClicksInPurchases': 0, 'ClicksInRewards': 0, 'ClicksInOrders': 0, 'ClicksInInteractions': 0, 'ClicksInProfile': 0}
    lastTabs = {'lastOverview': 0, 'lastAddress': 0, 'lastContact': 0, 'lastIndividual': 0, 'lastTransaction': 0, 'lastPurchases': 0, 'lastRewards': 0, 'lastOrders': 0, 'lastInteractions': 0, 'lastProfile': 0}
    #set version to treatment
    def setToTreatment():
        nonlocal version
        version = 'treatment'    
    def increaseTabClicks(action):
        if action not in ['Non Key Clicks', 'Enter Profile', 'Search Page', 'Reset Password', 'Save Changes']:
            tabClicks[action.replace(' ', '')] += 1
        if action in ['PurchasesClicks', 'RewardsClicks', 'OrdersClicks', 'InteractionsClicks', 'ProfileClicks']:
            setToTreatment()
    #incrase click for action's respective tab 
    def increaseClicksInTab(action):
        clicksInTab[action.replace(' ', '')] += 1
    session = session.sort_values(by=['Timestamp'])#Sort events after ascending timestamp
    
    for index, row in session.iterrows(): #Iterate through every event within session
        date = row['Date Hour and Minute']
        if row['Event Action'] not in ['Non Key Clicks', 'Enter Profile', 'Search Page', 'Reset Password', 'Save Changes']:
            increaseTabClicks(row['Event Action'])
        if clicks and row['Event Action'] != 'Enter Profile' and not changeSaved:#increase number of clicks only between entering profile and saving change or within the same profile
            clicks +=1
            if lastTab != 'Landing page' and row['Event Action'] == 'Non Key Clicks':
                increaseClicksInTab(lastTab)
        if 'Click' in row['Event Action'] and not changeSaved and row['Event Action'] != 'Non Key Clicks':#Update latest tab
            lastTab = row['Event Action']
        if 'Full URL' in row and '/start' in row['Full URL']:#if URL ends with start, the errand is in treatment
            version = 'treatment'
        if row['Event Action'] == 'Enter Profile' or row['Event Action'] == 'Search Page' and row['Total Events'] == 1.0:
            if not clicks or clicks == 1:#if profile triggers for the first time or second in a row, an errand is initiated
                startTime = float(row['Timestamp'])
                clicks = 1
            elif clicks>1:#if profile triggers after other activities, the errand has ended
                if not changeSaved:#if no changes have been made, endTime is updated
                    endTime = float(row['Timestamp'])
                errands.append({'changeSaved': changeSaved, 'timeDiff': endTime-startTime, 'clicks': clicks, 'date': date, 'version': version, 'lastTab': lastTab,
                'OverviewClicks': tabClicks['OverviewClick'], 'AddressClicks': tabClicks['AddressClick'], 'ContactClicks': tabClicks['ContactClick'], 'IndividualClicks': tabClicks['IndividualClick'], 
                'TransactionClicks': tabClicks['TransactionClick'], 'PurchasesClicks': tabClicks['PurchasesClicks'], 'RewardsClicks': tabClicks['RewardsClicks'], 'OrdersClicks': tabClicks['OrdersClicks'], 
                'InteractionsClicks': tabClicks['InteractionsClicks'], 'ProfileClicks': tabClicks['ProfileClicks'], 'ClicksInOverview': clicksInTab['ClicksInOverview'], 'ClicksInAddress': clicksInTab['ClicksInAddress'], 
                'ClicksInContact': clicksInTab['ClicksInContact'], 'ClicksInIndividual': clicksInTab['ClicksInIndividual'], 'ClicksInTransaction': clicksInTab['ClicksInTransaction'], 'ClicksInPurchases': clicksInTab['ClicksInPurchases'], 
                'ClicksInRewards': clicksInTab['ClicksInRewards'], 'ClicksInOrders': clicksInTab['ClicksInOrders'], 'ClicksInInteractions': clicksInTab['ClicksInInteractions'], 'ClicksInProfile': clicksInTab['ClicksInProfile']})
                startTime, endTime, clicks = 0, 0, 0
                changeSaved = False
                tabClicks = {'OverviewClick': 0, 'AddressClick': 0, 'ContactClick': 0, 'IndividualClick': 0, 'TransactionClick': 0, 'PurchasesClicks': 0, 'RewardsClicks': 0, 'OrdersClicks': 0, 'InteractionsClicks': 0, 'ProfileClicks': 0}
                clicksInTab = {'ClicksInOverview': 0, 'ClicksInAddress': 0, 'ClicksInContact': 0, 'ClicksInIndividual': 0, 'ClicksInTransaction': 0, 'ClicksInPurchases': 0, 'ClicksInRewards': 0, 'ClicksInOrders': 0, 'ClicksInInteractions': 0, 'ClicksInProfile': 0}
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
    d1 = pd.DataFrame(columns = ['changeSaved', 'timeDiff', 'clicks', 
                    'date', 'version', 'lastTab', 'OverviewClicks', 'AddressClicks', 
                    'ContactClicks', 'IndividualClicks', 'TransactionClicks', 'PurchasesClicks', 
                    'RewardsClicks', 'OrdersClicks', 'InteractionsClicks', 'ProfileClicks', 
                    'ClicksInOverview', 'ClicksInAddress', 'ClicksInContact', 'ClicksInIndividual', 
                    'ClicksInTransaction', 'ClicksInPurchases', 'ClicksInRewards', 'ClicksInOrders', 
                    'ClicksInInteractions', 'ClicksInProfile'])
    d2 = pd.DataFrame(columns = ['changeSaved', 'timeDiff', 'clicks', 
                    'date', 'version', 'lastTab', 'OverviewClicks', 'AddressClicks', 
                    'ContactClicks', 'IndividualClicks', 'TransactionClicks', 'PurchasesClicks', 
                    'RewardsClicks', 'OrdersClicks', 'InteractionsClicks', 'ProfileClicks', 
                    'ClicksInOverview', 'ClicksInAddress', 'ClicksInContact', 'ClicksInIndividual', 
                    'ClicksInTransaction', 'ClicksInPurchases', 'ClicksInRewards', 'ClicksInOrders', 
                    'ClicksInInteractions', 'ClicksInProfile'])
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

#Return dataFrame with mean, count, std and 95% confidence intervall for pairs of units and dataframes in dict.
def getStats(dict):
    #return the frequence of which each tabs were lastly used in an errand
    def getLastTabDistribution(lastTab):
        overview, address, contact, individual, transaction, purchases, rewards, orders, interactions, profile = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        if lastTab == 'Overview Click':
            overview = 1
        elif lastTab == 'Address Click':
            address = 1
        elif lastTab == 'Contact Click':
            contact = 1
        elif lastTab == 'Individual Click':
            individual = 1
        elif lastTab == 'Transaction Click':
            transaction = 1
        elif lastTab == 'Purchases Clicks':
            purchases = 1
        elif lastTab == 'Rewards Clicks':
            rewards = 1
        elif lastTab == 'Orders Clicks':
            orders = 1 
        elif lastTab == 'Interactions Clicks':
            interactions = 1
        elif lastTab == 'Profile Clicks':
            profile = 1
        return {'Overview': overview, 'Address': address, 'Contact': contact, 'Individual': individual, 'Transaction': transaction, 
        'Purchases': purchases, 'Rewards': rewards, 'Orders': orders, 'Interactions': interactions, 'Profile': profile}

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
    ci95_hi = []
    ci95_lo = []
    for i in stats.index:
        m, c, s = stats.loc[i]
        ci95_hi.append(m + 1.95*s/math.sqrt(c))
        ci95_lo.append(m - 1.95*s/math.sqrt(c))
    stats['ci95_hi'] = ci95_hi
    stats['ci95_lo'] = ci95_lo
    return stats

d1, d2 = populateDataFrames()
print(d1)
print(d2)

#userNeeded(d1, 16)

logging.basicConfig(filename='logfile' + str(date.today()) + '.log', level=logging.INFO, format='%(message)s')
logging.info('Original')
logging.info('Statistics')
logging.info(getStats({'timeDiff': d1[d1['changeSaved']], 'changeSaved, clicks, OverviewClicks, AddressClicks, ContactClicks, IndividualClicks, TransactionClicks, PurchasesClicks, RewardsClicks, OrdersClicks, InteractionsClicks, ProfileClicks, ClicksInOverview, ClicksInAddress, ClicksInContact, ClicksInIndividual, ClicksInTransaction, ClicksInPurchases, ClicksInRewards, ClicksInOrders, ClicksInInteractions, ClicksInProfile': d1}))
#logging.info('Dickey-Fuller Test')
#dickeyfuller(d1)
logging.info('Treatment')
logging.info('Statistics')
logging.info(getStats({'timeDiff': d2[d2['changeSaved']], 'changeSaved, clicks, OverviewClicks, AddressClicks, ContactClicks, IndividualClicks, TransactionClicks, PurchasesClicks, RewardsClicks, OrdersClicks, InteractionsClicks, ProfileClicks, ClicksInOverview, ClicksInAddress, ClicksInContact, ClicksInIndividual, ClicksInTransaction, ClicksInPurchases, ClicksInRewards, ClicksInOrders, ClicksInInteractions, ClicksInProfile': d2}))
#logging.info('Dickey-Fuller Test')
#dickeyfuller(d2)

#graph plot
d1 = d1.sort_values(by=['date'])#Sort sessions after ascending date
d1.plot(x='date', y='timeDiff', style = 'o')
plt.show()

#forecastULeffect(d1)
#d1 = d1.set_index('date')
#print(d1)
#print('-'*50)