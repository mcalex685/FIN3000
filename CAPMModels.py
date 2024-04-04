# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 09:01:00 2024

@author: alexm
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#histroical data for selected stocks 

#risk free rate (3 month t bill)
rf = pd.read_csv('/Users/alexm/Downloads/^IRX.csv')
rf.dropna(inplace=True)
rf['Date'] = pd.to_datetime(rf['Date'])
rf.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)
rf.rename(columns={'Close': 'rf'}, inplace=True)
rf.set_index('Date', inplace=True)
# Divide rf rate by avg number of trading days in a month * 3 months 
rf['rf'] = rf['rf'] / (3*21)

rf2 = rf['rf'].mean() * 264


#S&P500
rm = pd.read_csv('/Users/alexm/Downloads/SP500.csv')
rm.replace('.', pd.NA, inplace=True)
rm.dropna(inplace=True)
rm['Date'] = pd.to_datetime(rm['Date'], dayfirst=True)
rm.set_index('Date', inplace=True)
# Convert 'Close' column to numeric, coerce errors to NaN
rm['Close'] = pd.to_numeric(rm['Close'], errors='coerce')
#calculate daily returns 
rm['return']= rm['Close'].pct_change() * 100
#add rf 
rm = rm.join(rf)
#excess returns 
rm['excessR'] = rm['return'] - rm['rf']
#annualized 
erm_Er = rm['excessR'].mean()*(264)

#annualize historical returns 
rm_Er = rm['return'].mean()*(264)
#annualize standard devaition 
rm_sd = rm['return'].std()*(264)**(1/2)

#variance 
marketVar=rm['excessR'].var()
#beta 
rmB = rm['excessR'].cov(rm['excessR'])/marketVar

#1 microsoft 
msft = pd.read_csv('/Users/alexm/Downloads/MSFT.csv')
msft.dropna(inplace=True)
msft.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)
msft['Date'] = pd.to_datetime(msft['Date'])
msft.set_index('Date', inplace=True)
#calculate daily returns 
msft['MSFTreturn']= msft['Close'].pct_change() * 100
#add rf 
msft = msft.join(rf)

#excess returns 
msft['excessR'] = msft['MSFTreturn'] - msft['rf']

#annualize returns 
msft_Er = msft['MSFTreturn'].mean()*(264)
#annualize standard devaition 
msft_sd = msft['MSFTreturn'].std()*(264)**(1/2)

#beta 
msftB = msft['excessR'].cov(rm['excessR'])/marketVar

#expected retrun w/ beta 
msftEr = rf2 + msftB*erm_Er 

#2 Lululemon 
lulu = pd.read_csv('/Users/alexm/Downloads/LULU.csv')
lulu.dropna(inplace=True)
lulu.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)
lulu['Date'] = pd.to_datetime(lulu['Date'])
lulu.set_index('Date', inplace=True)
#calculate daily returns 
lulu['return']= lulu['Close'].pct_change() * 100
#add rf 
lulu = lulu.join(rf)

#excess returns 
lulu['excessR'] = lulu['return'] - lulu['rf']

#annualize returns 
lulu_Er = lulu['return'].mean()*(264)
#annualize standard devaition 
lulu_sd = lulu['return'].std()*(264)**(1/2)

#beta 
luluB = lulu['excessR'].cov(rm['excessR'])/marketVar

#expected retrun w/ beta 
luluEr = rf2 + luluB*erm_Er 

#3 NVIDA
nvda = pd.read_csv('/Users/alexm/Downloads/NVDA.csv')
nvda.dropna(inplace=True)
nvda.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)
nvda['Date'] = pd.to_datetime(nvda['Date'])
nvda.set_index('Date', inplace=True)
#calculate daily returns 
nvda['return']= nvda['Close'].pct_change() * 100
#add rf 
nvda = nvda.join(rf)
#excess returns 
nvda['excessR'] = nvda['return'] - nvda['rf']
#annualize returns 
nvda_Er = nvda['return'].mean()*(264)
#annualize standard devaition 
nvda_sd = nvda['return'].std()*(264)**(1/2)

#beta 
nvdaB = nvda['excessR'].cov(rm['excessR'])/marketVar

#expected retrun w/ beta 
nvdaEr = rf2 + nvdaB*erm_Er 

#4 Walmart
wmt = pd.read_csv('/Users/alexm/Downloads/WMT.csv')
wmt.dropna(inplace=True)
wmt.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)
wmt['Date'] = pd.to_datetime(wmt['Date'])
wmt.set_index('Date', inplace=True)
#calculate daily returns 
wmt['return']= wmt['Close'].pct_change() * 100
#add rf 
wmt = wmt.join(rf)
#excess returns 
wmt['excessR'] = wmt['return'] - wmt['rf']

#annualize returns 
wmt_Er = wmt['return'].mean()*(264)
#annualize standard devaition 
wmt_sd = wmt['return'].std()*(264)**(1/2)

#beta 
wmtB = wmt['excessR'].cov(rm['excessR'])/marketVar

#expected retrun w/ beta 
wmtEr = rf2 + wmtB*erm_Er 

#5 Next Era Energy 
nee = pd.read_csv('/Users/alexm/Downloads/NEE.csv')
nee.dropna(inplace=True)
nee.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)
nee['Date'] = pd.to_datetime(nee['Date'])
nee.set_index('Date', inplace=True)
#calculate daily returns 
nee['return']= nee['Close'].pct_change() * 100
#add rf 
nee = nee.join(rf)
#excess returns 
nee['excessR'] = nee['return'] - nee['rf']

#annualize returns 
nee_Er = nee['return'].mean()*(264)
#annualize standard devaition 
nee_sd = nee['return'].std()*(264)**(1/2)

#beta 
neeB = nee['excessR'].cov(rm['excessR'])/marketVar

#expected retrun w/ beta 
neeEr = rf2 + neeB*erm_Er 

#6 Enbridge  
enb = pd.read_csv('/Users/alexm/Downloads/ENB.csv')
enb.dropna(inplace=True)
enb.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)
enb['Date'] = pd.to_datetime(enb['Date'])
enb.set_index('Date', inplace=True)
#calculate daily returns 
enb['return']= enb['Close'].pct_change() * 100
#add rf 
enb = enb.join(rf)
#excess returns 
enb['excessR'] = enb['return'] - enb['rf']

#annualize returns 
enb_Er = enb['return'].mean()*(264)
#annualize standard devaition 
enb_sd = enb['return'].std()*(264)**(1/2)

#beta 
enbB = enb['excessR'].cov(rm['excessR'])/marketVar

#expected retrun w/ beta 
enbEr = rf2 + enbB*erm_Er 

#7 Canadian Natural Reasources Ltd  
cnq = pd.read_csv('/Users/alexm/Downloads/CNQ.csv')
cnq.dropna(inplace=True)
cnq.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)
cnq['Date'] = pd.to_datetime(cnq['Date'])
cnq.set_index('Date', inplace=True)
#calculate daily returns 
cnq['return']= cnq['Close'].pct_change() * 100
#add rf 
cnq = cnq.join(rf)
#excess returns 
cnq['excessR'] = cnq['return'] - cnq['rf']

#annualize returns 
cnq_Er = cnq['return'].mean()*(264)
#annualize standard devaition 
cnq_sd = cnq['return'].std()*(264)**(1/2)

#beta 
cnqB = cnq['excessR'].cov(rm['excessR'])/marketVar

#expected retrun w/ beta 
cnqEr = rf2 + cnqB*erm_Er 

#8 Royal Bank of Canada 
ry = pd.read_csv('/Users/alexm/Downloads/RY.csv')
ry.dropna(inplace=True)
ry.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)
ry['Date'] = pd.to_datetime(ry['Date'])
ry.set_index('Date', inplace=True)
#calculate daily returns 
ry['return']= ry['Close'].pct_change() * 100
#add rf 
ry = ry.join(rf)
#excess returns 
ry['excessR'] = ry['return'] - ry['rf']

#annualize returns 
ry_Er = ry['return'].mean()*(264)
#annualize standard devaition 
ry_sd = ry['return'].std()*(264)**(1/2)

#beta 
ryB = ry['excessR'].cov(rm['excessR'])/marketVar

#expected retrun w/ beta 
ryEr = rf2 + ryB*erm_Er 

#9 BMO 
bmo = pd.read_csv('/Users/alexm/Downloads/BMO.csv')
bmo.dropna(inplace=True)
bmo.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)
bmo['Date'] = pd.to_datetime(bmo['Date'])
bmo.set_index('Date', inplace=True)
#calculate daily returns 
bmo['return']= bmo['Close'].pct_change() * 100
#add rf 
bmo = bmo.join(rf)
#excess returns 
bmo['excessR'] = bmo['return'] - bmo['rf']

#annualize returns 
bmo_Er = bmo['return'].mean()*(264)
#annualize standard devaition 
bmo_sd = bmo['return'].std()*(264)**(1/2)

#beta 
bmoB = bmo['excessR'].cov(rm['excessR'])/marketVar

#expected retrun w/ beta 
bmoEr = rf2 + bmoB*erm_Er 

#10 IBM
ibm = pd.read_csv('/Users/alexm/Downloads/IBM.csv')
ibm.dropna(inplace=True)
ibm.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)
ibm['Date'] = pd.to_datetime(ibm['Date'])
ibm.set_index('Date', inplace=True)
#calculate daily returns 
ibm['return']= ibm['Close'].pct_change() * 100
#add rf 
ibm = ibm.join(rf)
#excess returns 
ibm['excessR'] = ibm['return'] - ibm['rf']

#annualize returns 
ibm_Er = ibm['return'].mean()*(264)
#annualize standard devaition 
ibm_sd = ibm['return'].std()*(264)**(1/2)

#beta 
ibmB = ibm['excessR'].cov(rm['excessR'])/marketVar

#expected retrun w/ beta 
ibmEr = rf2 + ibmB*erm_Er 

#11 Netflix 
nflx = pd.read_csv('/Users/alexm/Downloads/NFLX.csv')
nflx.dropna(inplace=True)
nflx.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)
nflx['Date'] = pd.to_datetime(nflx['Date'])
nflx.set_index('Date', inplace=True)
#calculate daily returns 
nflx['return']= nflx['Close'].pct_change() * 100
#add rf 
nflx = nflx.join(rf)
#excess returns 
nflx['excessR'] = nflx['return'] - nflx['rf']

#annualize returns 
nflx_Er = nflx['return'].mean()*(264)
#annualize standard devaition 
nflx_sd = nflx['return'].std()*(264)**(1/2)

#beta 
nflxB = nflx['excessR'].cov(rm['excessR'])/marketVar

#expected retrun w/ beta 
nflxEr = rf2 + nflxB*erm_Er 

#12 Ford Motor Co 
f = pd.read_csv('/Users/alexm/Downloads/F.csv')
f.dropna(inplace=True)
f.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)
f['Date'] = pd.to_datetime(f['Date'])
f.set_index('Date', inplace=True)
#calculate daily returns 
f['return']= f['Close'].pct_change() * 100
#add rf 
f = f.join(rf)
#excess returns 
f['excessR'] = f['return'] - f['rf']

#annualize returns 
f_Er = f['return'].mean()*(264)
#annualize standard devaition 
f_sd = f['return'].std()*(264)**(1/2)

#beta 
fB = f['excessR'].cov(rm['excessR'])/marketVar

#expected retrun w/ beta 
fEr = rf2 + fB*erm_Er 

#13 exon mobil 
xom = pd.read_csv('/Users/alexm/Downloads/XOM.csv')
xom.dropna(inplace=True)
xom.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)
xom['Date'] = pd.to_datetime(xom['Date'])
xom.set_index('Date', inplace=True)
#calculate daily returns 
xom['return']= xom['Close'].pct_change() * 100
#add rf 
xom = xom.join(rf)
#excess returns 
xom['excessR'] = xom['return'] - xom['rf']

#annualize returns 
xom_Er = xom['return'].mean()*(264)
#annualize standard devaition 
xom_sd = xom['return'].std()*(264)**(1/2)

#beta 
xomB = xom['excessR'].cov(rm['excessR'])/marketVar

#expected retrun w/ beta 
xomEr = rf2 + xomB*erm_Er 

#14 BP 
bp = pd.read_csv('/Users/alexm/Downloads/BP.csv')
bp.dropna(inplace=True)
bp.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)
bp['Date'] = pd.to_datetime(bp['Date'])
bp.set_index('Date', inplace=True)
#calculate daily returns 
bp['return']= bp['Close'].pct_change() * 100
#add rf 
bp = bp.join(rf)
#excess returns 
bp['excessR'] = bp['return'] - bp['rf']

#annualize returns 
bp_Er = bp['return'].mean()*(264)
#annualize standard devaition 
bp_sd = bp['return'].std()*(264)**(1/2)

#beta 
bpB = bp['excessR'].cov(rm['excessR'])/marketVar

#expected retrun w/ beta 
bpEr = rf2 + bpB*erm_Er 

#15 Nike 
nke = pd.read_csv('/Users/alexm/Downloads/NKE.csv')
nke.dropna(inplace=True)
nke.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)
nke['Date'] = pd.to_datetime(nke['Date'])
nke.set_index('Date', inplace=True)
#calculate daily returns 
nke['return']= nke['Close'].pct_change() * 100
#add rf 
nke = nke.join(rf)
#excess returns 
nke['excessR'] = nke['return'] - nke['rf']

#annualize returns 
nke_Er = nke['return'].mean()*(264)
#annualize standard devaition 
nke_sd = nke['return'].std()*(264)**(1/2)

#beta 
nkeB = nke['excessR'].cov(rm['excessR'])/marketVar

#expected retrun w/ beta 
nkeEr = rf2 + nkeB*erm_Er 

# Create an empty DataFrame with the columns
df = pd.DataFrame(columns=['Company','Ticker', 'Expected Return (Historical)', 'Expected Return (CAPM)', 'Volatility(%)', 'Beta'])

# Add rows one entry at a time
df.loc[0] = ['Microsoft','MSFT', msft_Er, msftEr, msft_sd, msftB]  
df.loc[1] = ['Lululemon','LULU', lulu_Er, luluEr, lulu_sd, luluB]  
df.loc[2] = ['NVIDIA','NVDA', nvda_Er, nvdaEr, nvda_sd, nvdaB]
df.loc[3] = ['Walmart','WMT', wmt_Er, wmtEr, wmt_sd, wmtB]  
df.loc[4] = ['Next Era Energy', 'NEE', nee_Er, neeEr, nee_sd, neeB]  
df.loc[5] = ['Enbridge', 'ENB', enb_Er, enbEr, enb_sd, enbB]
df.loc[6] = ['Canadian Natural Resources Ltd','CNQ', cnq_Er, cnqEr, cnq_sd, cnqB]  
df.loc[7] = ['Royal Bank','RY', ry_Er, ryEr, ry_sd, ryB]
df.loc[8] = ['Bank of Montreal','BMO', bmo_Er, bmoEr, bmo_sd, bmoB]
df.loc[9] = ['IBM', 'IBM', ibm_Er, ibmEr, ibm_sd, ibmB]  
df.loc[10] = ['Netflix','NFLX', nflx_Er, nflxEr, nflx_sd, nflxB]  
df.loc[11] = ['Ford Motor Co', 'F', f_Er, fEr, f_sd, fB]
df.loc[12] = ['Exxon Mobil Corp', 'XOM', xom_Er, xomEr, xom_sd, xomB]  
df.loc[13] = ['British Petroleum PLC','BP', bp_Er, bpEr, bp_sd, bpB]  
df.loc[14] = ['Nike', 'NKE', nke_Er, nkeEr, nke_sd, nkeB]

# Reset the index
df.reset_index(drop=True, inplace=True)


#plot Expected retun vs volitility for all stocks 
palette = sns.color_palette("husl", len(df))

plt.figure(figsize=(8, 6))
plt.scatter(rm_sd, rm_Er, color='red', label='Market Portfolio')
plt.text(rm_sd + 0.002, rm_Er + 0.002, 'Market Return', fontsize=8)
plt.scatter(df['Volatility(%)'], df['Expected Return (Historical)'], alpha=0.5, c=palette)

# Add annotations for each point with the company name
for i, row in df.iterrows():
    plt.text(row['Volatility(%)'] + 0.002, row['Expected Return (Historical)'] + 0.002, row['Ticker'], fontsize=8)

# Set plot title and labels
plt.title('Expected Return vs. Volatility of Selected Stocks')
plt.xlabel('Volatility (Standard Deviation %)')
plt.ylabel('Expected Return (%/year)')
plt.grid(True)
plt.show()

#plot CAPM B vs Expected Return for all stocks 
palette = sns.color_palette("husl", len(df))

plt.figure(figsize=(14, 8))
plt.scatter(rmB, rm_Er, color='red', label='Market Beta')
plt.text(rmB - 0.02, rm_Er - 0.02, 'Market Beta', fontsize=8)
plt.scatter(df['Beta'], df['Expected Return (CAPM)'], alpha=0.5, c=palette)
# Add annotations for each point with the company name
for i, row in df.iterrows():
    plt.annotate(row['Ticker'], (row['Beta'], row['Expected Return (CAPM)']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
# Create x-values for the line
x_values = np.linspace(0,2)
# Calculate y-values (Er) for the line using the intercept and slope
y_values = 5.781043379513968 + ((12.483289909308533-5.781043379513968)/(1-0)) * x_values
# Add a line representing the regression line
plt.plot(x_values, y_values, color='red', linestyle='--', label='Regression Line')
# Set plot title and labels
plt.title('Expected Return vs. CAPM Beta of Selected Stocks')
plt.xlabel('Beta')
plt.ylabel('Expected Return (%/year)')
plt.xlim(0, 1.7) 
plt.grid(True)
plt.show()

# Save DataFrame to a CSV file
df.to_csv('table.csv', index=False)  # Set index=False to exclude row indices

