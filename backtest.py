import yfinance as yf
import numpy as np
from tabulate import tabulate
import pandas as pd
import datetime
import datetime
from tabulate import tabulate
import matplotlib.pyplot as plt




def ema_greater_than_knn(ema, knn_ma):
    if ema *1.01> knn_ma:
        return 1
    else:
        return 0


def calculate_ema(price_values, ema_len):
    ema = np.zeros(len(price_values))
    ema[ema_len-1] = np.mean(price_values[:ema_len])
    multiplier = 2 / (ema_len + 1)
    
    for i in range(ema_len, len(price_values)):
        ema[i] = (price_values[i] - ema[i-1]) * multiplier + ema[i-1]

    return ema


def calculate_knn_ma(price_values, ma_len):
    knn_ma = [np.mean(price_values[i-ma_len:i]) for i in range(ma_len, len(price_values))]
    knn_ma = [0]*ma_len + knn_ma
    return knn_ma


def calculate_knn_prediction(price_values, ma_len, num_closest_values=3, smoothing_period=50):
    def mean_of_k_closest(value, target, num_closest):
        closest_values = []
        for i in range(len(value)):
            distances = [abs(target[i] - v) for v in closest_values]
            if len(distances) < num_closest or min(distances) < min(distances):
                closest_values.append(value[i])
            if len(distances) >= num_closest:
                max_dist_index = distances.index(max(distances))
                if distances[max_dist_index] > min(distances):
                    closest_values[max_dist_index] = value[i]
        return sum(closest_values) / len(closest_values)

    knn_ma = [mean_of_k_closest(price_values[i-ma_len:i], price_values[i-ma_len:i], num_closest_values)
              for i in range(ma_len, len(price_values))]

    if len(knn_ma) < smoothing_period:
        return []

    knn_smoothed = np.convolve(knn_ma, np.ones(smoothing_period) / smoothing_period, mode='valid')

    def knn_prediction(price, knn_ma, knn_smoothed):
        pos_count = 0
        neg_count = 0
        min_distance = 1e10
        nearest_index = 0
        
        # Check if there are enough elements in knn_ma and knn_smoothed
        if len(knn_ma) < 2 or len(knn_smoothed) < 2:
            return 0  # Return 0 for neutral if there aren't enough elements
        
        for j in range(1, min(10, len(knn_ma))):
            distance = np.sqrt((knn_ma[j] - price) ** 2)
            if distance < min_distance:
                min_distance = distance
                nearest_index = j
                
                # Check if there are enough elements to compare
                if nearest_index >= 1:
                    if knn_smoothed[nearest_index] > knn_smoothed[nearest_index - 1]:
                        pos_count += 1
                    if knn_smoothed[nearest_index] < knn_smoothed[nearest_index - 1]:
                        neg_count += 1
        
        return 1 if pos_count > neg_count else -1

    knn_predictions = [knn_prediction(price_values[i], knn_ma[i - smoothing_period:i], knn_smoothed[i - smoothing_period:i])
                       for i in range(smoothing_period, len(price_values))]
    return knn_predictions

sma_len = 9 

def calculate_sma(price_values, sma_len):
    sma = [np.mean(price_values[i-sma_len:i]) for i in range(sma_len, len(price_values))]
    sma = [0]*sma_len + sma
    return sma



def run_program(symbol, start_date, end_date):
    #Ticker Detailss
    historical_data = []
    tradeOpen = False
    buyPrice = 0
    sellPrice = 0
    buyTime = None
    sellTime = None

    buyPriceArray = []
    sellPriceArray = []
    buyTimeArray = []
    sellTimeArray = []
    profitArray = []
    positive = []
    negative = []
    profit_by_year = {}
    capitalArray = []


    ticker = yf.Ticker(symbol)
    #end_date = "2023-07-11"

    interval = "1d"
    data = ticker.history(start=start_date, end=end_date, interval='1d') #could try doing hourly with confirmation on daily or weekly
    if data is None:
        return tradeOpen, 0, 0

    historical_data.append(data)

    

    #Getting weekly data 
    weekly_data = []
    weekData = ticker.history(start=start_date, end=end_date, interval="1wk")
    weekly_data.append(weekData)



    for i in range(len(weekly_data)):
        ma_len = 5
        ema_len_5 = 5
        
        weekly_data[i]['EMA_5'] = calculate_ema(weekly_data[i]['Close'], ema_len_5)
        weekly_data[i]['KNN_MA'] = calculate_knn_ma(weekly_data[i]['Close'], ma_len)

        weeklyTable = []

        for index, row in weekly_data[i].iterrows():
                
                date = index
                open_price = row['Open']
                close_price = row['Close']
                volume = row['Volume']
                ema = row['EMA_5']
                knn_ma = row['KNN_MA']
            
        
        
        
                if ema != None and knn_ma != None:
                    KnnEmaX = ema_greater_than_knn(ema, knn_ma)
                else:
                    KnnEmaX = None
                    ema = None
                    knn_ma = None
        
        
                weeklyTable.append([date, open_price, close_price, volume, ema, knn_ma, KnnEmaX])
    weeklyHeader = ['Date', 'Open', 'Close', 'Volume', 'EMA_5', 'KNN_MA', 'KnnEmaX']


    j = 0
    weeklyKnnEmaX = 0
    weeklyEma = 0
    weeklyKnn = 0
    weekKnnArray = []
    weekEmaArray = []

    for i in range(len(historical_data)):
        ma_len = 5
        ema_len_5 = 9
        historical_data[i]['EMA_5'] = calculate_ema(historical_data[i]['Close'], ema_len_5)
        historical_data[i]['KNN_MA'] = calculate_knn_ma(historical_data[i]['Close'], ma_len)
        historical_data[i]['SMA'] = calculate_sma(historical_data[i]['Close'], sma_len)


        table = []
        capital = 1000

        for index, row in historical_data[i].iterrows():
            
            date = index
            open_price = row['Open']
            close_price = row['Close']
            volume = row['Volume']
            ema = row['EMA_5']
            knn_ma = row['KNN_MA']
            sma = row['SMA']
           

            for weekly_row in weeklyTable:
                if date.date() == weekly_row[0].date():
                    weeklyKnnEmaX = weekly_row[6]
                    weeklyClose = weekly_row[2]
                    weeklyOpen = weekly_row[1]
                    weeklyVolume = weekly_row[3]
                    weeklyEma = weekly_row[4]
                    weeklyKnn = weekly_row[5]
                    weeklyDate = weekly_row[0]
                    break


            if ema != None and knn_ma != None and sma != None:
                KnnEmaX = ema_greater_than_knn(ema, knn_ma)
                TrendConfirmation = ema_greater_than_knn(ema, sma)

            else:
                KnnEmaX = None
                TrendConfirmation = None
     


        # if (KnnEmaX == 1) and (tradeOpen == False) and (TrendConfirmation == 1) and (MACDConverge == 1) and (vwap < close_price * 1.01) and (weeklyKnnEmaX == 1) or (tradeOpen == False and (weeklyKnnEmaX == 1) and (KnnEmaX == 1)):
            if (tradeOpen == False and (weeklyKnnEmaX == 1) and (KnnEmaX == 1)):
                buyPrice = close_price
                #print("Buy Price: ", buyPrice)
                time = date
                tradeOpen = True
            
                
            elif ((KnnEmaX == 0) and (tradeOpen == True) and (TrendConfirmation == 0)) or (tradeOpen == True and (weeklyKnnEmaX == 0)) :
                price = close_price
                time = date
                #print("Sell Price: ", sellPrice)
                tradeOpen = False
                
                


            
            table.append([date, open_price, close_price, volume, ema, knn_ma, KnnEmaX, weeklyEma, weeklyKnn, weeklyKnnEmaX, TrendConfirmation, tradeOpen])

    header = ['Date', 'Open', 'Close', 'Volume', 'EMA_5', 'KNN_MA', 'KnnEmaX', 'WeeklyEMA', 'WeeklyKNN', 'WeeklyKnnEmaX', 'TrendConfirmation', 'TradeOpen']
    output = tabulate(table, headers=header, tablefmt='orgtbl')






    with open("outputTest.txt", "w") as f:
        f.write(output)



    return tradeOpen, close_price, date

    

symbol = "tqqq" #tqqq, tsll
start_date = "2008-11-06"
end_date = "2010-11-20"
tradeTaken = False

buyPriceArray = []
sellPriceArray = []
buyTimeArray = []
sellTimeArray = []
profitArray = []
capitalArray = []
profitTrades = 0
lossTrades = 0

capital = 1000

profitByYear = {}  # Dictionary to store profit by year

while True:

    end_date = (datetime.datetime.strptime(end_date, "%Y-%m-%d") + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        

    tradeOpen, price, date = run_program(symbol, start_date, end_date)
    if tradeOpen == True:
        if not tradeTaken:
            tradeTaken = True
            buyPrice = price
            buyTime = date
            shares = capital / buyPrice
            print("Buy at: ", price, "on: ", date, "Shares: ", shares)
    else:
        if tradeTaken:
            tradeTaken = False
            sellPrice = price
            sellTime = date
            profit = sellPrice - buyPrice
            buyPriceArray.append(buyPrice)
            sellPriceArray.append(sellPrice)
            buyTimeArray.append(buyTime)
            sellTimeArray.append(sellTime)
            profitArray.append(profit)
            capital = shares * sellPrice
            capitalArray.append(capital)
            print("Sell at: ", price, "on: ", date, "Profit: ", profit, "Capital: ", capital)
            
            # Calculate the year of the sellTime
            year = datetime.datetime.strptime(sellTime.strftime("%Y-%m-%d"), "%Y-%m-%d").year
            if year in profitByYear:
                profitByYear[year] += profit
            else:
                profitByYear[year] = profit


            if profit > 0:
                profitTrades += 1
            else:
                lossTrades += 1

    print(date, tradeOpen, price)
    

    if end_date == "2023-11-17":
        break



# Print table of trade arrays
table = {
    "Buy Price": buyPriceArray,
    "Sell Price": sellPriceArray,
    "Buy Time": buyTimeArray,
    "Sell Time": sellTimeArray,
    "Profit": profitArray,
    "Capital": capitalArray
}

print(tabulate(table, headers="keys"))

# Print profit by year
print("Profit by Year:")
for year, profit in profitByYear.items():
    print(f"Year: {year}, Profit: {profit}")

# Print number of profit and loss trades
print(f"Profit Trades: {profitTrades}, Loss Trades: {lossTrades}, Percentage: {profitTrades / (profitTrades + lossTrades)}")