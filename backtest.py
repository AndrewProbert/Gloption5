import yfinance as yf
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from tabulate import tabulate
 



def ema_greater_than_knn(ema, knn_ma):
    if ema *1> knn_ma:
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



def calcMACD(data, short_period=12, long_period=26, signal_period=9):
    short_ema = calcEMA(data, short_period)
    long_ema = calcEMA(data, long_period)

    macd = [short - long for short, long in zip(short_ema, long_ema)]

    signal = calcEMA(macd, signal_period)

    return macd, signal

def calcEMA(data, period):
    multiplier = 2 / (period + 1)
    ema = [data[0]]

    for i in range(1, len(data)):
        ema_val = (data[i] - ema[-1]) * multiplier + ema[-1]
        ema.append(ema_val)

    return ema    

def macdCross(macd, signal):
    if macd > signal:
        return 1
    else:
        return 0

def calculate_rsi(price_values, period=14):
    delta = np.diff(price_values)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = np.zeros_like(price_values)
    avg_loss = np.zeros_like(price_values)

    # Calculate average gain and loss for the initial period
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])

    for i in range(period + 1, len(price_values)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period

        # Check for zero division and NaN values
        if avg_loss[i] == 0 or np.isnan(avg_loss[i]):
            rs = 0 if avg_gain[i] == 0 else np.inf
        else:
            rs = avg_gain[i] / avg_loss[i]

        rsi = 100 - (100 / (1 + rs))

    return rsi


def rsiEMA(rsi, rsi_ema):
    if rsi > rsi_ema:
        return 1
    else:
        return 0
    


def run_program(symbol, start_date, end_date, saveFile):
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

    
    weeklyEnd = (datetime.datetime.strptime(end_date, "%Y-%m-%d") - datetime.timedelta(days=1)).strftime("%Y-%m-%d")    
    #Getting weekly data 
    weekly_data = []
    weekData = ticker.history(start=start_date, end=weeklyEnd, interval="1wk")
    weekly_data.append(weekData)



    for i in range(len(weekly_data)):
        ma_len = 2
        ema_len_5 = 2
        
        weekly_data[i]['EMA_5'] = calculate_ema(weekly_data[i]['Close'], ema_len_5)
        weekly_data[i]['KNN_MA'] = calculate_knn_ma(weekly_data[i]['Close'], ma_len)
        weekly_data[i]['RSI'] = calculate_rsi(weekly_data[i]['Close'], period=14)
        weekly_data[i]['RSI_EMA_14'] = calculate_ema(weekly_data[i]['RSI'], 14)


        weeklyTable = []

        for index, row in weekly_data[i].iterrows():
                
                date = index
                open_price = row['Open']
                close_price = row['Close']
                volume = row['Volume']
                ema = row['EMA_5']
                knn_ma = row['KNN_MA']
                rsi = row['RSI']
                rsi_ema = row['RSI_EMA_14']


            
        
        
        
                if ema != None and knn_ma != None and rsi != None and rsi_ema != None:
                    KnnEmaX = ema_greater_than_knn(ema, knn_ma)
                    RsiEmaX = rsiEMA(rsi, rsi_ema)

                else:
                    KnnEmaX = None
                    ema = None
                    knn_ma = None
                    RsiEmaX = None


            
        
        
                weeklyTable.append([date, open_price, close_price, volume, ema, knn_ma, KnnEmaX, rsi, rsi_ema, RsiEmaX])
    weeklyHeader = ['Date', 'Open', 'Close', 'Volume', 'EMA_5', 'KNN_MA', 'KnnEmaX']


    j = 0
    weeklyKnnEmaX = 0
    weeklyEma = 0
    weeklyKnn = 0
    weekKnnArray = []
    weekEmaArray = []
    weeklyClose = 0
    weeklyRsiEmaX = 0
    previousPrice = 0

    for i in range(len(historical_data)):
        ma_len = 9
        ema_len_5 = 5
        historical_data[i]['EMA_5'] = calculate_ema(historical_data[i]['Close'], ema_len_5)
        historical_data[i]['KNN_MA'] = calculate_ema(historical_data[i]['Close'], ma_len) #Replace with regular ema
        historical_data[i]['SMA'] = calculate_sma(historical_data[i]['Close'], sma_len)
        historical_data[i]['MACD'], historical_data[i]['Signal'] = calcMACD(historical_data[i]['Close'])
        historical_data[i]['RSI'] = calculate_rsi(historical_data[i]['Close'], period=14)
        historical_data[i]['RSI_EMA_14'] = calculate_ema(historical_data[i]['RSI'], 14)





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
            MACD = row['MACD']
            Signal = row['Signal']
            rsi = row['RSI']
            rsi_ema = row['RSI_EMA_14']


       


           

            for weekly_row in weeklyTable:
                if date.date() == weekly_row[0].date():
                    weeklyKnnEmaX = weekly_row[6]
                    weeklyClose = weekly_row[2]
                    weeklyOpen = weekly_row[1]
                    weeklyVolume = weekly_row[3]
                    weeklyEma = weekly_row[4]
                    weeklyKnn = weekly_row[5]
                    weeklyDate = weekly_row[0]
                    weeklyRsi = weekly_row[7]
                    weeklyRsiEma = weekly_row[8]
                    weeklyRsiEmaX = weekly_row[9]


                    break


            if ema != None and knn_ma != None and sma != None and MACD != None and Signal != None and rsi != None and rsi_ema != None:
                KnnEmaX = ema_greater_than_knn(ema, knn_ma)
                TrendConfirmation = ema_greater_than_knn(ema, sma)
                MACDConverge = macdCross(MACD, Signal)
                RsiEmaX = rsiEMA(rsi, rsi_ema)



            else:
                KnnEmaX = None
                TrendConfirmation = None
                MACDConverge = None
                RsiEmaX = None

            
     


        # if (KnnEmaX == 1) and (tradeOpen == False) and (TrendConfirmation == 1) and (MACDConverge == 1) and (vwap < close_price * 1.01) and (weeklyKnnEmaX == 1) or (tradeOpen == False and (weeklyKnnEmaX == 1) and (KnnEmaX == 1)):
            if (tradeOpen == False  and KnnEmaX == 0 and weeklyKnnEmaX == 1 and weeklyRsiEmaX == 1 and RsiEmaX == 0):
                
                buyPrice = close_price
                #print("Buy Price: ", buyPrice)
                time = date
                tradeOpen = True
            
                
            elif (tradeOpen == False  or KnnEmaX == 1 or weeklyKnnEmaX == 0 or weeklyRsiEmaX == 0 or RsiEmaX == 1):
                price = close_price
                time = date
                #print("Sell Price: ", sellPrice)
                tradeOpen = False
                            


            previousPrice = close_price


            
            table.append([date, open_price, close_price, volume, ema, knn_ma, KnnEmaX, weeklyEma, weeklyKnn, weeklyKnnEmaX, TrendConfirmation, tradeOpen, weeklyClose])

    header = ['Date', 'Open', 'Close', 'Volume', 'EMA_5', 'KNN_MA', 'KnnEmaX', 'WeeklyEMA', 'WeeklyKNN', 'WeeklyKnnEmaX', 'TrendConfirmation', 'TradeOpen', 'WeeklyClose']
    output = tabulate(table, headers=header, tablefmt='orgtbl')





    
    with open("outputTest.txt", "w") as f:
        f.write(output)
    


    return tradeOpen, close_price, date

    

symbol = "tqqq" #soxl, soxs, boil, uvxy
start_date = "2020-11-06"
end_date = "2021-11-01"
final_date = "2023-11-27"
saveFile = symbol + "_" + end_date + "_" + final_date + ".txt"
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


    '''
    if input():
        continue   
    '''
    

    end_date = (datetime.datetime.strptime(end_date, "%Y-%m-%d") + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        

    tradeOpen, price, date = run_program(symbol, start_date, end_date, saveFile)
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
    

    if end_date == final_date:
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


with open(saveFile, "w") as f:
    f.write(tabulate(table, headers="keys"))
    print(tabulate(table, headers="keys"))

# Print profit by year
with open(saveFile, "a") as f:
    f.write("\n\nProfit by Year:\n")
    print("Profit by Year:")
    for year, profit in profitByYear.items():
        f.write(f"Year: {year}, Profit: {profit}\n")
        print(f"Year: {year}, Profit: {profit}")

# Print number of profit and loss trades
with open(saveFile, "a") as f:
    f.write(f"\nProfit Trades: {profitTrades}, Loss Trades: {lossTrades}, Percentage: {profitTrades / (profitTrades + lossTrades)}\n")
    print(f"Profit Trades: {profitTrades}, Loss Trades: {lossTrades}, Percentage: {profitTrades / (profitTrades + lossTrades)}")

average_return = np.mean(profitArray)
standard_deviation = np.std(profitArray)
downside_returns = [r for r in profitArray if r < 0]
downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0

sharpe_ratio = average_return / standard_deviation
sortino_ratio = average_return / downside_deviation if downside_deviation > 0 else 0

with open(saveFile, "a") as f:
    f.write("\nSharpe Ratio: " + str(sharpe_ratio) + "\n")
    f.write("Sortino Ratio: " + str(sortino_ratio) + "\n")
    print("Sharpe Ratio: " + str(sharpe_ratio))
    print("Sortino Ratio: " + str(sortino_ratio))
