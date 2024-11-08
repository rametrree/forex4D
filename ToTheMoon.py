import tkinter as tk
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import mplfinance as mpf

# Load the prediction model
model = load_model('C:/Users/ramet/Downloads/model (2).h5')

# Function to load recent data and prepare it for prediction
def load_recent_data():
    df = pd.read_csv('C:/Users/ramet/WebScarping-Forex/data/raw/visualizeData_10min.csv')
    data = df.tail(60)
    data = data[['BO', 'BH', 'BL', 'BC', 'BCh', 'AO', 'AH', 'AL']].values
    data = np.expand_dims(data, axis=0)
    return data

# Function to predict price and display recommendation
def predict_price():
    buy_price_text = entry_price.get()
    try:
        buy_price = float(buy_price_text)
    except ValueError:
        result_label.config(text="Error: Please enter a valid number.", fg="red")
        return

    recent_data = load_recent_data()
    predicted_price = model.predict(recent_data)[0][0]
    current_price = recent_data[0][-1][3]
    predicted_price = predicted_price + 0.0344

    if predicted_price < buy_price and current_price > buy_price:
        result_label.config(text=f"Predicted price: {predicted_price:.5f}\nRecommendation: Sell", fg="green")
    else:
        result_label.config(text=f"Predicted price: {predicted_price:.5f}\nRecommendation: Hold", fg="red")

# Function to provide buy recommendation
def buy_recommendation():
    recent_data = load_recent_data()
    predicted_price = model.predict(recent_data)[0][0]
    past_trend = recent_data[0][:, 3]
    predicted_price = predicted_price + 0.0344
    current_price = recent_data[0][-1][3]

    # if predicted_price > past_trend[-1] and all(x < y for x, y in zip(past_trend, past_trend[1:])):
    if predicted_price > current_price:
        result_label.config(text=f"Predicted price: {predicted_price:.5f}\nRecommendation: Buy", fg="blue")
    else:
        result_label.config(text=f"Predicted price: {predicted_price:.5f}\nRecommendation: Do Not Buy", fg="orange")

# Function to update the candlestick chart in real-time
def update_graph(i):
    df = pd.read_csv('C:/Users/ramet/WebScarping-Forex/data/raw/visualizeData_10min.csv')
    df = df.tail(60)
    df['Date'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.set_index('Date', inplace=True)
    
    # Rename columns to match mplfinance requirements
    df = df.rename(columns={'BO': 'Open', 'BH': 'High', 'BL': 'Low', 'BC': 'Close'})
    
    # Get the last closing price
    current_close = df['Close'].iloc[-1]

    ax.clear()
    
    # Plot candlestick chart with no date labels
    mpf.plot(df, type='candle', ax=ax, style='charles', 
             ylabel='Price', volume=False, xrotation=20,
             warn_too_much_data=60)
    
    # Add a dashed line at the current close price
    ax.axhline(y=current_close, color='red', linestyle='--', linewidth=1)
    
    # Display the current close price on the y-axis
    ax.annotate(f'{current_close}', xy=(1, current_close), xycoords=('axes fraction', 'data'),
                xytext=(5, 0), textcoords='offset points',
                color='red', fontsize=10, ha='left', va='center',
                bbox=dict(facecolor='white', edgecolor='red'))
    
    # Set title and hide x-axis date labels
    ax.set_title("Real-Time Price Chart", color="#333333", fontsize=14, pad=15)
    ax.set_xticks([])  # Remove x-axis labels


# Main window setup
window = tk.Tk()
window.title("AI Forex Prediction")
window.configure(bg="#f5f5f5")

# Frame setup for layout
frame = tk.Frame(window, bg="#f5f5f5")
frame.pack(pady=20, padx=20, fill="both", expand=True)

# Plotting area
fig = Figure(figsize=(8, 4), dpi=100)
ax = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=frame)
canvas.get_tk_widget().pack(pady=10, fill=tk.X)

# Animation for real-time updates
ani = FuncAnimation(fig, update_graph, interval=500)

# Input label and entry for buying price
tk.Label(frame, text="Buying Price:", font=("Helvetica", 12), bg="#f5f5f5", fg="#333333").pack(pady=5)
entry_price = tk.Entry(frame, font=("Helvetica", 12), width=15, justify='center', bd=2, relief="solid")
entry_price.pack(pady=5)

# Predict button
predict_button = tk.Button(frame, text="Predict", command=predict_price, font=("Helvetica", 12), bg="#4CAF50", fg="white", relief="flat", width=10)
predict_button.pack(pady=5)

# Buy recommendation button
buy_button = tk.Button(frame, text="Buy", command=buy_recommendation, font=("Helvetica", 12), bg="#2196F3", fg="white", relief="flat", width=10)
buy_button.pack(pady=5)

# Result label
result_label = tk.Label(frame, text="", font=("Helvetica", 16, "bold"), bg="#f5f5f5", fg="#333333")
result_label.pack(pady=10)

# Run the GUI
window.geometry("700x700")
window.mainloop()
