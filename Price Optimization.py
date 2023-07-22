import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv('sales_data.csv')

# Visualize the relationship between 'price' and 'sales'
plt.scatter(data['price'], data['sales'])
plt.xlabel('Price')
plt.ylabel('Sales')
plt.title('Scatter plot of Price vs. Sales')
plt.show()

X = data[['price']]
y = data['sales']

# Create and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Generate price points for prediction
prices = np.linspace(data['price'].min(), data['price'].max(), num=100).reshape(-1, 1)

# Predict sales for each price point
sales_pred = model.predict(prices)

# Find the price that maximizes sales
optimal_price_index = np.argmax(sales_pred)
optimal_price = prices[optimal_price_index][0]

# Calculate the predicted sales at the optimal price
predicted_sales_at_optimal_price = sales_pred[optimal_price_index]

print(f"Optimal Price: ${optimal_price:.2f}")
print(f"Predicted Sales at Optimal Price: {predicted_sales_at_optimal_price:.2f}")

plt.plot(prices, sales_pred, label='Predicted Sales')
plt.scatter(optimal_price, predicted_sales_at_optimal_price, color='red', label='Optimal Price')
plt.xlabel('Price')
plt.ylabel('Sales')
plt.title('Price Optimization')
plt.legend()
plt.show()

