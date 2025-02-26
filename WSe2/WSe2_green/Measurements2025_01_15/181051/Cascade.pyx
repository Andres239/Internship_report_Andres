
import pickle
import matplotlib.pyplot as plt

# Load the figure from the file
with open(r'C:\Users\Administrateur\Desktop\Sotos\\Measurements2025_01_15\181051\Cascade.pkl', 'rb') as f:
    fig = pickle.load(f)

# Show the figure
fig.show()
plt.show()  # Keep the plot open until the user closes it
