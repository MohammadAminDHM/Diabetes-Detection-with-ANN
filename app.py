import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tensorflow.keras.models import load_model
import pickle
from sklearn.preprocessing import StandardScaler

# Load the ANN model
ann_model = load_model("models/ann_model.h5")

# Load the StandardScaler
scaler = pickle.load(open("models/standard.pkl", "rb"))

# Create the main application window
app = tk.Tk()
app.title("Diabetes Prediction App")

# Create and place labels and entry widgets for user input
labels = ["Pregnancies",
            "Glucose",
            "Blood Pressure",
            "Skin Thickness",
            "Insulin",
            "BMI",
            "Diabetes Pedigree Function",
            "Age"]
entries = []

for i, label in enumerate(labels):
    label_frame = ttk.Label(app, text=label)
    label_frame.grid(row=i, column=0, padx=10, pady=5)
    entry_frame = ttk.Entry(app)
    entry_frame.grid(row=i, column=1, padx=10, pady=5)
    entries.append(entry_frame)

# Function to make predictions using the ANN model
def predict_ann():
    global entries
    try:
        input_data = []
        for entry in entries:
            input_data.append(float(entry.get()))

        # Standardize the input data
        standardized_input = scaler.transform([input_data])

        # Make the prediction
        prediction = ann_model.predict(standardized_input)

        if prediction[0][0] >= prediction[0][1]:
            result_label_ann.config(text="No Diabetes")
        else:
            result_label_ann.config(text="Diabetes")

    except ValueError:
        messagebox.showerror("Error", "Please enter valid numeric values.")

# Create prediction buttons for ANN models
predict_button_ann = ttk.Button(app, text="Predict (ANN)", command=predict_ann)
predict_button_ann.grid(row=9, column=0, padx=10, pady=10, columnspan=2)

# Create labels to display results
result_label_ann = ttk.Label(app, text="")
result_label_ann.grid(row=11, column=0, columnspan=2)

app.mainloop()
