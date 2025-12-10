import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor
import joblib
import os

PREDICTION_TYPES = ["OAC_W", "OCC_D", "OCC_W"]

PARAMETER_RANGES = {
    "Anode_Copper_content": (99.46, 99.90),
    "Anode_Bismuth_content": (0.0074, 0.0406),
    "Anode_Antimony_content": (0.0161, 0.0534),
    "Anode_Nickel_content": (0.0145, 0.394),
    "Anode_Arsenic_content": (0.0391, 0.190),
    "Current Density": (259.073, 384.352),
    "Inlet Sulfate ion concentration": (161.86, 191.73),
    "Inlet Copper ion concentration": (38.81, 53.65),
    "Inlet Antimony ion concentration": (0.16, 0.27),
    "Inlet Bismuth ion concentration": (0.08, 0.319),
    "Inlet Nickel ion concentration": (5.68, 14.98),
    "Inlet Arsenic ion concentration": (5, 14.82),
    "Electrolyte_Time": (3.33, 334),
    "Inlet Chloride ion concentration": (0.043, 0.072),
    "Number of Electrolysis Tanks": (42, 436)
}

OUTPUT_CONCENTRATION_RANGES = {
    "OAC_W": (6.36, 9.11),
    "OCC_D": (44.35, 50.14),
    "OCC_W": (44.35, 50.14)
}

MODEL_CONFIG = {
    "OAC_W": {
        "model_file": "catboost_OAC_W_model.pkl",
        "scaler_X_file": "scaler_X_OAC_W.pkl",
        "scaler_y_file": "scaler_y_OAC_W.pkl",
        "model_type": "joblib"
    },
    "OCC_D": {
        "model_file": "catboost_OCC_D_model.cbm",
        "scaler_X_file": "scaler_X_OCC_D.pkl",
        "scaler_y_file": "scaler_y_OCC_D.pkl",
        "model_type": "catboost"
    },
    "OCC_W": {
        "model_file": "adaboost_OCC_W_model.pkl",
        "scaler_X_file": "scaler_X_OCC_W.pkl",
        "scaler_y_file": "scaler_y_OCC_W.pkl",
        "model_type": "joblib"
    }
}

PARAMETERS_CONFIG = {
    "OAC_W": {
        "chemical": {
            "Anode_Copper_content": 99.83,
            "Anode_Bismuth_content": 0.0156,
            "Anode_Antimony_content": 0.0255,
            "Anode_Nickel_content": 0.0406,
            "Anode_Arsenic_content": 0.0873
        },
        "physical": {
            "Inlet Sulfate ion concentration": 178.18,
            "Inlet Copper ion concentration": 48.97,
            "Inlet Antimony ion concentration": 0.200,
            "Inlet Bismuth ion concentration": 0.170,
            "Inlet Nickel ion concentration": 10.700,
            "Inlet Chloride ion concentration": 0.057,
            "Inlet Arsenic ion concentration": 8.67
        },
        "experimental": {
            "Current Density": 340.417,
            "Electrolyte_Time": 164,
            "Number of Electrolysis Tanks": 371.67,
        }
    },
    "OCC_D": {
        "chemical": {
            "Anode_Copper_content": 99.83,
            "Anode_Bismuth_content": 0.0226,
            "Anode_Antimony_content": 0.02,
            "Anode_Nickel_content": 0.0271,
            "Anode_Arsenic_content": 0.09
        },
        "physical": {
            "Inlet Sulfate ion concentration": 171.61,
            "Inlet Copper ion concentration": 50.38,
            "Inlet Chloride ion concentration": 0.056
        },
        "experimental": {
            "Current Density": 327.204,
            "Electrolyte_Time": 37.33,
            "Number of Electrolysis Tanks": 415.83,
        }
    },
    "OCC_W": {
        "chemical": {
            "Anode_Copper_content": 99.79,
            "Anode_Bismuth_content": 0.0199,
            "Anode_Antimony_content": 0.0437,
            "Anode_Nickel_content": 0.0392,
            "Anode_Arsenic_content": 0.1015
        },
        "physical": {
            "Inlet Sulfate ion concentration": 168,
            "Inlet Copper ion concentration": 49.33,
            "Inlet Antimony ion concentration": 0.240,
            "Inlet Bismuth ion concentration": 0.093,
            "Inlet Nickel ion concentration": 9.9,
            "Inlet Chloride ion concentration": 0.056,
            "Inlet Arsenic ion concentration": 9.30
        },
        "experimental": {
            "Current Density": 327.015,
            "Electrolyte_Time": 108,
            "Number of Electrolysis Tanks": 269.5,
        }
    }
}

INPUT_LABELS = {
    "Anode_Copper_content": "Anode Copper content (%):",
    "Anode_Bismuth_content": "Anode Bismuth content (%):",
    "Anode_Antimony_content": "Anode Antimony content (%):",
    "Anode_Nickel_content": "Anode Nickel content (%):",
    "Anode_Arsenic_content": "Anode Arsenic content (%):",
    "Current Density": "Current Density (A/m2):",
    "Inlet Sulfate ion concentration": "Inlet Sulfate ion concentration (g/L):",
    "Inlet Copper ion concentration": "Inlet Copper ion concentration (g/L):",
    "Inlet Antimony ion concentration": "Inlet Antimony ion concentration (g/L):",
    "Inlet Bismuth ion concentration": "Inlet Bismuth ion concentration (g/L):",
    "Inlet Nickel ion concentration": "Inlet Nickel ion concentration (g/L):",
    "Inlet Arsenic ion concentration": "Inlet Arsenic ion concentration (g/L):",
    "Electrolyte_Time": "Electrolyte Time (Day):",
    "Inlet Chloride ion concentration": "Inlet Chloride ion concentration (g/L):",
    "Number of Electrolysis Tanks": "Number of Electrolysis Tanks (Set):"
}

PARAMETER_STEPS = {
    "Electrolyte_Time": 0.2,
    "Anode_Copper_content": 0.5,
    "Anode_Bismuth_content": 0.02,
    "Anode_Antimony_content": 0.02,
    "Anode_Nickel_content": 0.5,
    "Anode_Arsenic_content": 0.2,
    "Current Density": 1.0,
    "Inlet Sulfate ion concentration": 5.0,
    "Inlet Copper ion concentration": 0.1,
    "Inlet Antimony ion concentration": 0.1,
    "Inlet Bismuth ion concentration": 0.1,
    "Inlet Nickel ion concentration": 0.1,
    "Inlet Chloride ion concentration": 0.02,
    "Inlet Arsenic ion concentration": 0.1,
    "Number of Electrolysis Tanks": 5.0,
}

INPUT_ORDER = {
    "OAC_W": [
        "Electrolyte_Time",
        "Anode_Copper_content",
        "Anode_Bismuth_content",
        "Anode_Antimony_content",
        "Anode_Nickel_content",
        "Anode_Arsenic_content",
        "Current Density",
        "Inlet Sulfate ion concentration",
        "Inlet Copper ion concentration",
        "Inlet Antimony ion concentration",
        "Inlet Bismuth ion concentration",
        "Inlet Nickel ion concentration",
        "Inlet Chloride ion concentration",
        "Inlet Arsenic ion concentration",
        "Number of Electrolysis Tanks"
    ],
    "OCC_D": [
        "Electrolyte_Time",
        "Anode_Copper_content",
        "Anode_Bismuth_content",
        "Anode_Antimony_content",
        "Anode_Nickel_content",
        "Anode_Arsenic_content",
        "Current Density",
        "Inlet Sulfate ion concentration",
        "Inlet Copper ion concentration",
        "Inlet Chloride ion concentration",
        "Number of Electrolysis Tanks"
    ],
    "OCC_W": [
        "Electrolyte_Time",
        "Anode_Copper_content",
        "Anode_Bismuth_content",
        "Anode_Antimony_content",
        "Anode_Nickel_content",
        "Anode_Arsenic_content",
        "Current Density",
        "Inlet Sulfate ion concentration",
        "Inlet Copper ion concentration",
        "Inlet Antimony ion concentration",
        "Inlet Bismuth ion concentration",
        "Inlet Nickel ion concentration",
        "Inlet Chloride ion concentration",
        "Inlet Arsenic ion concentration",
        "Number of Electrolysis Tanks"
    ]
}


class IntegratedPredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Integrated ML Prediction for electrolyte outlet Copper and Arsenic ion concentration")
        self.root.geometry("1300x900")
        self.root.configure(bg="#2AD4FF")

        self.root.option_add("*TButton.Font", ("Times New Roman", 14, "bold"))
        self.root.option_add("*TLabel.Font", ("Times New Roman", 12))
        self.root.option_add("*TEntry.Font", ("Times New Roman", 12))

        self.models = {pred_type: {"model": None, "scaler_X": None, "scaler_y": None}
                       for pred_type in PREDICTION_TYPES}
        self.input_vars = {}
        self.current_prediction_type = tk.StringVar(value="OAC_W")

        self.load_all_models()
        self.create_widgets()

    def load_all_models(self):
        for pred_type in PREDICTION_TYPES:
            config = MODEL_CONFIG[pred_type]
            self.load_model(pred_type, config["model_file"],
                          config["scaler_X_file"], config["scaler_y_file"],
                          config["model_type"])

    def load_model(self, prediction_type, model_file, scaler_X_file, scaler_y_file, model_type):
        try:
            if not os.path.exists(model_file):
                messagebox.showerror("Error", f"Model file not found for {prediction_type}: {model_file}")
                return

            if model_type == "catboost":
                model = CatBoostRegressor()
                model.load_model(model_file)
            else:
                model = joblib.load(model_file)

            self.models[prediction_type]["model"] = model
            print(f"{prediction_type} model loaded successfully")

            if not (os.path.exists(scaler_X_file) and os.path.exists(scaler_y_file)):
                messagebox.showerror("Error", f"Scaler file not found for {prediction_type}")
                return

            self.models[prediction_type]["scaler_X"] = joblib.load(scaler_X_file)
            self.models[prediction_type]["scaler_y"] = joblib.load(scaler_y_file)
            print(f"{prediction_type} Scaler loaded successfully")

        except Exception as e:
            messagebox.showerror("Error", f"{prediction_type} loading failed: {str(e)}")

    def create_widgets(self):
        self._setup_styles()

        main_frame = ttk.Frame(self.root, padding="20", style="Main.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

        title_frame = ttk.Frame(main_frame, style="Title.TFrame")
        title_frame.pack(fill=tk.X, pady=(0, 20))
        ttk.Label(title_frame,
                 text="Copper and Arsenic ion concentration prediction system",
                 font=("Times New Roman", 22, "bold"),
                 style="Title.TLabel").pack(pady=10)

        pred_type_frame = ttk.Frame(main_frame)
        pred_type_frame.pack(fill=tk.X, pady=10)
        ttk.Label(pred_type_frame, text="Select prediction type:").pack(side=tk.LEFT, padx=10)
        ttk.Combobox(pred_type_frame, textvariable=self.current_prediction_type,
                    values=PREDICTION_TYPES).pack(side=tk.LEFT, padx=10)
        self.current_prediction_type.trace("w", lambda *args: self.update_input_fields())

        self.params_frame = ttk.Frame(main_frame, style="Params.TFrame")
        self.params_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.update_input_fields()

        ttk.Button(main_frame, text="Predict", command=self.predict,
                  style="Predict.TButton").pack(fill=tk.X, padx=5, pady=5)

        result_frame = ttk.Frame(main_frame, style="Result.TFrame")
        result_frame.pack(expand=True, fill=tk.BOTH, pady=20)

        self.result_var = tk.StringVar(value="Prediction results will be displayed here...")
        result_label = ttk.Label(result_frame, textvariable=self.result_var,
                                font=("Times New Roman", 18, "bold"),
                                foreground="#004d99", anchor='center')
        result_label.pack(expand=True, fill=tk.BOTH, pady=10)

        self.range_var = tk.StringVar(value="")
        range_label = ttk.Label(result_frame, textvariable=self.range_var,
                               font=("Times New Roman", 18, "bold"),
                               foreground="#004d99", anchor='center')
        range_label.pack(expand=True, fill=tk.BOTH, pady=10)

    def _setup_styles(self):
        style = ttk.Style()
        bg_color = "#f0f8ff"

        style.configure("Main.TFrame", background=bg_color)
        style.configure("Title.TFrame", background=bg_color)
        style.configure("Params.TFrame", background=bg_color)
        style.configure("Result.TFrame", background=bg_color)
        style.configure("TLabelFrame", background="#F5CB5C", borderwidth=2, relief="groove")

        style.configure("TLabel", background=bg_color, font=("Times New Roman", 10))
        style.configure("Title.TLabel", background=bg_color, foreground="#004d99")
        style.configure("Input.TLabel", background=bg_color)

        style.configure("TButton", font=("Times New Roman", 12), padding=5)
        style.configure("Predict.TButton", font=("Times New Roman", 22, "bold"),
                       background="#0066cc", foreground="#004d99", padding=15, borderwidth=0)
        style.map("Predict.TButton",
                 background=[('active', '#0052a3'), ('pressed', '#003d7a')])
        style.configure("Param.TButton", font=("Times New Roman", 10))

        style.configure("TEntry", padding=8, font=("Times New Roman", 10))
        style.configure("Input.TEntry", padding=8)

    def update_input_fields(self, event=None):
        for widget in self.params_frame.winfo_children():
            widget.destroy()

        prediction_type = self.current_prediction_type.get()
        params = PARAMETERS_CONFIG.get(prediction_type)

        if params:
            self._create_parameter_group(self.params_frame, "Copper anode metal content",
                                        params["chemical"])
            self._create_parameter_group(self.params_frame, "Electrolyte inlet ion concentration",
                                        params["physical"])
            self._create_parameter_group(self.params_frame, "Electrolysis operating conditions",
                                        params["experimental"])

    def _create_parameter_group(self, parent, title, params):
        frame = ttk.LabelFrame(parent, text=title, padding="15")
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        for row, (key, value) in enumerate(params.items()):
            var = tk.DoubleVar(value=value)
            self.input_vars[key] = var
            self._create_input_row(frame, key, var, row)

    def _create_input_row(self, parent, key, var, row):
        label_text = INPUT_LABELS.get(key, key)
        ttk.Label(parent, text=label_text, style="Input.TLabel").grid(
            row=row, column=0, sticky=tk.W, pady=10, padx=5)

        ttk.Entry(parent, textvariable=var, width=12, style="Input.TEntry").grid(
            row=row, column=1, padx=10, pady=10, sticky=tk.W)

        btn_frame = ttk.Frame(parent)
        btn_frame.grid(row=row, column=2, padx=5, pady=10, sticky=tk.W)

        step = PARAMETER_STEPS.get(key, 0.1)
        ttk.Button(btn_frame, text="+", width=3, style="Param.TButton",
                  command=lambda v=var, s=step: self.update_value(v, s)).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="-", width=3, style="Param.TButton",
                  command=lambda v=var, s=step: self.update_value(v, -s)).pack(side=tk.LEFT, padx=2)

    def update_value(self, var, delta):
        current = var.get()
        var.set(current + delta)

    def validate_parameters(self, prediction_type):
        input_order = INPUT_ORDER[prediction_type]
        invalid_params = []

        for key in input_order:
            value = self.input_vars[key].get()
            if key in PARAMETER_RANGES:
                min_val, max_val = PARAMETER_RANGES[key]
                if value < min_val or value > max_val:
                    param_label = INPUT_LABELS.get(key, key)
                    invalid_params.append(
                        f"{param_label}\nCurrent value: {value:.4f}\nAllowed range: {min_val} – {max_val}"
                    )

        return invalid_params

    def validate_output_concentration(self, prediction_type, prediction_value):
        if prediction_type in OUTPUT_CONCENTRATION_RANGES:
            min_val, max_val = OUTPUT_CONCENTRATION_RANGES[prediction_type]
            is_in_range = min_val <= prediction_value <= max_val
            return is_in_range, min_val, max_val
        return True, None, None

    def predict(self):
        prediction_type = self.current_prediction_type.get()
        model_info = self.models[prediction_type]

        if not all([model_info["model"], model_info["scaler_X"], model_info["scaler_y"]]):
            messagebox.showerror("Error", f"{prediction_type} model or scaler not loaded")
            return

        invalid_params = self.validate_parameters(prediction_type)
        if invalid_params:
            error_message = "The following parameter values are out of allowed range:\n\n" + "\n\n".join(invalid_params)
            messagebox.showerror("Parameter Range Error", error_message)
            self.result_var.set("Parameter values out of range, prediction failed")
            self.range_var.set("")
            return

        try:
            input_order = INPUT_ORDER[prediction_type]
            input_values = [self.input_vars[key].get() for key in input_order]
            input_array = np.array([input_values])

            normalized_input = model_info["scaler_X"].transform(input_array)
            normalized_prediction = model_info["model"].predict(normalized_input)[0]
            prediction = model_info["scaler_y"].inverse_transform([[normalized_prediction]])[0][0]
            prediction = max(0, prediction)

            is_in_range, min_val, max_val = self.validate_output_concentration(prediction_type, prediction)

            range_text = f"Optimal concentration range: {min_val} – {max_val} g/L"
            self.range_var.set(range_text)

            if is_in_range:
                result_text = f"✅ {prediction_type} Prediction ion concentration: {prediction:.2f} g/L"
                self.result_var.set(result_text)
            else:
                result_text = f"⚠️ {prediction_type} Prediction ion concentration: {prediction:.2f} g/L (Out of optimal range)"
                self.result_var.set(result_text)
                warning_message = (
                    f"Predicted concentration is out of optimal range!\n\n"
                    f"Predicted value: {prediction:.2f} g/L\n"
                    f"Optimal range: {min_val} – {max_val} g/L"
                )
                messagebox.showwarning("Concentration Range Warning", warning_message)

        except Exception as e:
            messagebox.showerror("Prediction Error", f"Prediction failed: {str(e)}")
            self.range_var.set("")


if __name__ == "__main__":
    root = tk.Tk()
    app = IntegratedPredictor(root)
    root.mainloop()