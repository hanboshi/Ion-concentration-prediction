import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor
import joblib
import os

# ==================== 常量配置 ====================
PREDICTION_TYPES = ["OAC_W", "OCC_D", "OCC_W"]

# 参数范围配置
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

# 输出浓度范围配置
OUTPUT_CONCENTRATION_RANGES = {
    "OAC_W": (6.36, 9.11),
    "OCC_D": (44.35, 50.14),
    "OCC_W": (44.35, 50.14)
}

# 模型配置
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

# 参数配置
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

# 输入字段标签映射
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

# 参数步长
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

# 预测类型的输入顺序
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
        self.root.configure(bg="#f0f8ff")

        # 设置字体
        self.root.option_add("*TButton.Font", ("Times New Roman", 16, "bold"))
        self.root.option_add("*TLabel.Font", ("Times New Roman", 14))
        self.root.option_add("*TEntry.Font", ("Times New Roman", 14))

        # 初始化模型字典
        self.models = {pred_type: {"model": None, "scaler_X": None, "scaler_y": None}
                       for pred_type in PREDICTION_TYPES}
        self.input_vars = {}
        self.current_prediction_type = tk.StringVar(value="OAC_W")

        # 加载模型和创建界面
        self.load_all_models()
        self.create_widgets()

    def load_all_models(self):
        """加载所有预测类型的模型"""
        for pred_type in PREDICTION_TYPES:
            config = MODEL_CONFIG[pred_type]
            self.load_model(pred_type, config["model_file"],
                          config["scaler_X_file"], config["scaler_y_file"],
                          config["model_type"])

    def load_model(self, prediction_type, model_file, scaler_X_file, scaler_y_file, model_type):
        """加载单个模型和对应的scaler"""
        try:
            if not os.path.exists(model_file):
                messagebox.showerror("错误", f"未找到 {prediction_type} 的模型文件: {model_file}")
                return

            # 加载模型
            if model_type == "catboost":
                model = CatBoostRegressor()
                model.load_model(model_file)
            else:
                model = joblib.load(model_file)

            self.models[prediction_type]["model"] = model
            print(f"{prediction_type} 模型加载成功")

            # 加载scaler
            if not (os.path.exists(scaler_X_file) and os.path.exists(scaler_y_file)):
                messagebox.showerror("错误", f"未找到 {prediction_type} 的scaler文件")
                return

            self.models[prediction_type]["scaler_X"] = joblib.load(scaler_X_file)
            self.models[prediction_type]["scaler_y"] = joblib.load(scaler_y_file)
            print(f"{prediction_type} Scaler加载成功")

        except Exception as e:
            messagebox.showerror("错误", f"{prediction_type} 加载失败: {str(e)}")

    def create_widgets(self):
        """创建主界面"""
        # 设置样式
        self._setup_styles()

        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="20", style="Main.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

        # 创建标题
        title_frame = ttk.Frame(main_frame, style="Title.TFrame")
        title_frame.pack(fill=tk.X, pady=(0, 20))
        ttk.Label(title_frame,
                 text="Copper and Arsenic ion concentration prediction system",
                 font=("Times New Roman", 22, "bold"),
                 style="Title.TLabel").pack(pady=10)

        # 创建预测类型选择
        pred_type_frame = ttk.Frame(main_frame)
        pred_type_frame.pack(fill=tk.X, pady=10)
        ttk.Label(pred_type_frame, text="Select prediction type:").pack(side=tk.LEFT, padx=10)
        ttk.Combobox(pred_type_frame, textvariable=self.current_prediction_type,
                    values=PREDICTION_TYPES).pack(side=tk.LEFT, padx=10)
        self.current_prediction_type.trace("w", lambda *args: self.update_input_fields())

        # 创建参数输入框架
        self.params_frame = ttk.Frame(main_frame, style="Params.TFrame")
        self.params_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # 初始化输入字段
        self.update_input_fields()

        # 预测按钮
        ttk.Button(main_frame, text="Predict", command=self.predict,
                  style="Predict.TButton").pack(fill=tk.X, padx=5, pady=5)

        # 结果显示框架
        result_frame = ttk.Frame(main_frame, style="Result.TFrame")
        result_frame.pack(expand=True, fill=tk.BOTH, pady=20)

        # 预测结果标签
        self.result_var = tk.StringVar(value="Prediction results will be displayed here...")
        result_label = ttk.Label(result_frame, textvariable=self.result_var,
                                font=("Times New Roman", 18, "bold"),
                                foreground="#004d99", anchor='center')
        result_label.pack(expand=True, fill=tk.BOTH, pady=10)

        # 最佳范围标签
        self.range_var = tk.StringVar(value="")
        range_label = ttk.Label(result_frame, textvariable=self.range_var,
                               font=("Times New Roman", 18, "bold"),
                               foreground="#004d99", anchor='center')
        range_label.pack(expand=True, fill=tk.BOTH, pady=10)

    def _setup_styles(self):
        """设置UI样式"""
        style = ttk.Style()
        bg_color = "#f0f8ff"

        # 框架样式
        style.configure("Main.TFrame", background=bg_color)
        style.configure("Title.TFrame", background=bg_color)
        style.configure("Params.TFrame", background=bg_color)
        style.configure("Result.TFrame", background=bg_color)
        style.configure("ParamGroup.TLabelFrame", background="#f0f8ff")

        # 标签样式
        style.configure("TLabel", background=bg_color, font=("Times New Roman", 13))
        style.configure("Title.TLabel", background=bg_color, foreground="#004d99")
        style.configure("Input.TLabel", background=bg_color)

        # 按钮样式
        style.configure("TButton", font=("Times New Roman", 14), padding=5)
        style.configure("Predict.TButton", font=("Times New Roman", 24, "bold"),
                       background="#0066cc", foreground="#004d99", padding=15, borderwidth=0)
        style.map("Predict.TButton",
                 background=[('active', '#0052a3'), ('pressed', '#003d7a')])
        style.configure("Param.TButton", font=("Times New Roman", 12))

        # 输入框样式
        style.configure("TEntry", padding=8, font=("Times New Roman", 13))
        style.configure("Input.TEntry", padding=8)

    def update_input_fields(self, event=None):
        """更新输入字段"""
        # 清空当前的输入字段
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
        """创建参数分组"""
        frame = tk.LabelFrame(parent, text=title, font=("Times New Roman", 16, "bold"),
                             bg="#f0f8ff", padx=15, pady=15)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        for row, (key, value) in enumerate(params.items()):
            var = tk.DoubleVar(value=value)
            self.input_vars[key] = var
            self._create_input_row(frame, key, var, row)

    def _create_input_row(self, parent, key, var, row):
        """创建单个输入行"""
        # 创建标签
        label_text = INPUT_LABELS.get(key, key)
        ttk.Label(parent, text=label_text, style="Input.TLabel").grid(
            row=row, column=0, sticky=tk.W, pady=10, padx=5)

        # 创建输入框
        ttk.Entry(parent, textvariable=var, width=12, style="Input.TEntry").grid(
            row=row, column=1, padx=10, pady=10, sticky=tk.W)

        # 创建增减按钮
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
        """验证输入参数是否在允许范围内"""
        input_order = INPUT_ORDER[prediction_type]
        invalid_params = []

        for key in input_order:
            value = self.input_vars[key].get()
            if key in PARAMETER_RANGES:
                min_val, max_val = PARAMETER_RANGES[key]
                if value < min_val or value > max_val:
                    param_label = INPUT_LABELS.get(key, key)
                    invalid_params.append(
                        f"{param_label}\n当前值: {value:.4f}\n允许范围: {min_val} – {max_val}"
                    )

        return invalid_params

    def validate_output_concentration(self, prediction_type, prediction_value):
        """验证输出浓度是否在允许范围内"""
        if prediction_type in OUTPUT_CONCENTRATION_RANGES:
            min_val, max_val = OUTPUT_CONCENTRATION_RANGES[prediction_type]
            is_in_range = min_val <= prediction_value <= max_val
            return is_in_range, min_val, max_val
        return True, None, None

    def predict(self):
        """执行预测"""
        prediction_type = self.current_prediction_type.get()
        model_info = self.models[prediction_type]

        # 检查模型是否加载
        if not all([model_info["model"], model_info["scaler_X"], model_info["scaler_y"]]):
            messagebox.showerror("错误", f"{prediction_type} 模型或scaler未加载")
            return

        # 验证参数范围
        invalid_params = self.validate_parameters(prediction_type)
        if invalid_params:
            error_message = "以下参数值不在允许范围内:\n\n" + "\n\n".join(invalid_params)
            messagebox.showerror("参数范围错误", error_message)
            self.result_var.set("❌ 参数值超出范围，预测失败")
            self.range_var.set("")
            return

        try:
            # 获取输入值
            input_order = INPUT_ORDER[prediction_type]
            input_values = [self.input_vars[key].get() for key in input_order]
            input_array = np.array([input_values])

            # 预测
            normalized_input = model_info["scaler_X"].transform(input_array)
            normalized_prediction = model_info["model"].predict(normalized_input)[0]
            prediction = model_info["scaler_y"].inverse_transform([[normalized_prediction]])[0][0]
            prediction = max(0, prediction)  # 确保预测值为正

            # 验证输出浓度范围
            is_in_range, min_val, max_val = self.validate_output_concentration(prediction_type, prediction)

            # 显示最佳范围
            range_text = f"Optimal concentration range: {min_val} – {max_val} g/L"
            self.range_var.set(range_text)

            # 显示预测结果
            if is_in_range:
                result_text = f"✅ {prediction_type} Prediction ion concentration: {prediction:.2f} g/L"
                self.result_var.set(result_text)
            else:
                result_text = f"⚠️ {prediction_type} Prediction ion concentration: {prediction:.2f} g/L (Out of optimal range)"
                self.result_var.set(result_text)
                # 显示警告对话框
                warning_message = (
                    f"Predicted concentration is out of optimal range!\n\n"
                    f"Predicted value: {prediction:.2f} g/L\n"
                    f"Optimal range: {min_val} – {max_val} g/L"
                )
                messagebox.showwarning("Concentration Range Warning", warning_message)

        except Exception as e:
            messagebox.showerror("预测错误", f"预测失败: {str(e)}")
            self.range_var.set("")


if __name__ == "__main__":
    root = tk.Tk()
    app = IntegratedPredictor(root)
    root.mainloop()