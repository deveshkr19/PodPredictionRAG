from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def train_models(df):
    features = ["TPS", "CPU_Cores", "Memory_GB", "ResponseTime_sec"]
    cpu_target = "CPU_Load"
    mem_target = "Memory_Load"

    X = df[features]
    y = df[[cpu_target, mem_target]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    cpu_model = LinearRegression().fit(X_train, y_train[cpu_target])
    mem_model = LinearRegression().fit(X_train, y_train[mem_target])

    cpu_r2 = r2_score(y_test[cpu_target], cpu_model.predict(X_test))
    mem_r2 = r2_score(y_test[mem_target], mem_model.predict(X_test))

    print("CPU Model Coefficients:", cpu_model.coef_)
    print("MEM Model Coefficients:", mem_model.coef_)

    return cpu_model, mem_model, cpu_r2, mem_r2
