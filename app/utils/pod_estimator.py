import pandas as pd

def predict_pods(tps, cpu, mem, response, cpu_model, mem_model, cpu_limit, mem_limit):
    max_pods = 50  # configurable max pods
    for pods in range(1, max_pods + 1):
        # Predict total CPU & memory usage for the entire load
        sample = pd.DataFrame([[tps, cpu, mem, response]],
                              columns=["TPS", "CPU_Cores", "Memory_GB", "ResponseTime_sec"])

        total_cpu = cpu_model.predict(sample)[0]
        total_mem = mem_model.predict(sample)[0]

        # Divide resource usage across pods
        cpu_per_pod = total_cpu / pods
        mem_per_pod = total_mem / pods

        if cpu_per_pod <= cpu_limit and mem_per_pod <= mem_limit:
            return pods, cpu_per_pod, mem_per_pod, " Configuration is within acceptable limits."

    # If nothing fits, return max pods tried
    return max_pods, total_cpu / max_pods, total_mem / max_pods, " Configuration exceeds limits. Not recommended for production."
