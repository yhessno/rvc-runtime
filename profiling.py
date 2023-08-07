import cProfile
import pstats
from onnx_inference_demo import cli_infer
import time
import random

out_dir = "profiling"
f0_methods = ["pm", "harvest", "dio", "crepe", "crepe-tiny", "mangio-crepe", "mangio-crepe-tiny", "rmvpe", "hybrid"]
random.shuffle(f0_methods)
cli_infer("Spongebob.pth audios/out.wav audios/spongebob_out.wav logs/added_IVF6717_Flat_nprobe_1_Spongebob_v2.index 0 10 harvest 120 3 0 1 0.78 0.33 false")

execution_times = []
for f0_method in f0_methods:
    profiler = cProfile.Profile()
    profiler.enable()
    start = time.time()
    cli_infer(f"Spongebob.pth audios/out.wav audios/spongebob_out.wav logs/added_IVF6717_Flat_nprobe_1_Spongebob_v2.index 0 10 {f0_method} 120 3 0 1 0.78 0.33 false")
    execution_times.append((f0_method, time.time() - start))
    profiler.disable()
    
    output_file = f"{out_dir}/{f0_method}_results.txt"
    with open(output_file, "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.print_stats()
        print(f"Profiling results exported to {output_file}")

results_out_path = f"{out_dir}/all_f0_runtimes.txt"
sorted_execution_times = sorted(execution_times, key=lambda x: x[1])
with open(results_out_path, "w") as f:
    for execution_time in sorted_execution_times:
        f.write(f"{execution_time[0]}: {execution_time[1]} seconds\n")
        print(f"{execution_time[0]}: {execution_time[1]} seconds")
